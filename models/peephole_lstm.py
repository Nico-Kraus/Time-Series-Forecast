import torch
import torch.nn as nn
import torch.nn.init as init
import math

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, init_method='kaiming'):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define the weights and biases
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        # Initialize the weights and biases
        self.init_weights(init_method)

    def init_weights(self, method):
        if method == 'normal':
            init.normal_(self.weight_ih)
            init.normal_(self.weight_hh)
            init.normal_(self.bias_ih)
            init.normal_(self.bias_hh)
        elif method == 'kaiming':
            init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
            init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
            init.zeros_(self.bias_ih)
            init.zeros_(self.bias_hh)
        elif method == 'xavier':
            init.xavier_uniform_(self.weight_ih)
            init.xavier_uniform_(self.weight_hh)
            init.zeros_(self.bias_ih)
            init.zeros_(self.bias_hh)
        elif method == 'uniform':
            std = 1.0 / math.sqrt(self.hidden_size)
            init.uniform_(self.weight_ih, -std, std)
            init.uniform_(self.weight_hh, -std, std)
            init.uniform_(self.bias_ih, -std, std)
            init.uniform_(self.bias_hh, -std, std)
        else:
            raise ValueError("Unknown initialization method")

    def forward(self, x, init_states=None):
        """
        Forward pass through the LSTM cell for a batch of inputs.
        
        Args:
            x: The input tensor for the current time step. Shape: (batch_size, input_size)
            init_states: A tuple of the initial hidden and cell states. Each of shape: (batch_size, hidden_size)

        Returns:
            Tuple of the output and the new hidden and cell states.
        """
        # Initialize hidden and cell states if not provided
        h_t, c_t = (torch.zeros(x.size(0), self.hidden_size).to(x.device),
                    torch.zeros(x.size(0), self.hidden_size).to(x.device)) if init_states is None else init_states

        # Extracting gate-specific weights from combined weight matrices
        weight_if, weight_ii, weight_io, weight_ig = self.weight_ih.chunk(4, 0)
        weight_cf, weight_ci, weight_co, weight_cg = self.weight_hh.chunk(4, 0)
        bias_if, bias_ii, bias_io, bias_ig = self.bias_ih.chunk(4, 0)
        bias_cf, bias_ci, bias_co, bias_cg = self.bias_hh.chunk(4, 0)

        f_t = torch.sigmoid(torch.mm(x, weight_if.t()) + bias_if + torch.mm(c_t, weight_cf.t()) + bias_cf)
        i_t = torch.sigmoid(torch.mm(x, weight_ii.t()) + bias_ii + torch.mm(c_t, weight_ci.t()) + bias_ci)
        o_t = torch.sigmoid(torch.mm(x, weight_io.t()) + bias_io + torch.mm(c_t, weight_co.t()) + bias_co)
        g_t = torch.tanh(torch.mm(x, weight_ig.t()) + bias_ig)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * c_t

        return h_t, c_t

    
class CustomLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, init_method='kaiming', batch_first=False):
        super(CustomLSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.lstm_cell = CustomLSTMCell(input_size, hidden_size, init_method)

    def forward(self, x, init_states=None):
        """
        Forward pass through the LSTM layer for a sequence of inputs.
        
        Args:
            x: The input sequence tensor. Shape: (sequence_length, batch_size, input_size)
            init_states: A tuple of the initial hidden and cell states. Each of shape: (batch_size, hidden_size)

        Returns:
            outputs: All hidden states for the sequence. Shape: (sequence_length, batch_size, hidden_size)
            (h_n, c_n): The final hidden and cell states. Shape: (batch_size, hidden_size) each
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        # Initialize hidden and cell states if not provided
        h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device),
                    torch.zeros(batch_size, self.hidden_size).to(x.device)) if init_states is None else init_states

        outputs = []

        for t in range(seq_len):
            h_t, c_t = self.lstm_cell(x[t, :, :], (h_t, c_t))
            outputs.append(h_t)

        # Stack outputs (hidden states) for each time step into a single tensor
        outputs = torch.stack(outputs, dim=0)

        return outputs, (h_t, c_t)
    
def init_stacked_lstm(num_layers, layer, input_size, hidden_size, init_method, batch_first):
    # The first layer's input size is input_size; for others, it's hidden_size
    layers = [layer(input_size, hidden_size, init_method, batch_first)] + [layer(hidden_size, hidden_size, init_method, False) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class StackedLSTM(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, init_method, batch_first):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(num_layers, CustomLSTMLayer, input_size, hidden_size, init_method, batch_first)

    def forward(self, x, states):
        output_states = []
        output = x

        for i, lstm_layer in enumerate(self.layers):
            state = states[i]
            output, state = lstm_layer(output, state)
            output_states.append(state)

        return output, output_states


class PeepholeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, init_method="kaiming"):
        super(PeepholeLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = StackedLSTM(input_dim, hidden_dim, n_layers, init_method, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        states = [(h0[i].detach(), c0[i].detach()) for i in range(self.n_layers)]

        # Forward pass through Stacked LSTM
        lstm_out, states = self.lstm(x, states)

        # Use the last hidden state of the last layer
        last_hidden_state = states[-1][0]

        # Fully connected layer
        out = self.fc(last_hidden_state)

        return out