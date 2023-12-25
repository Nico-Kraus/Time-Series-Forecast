# import sys
import gzip
# import math
import warnings
    
import numpy as np
import pandas as pd
import EntropyHub as EH

from itertools import permutations
from scipy.signal import welch
from scipy.stats import entropy
from tslearn.piecewise import SymbolicAggregateApproximation
from numba.core.errors import NumbaWarning
import statsmodels.api as sm
from scipy.fft import fft
# from hurst import compute_Hc

def sample_entropy(data_numpy, m, tau):
    Samp, Phi1, Phi2 = EH.SampEn(data_numpy, m=m, **({} if tau == 0 else {"tau": tau}))
    mod_Sample = [0 if np.isinf(s) or np.isnan(s) else s for s in Samp]
    return sum(mod_Sample) / len(mod_Sample)

def mean_absolute_difference(data_numpy):
    abs_diff_mean = np.mean(np.abs(np.diff(data_numpy)))
    return abs_diff_mean

def num_edges(data_numpy):
    differences = np.diff(data_numpy)
    sign_changes = np.sign(differences[:-1]) != np.sign(differences[1:])
    num_edges = np.sum(sign_changes)
    normalized_edges = num_edges / (len(data_numpy) - 1)
    return normalized_edges

def compression_complexity(data_numpy, sax_word_size=10, sax_alphabet_size=4, div=64):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaWarning)
        # Apply SAX
        sax = SymbolicAggregateApproximation(n_segments=sax_word_size, alphabet_size_avg=sax_alphabet_size)
        sax_ts = sax.inverse_transform(sax.fit_transform(data_numpy.reshape(1, -1, 1)))

        # Convert to bytes
        sax_bytes = sax_ts.tostring()

        # Compress using gzip
        compressed_data = gzip.compress(sax_bytes)

    # Return the size of the compressed data in bytes
    return len(compressed_data)/div

def normalized_permutation_entropy(data_numpy, n):
    # Generate all possible permutations of natural numbers from 0 to n-1
    perms = list(permutations(range(n)))
    perm_counts = {perm: 0 for perm in perms}
    
    total_windows = len(data_numpy) - n + 1

    # Count occurrences of each permutation pattern
    for i in range(total_windows):
        window = data_numpy[i:i + n]
        sorted_indices = tuple(np.argsort(window))
        if sorted_indices in perm_counts:
            perm_counts[sorted_indices] += 1

    # Calculate the probabilities of each permutation pattern
    probabilities = [count / total_windows for count in perm_counts.values()]

    # Compute the entropy
    return entropy(probabilities, base=2) / (n - 1)

def average_normalized_permutation_entropy(data_numpy, max_n):
    entropies = [normalized_permutation_entropy(data_numpy, n) for n in range(3, max_n + 1)]
    return np.mean(entropies)

def noise(data_numpy, model='additive'):
    estimated_freq = estimate_seasonal_frequency(data_numpy)
    period = min(int(round(estimated_freq)), len(data_numpy) // 2)
    series = pd.Series(data_numpy)
    decomposition = sm.tsa.seasonal_decompose(series, model=model, period=period)
    residual = decomposition.resid.dropna()
    complexity = np.std(residual)
    return complexity

def estimate_seasonal_frequency(time_series):
    # Apply Fourier Transform
    N = len(time_series)
    f = fft(time_series)
    frequencies = np.abs(f[:N // 2])

    # Find the peak frequency
    peak_frequency = np.argmax(frequencies[1:]) + 1  # Ignore the zero frequency
    estimated_freq = N / peak_frequency

    return estimated_freq

def trend_complexity(data_numpy, model='additive'):
    estimated_freq = estimate_seasonal_frequency(data_numpy)
    period = min(int(round(estimated_freq)), len(data_numpy) // 2)
    series = pd.Series(data_numpy)
    decomposition = sm.tsa.seasonal_decompose(series, model=model, period=period)
    trend = decomposition.trend.dropna()

    complexity = np.std(np.diff(trend))
    return complexity

def mean_trend(data_numpy, model='additive'):
    estimated_freq = estimate_seasonal_frequency(data_numpy)
    period = min(int(round(estimated_freq)), len(data_numpy) // 2)
    series = pd.Series(data_numpy)
    decomposition = sm.tsa.seasonal_decompose(series, model=model, period=period)
    trend = decomposition.trend.dropna()

    complexity = np.mean(trend)
    return complexity

def period_complexity(data_numpy, model='additive'):
    estimated_freq = estimate_seasonal_frequency(data_numpy)
    period = min(int(round(estimated_freq)), len(data_numpy) // 2)
    series = pd.Series(data_numpy)
    decomposition = sm.tsa.seasonal_decompose(series, model=model, period=period)
    seasonal = decomposition.seasonal.dropna()

    complexity = np.std(seasonal)
    return complexity


def spectral_entropy_fft(time_series):
    # Apply Fourier Transform
    N = len(time_series)
    freqs = fft(time_series)
    power_spectrum = np.abs(freqs)[:N // 2]  # Take only the positive frequencies

    # Normalize the power spectrum
    power_spectrum_normalized = power_spectrum / np.sum(power_spectrum)

    # Calculate the entropy of the power spectrum
    spectral_entropy = entropy(power_spectrum_normalized)

    return spectral_entropy

def spectral_entropy_psd(time_series):
    # Calculate the Power Spectral Density
    freqs, psd = welch(time_series)

    # Normalize the PSD
    psd_normalized = psd / np.sum(psd)

    # Calculate the entropy of the normalized PSD
    spectral_entropy = entropy(psd_normalized)

    return spectral_entropy

def period(data_numpy):
    estimated_freq = estimate_seasonal_frequency(data_numpy)
    complexity = estimated_freq / len(data_numpy)
    return complexity



characteristics = {
    "sample_entropy": sample_entropy,
    "mad": mean_absolute_difference,
    "num_edges": num_edges,
    "compression": compression_complexity,
    "perm_entropy": average_normalized_permutation_entropy,
    "noise": noise,
    "mean_trend": mean_trend,
    "trend_complexity": trend_complexity,
    "period": period,
    "period_complexity": period_complexity,
    "spectral_entropy_psd": spectral_entropy_psd,
    "spectral_entropy_fft": spectral_entropy_fft
}

