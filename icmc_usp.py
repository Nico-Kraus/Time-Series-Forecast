import yaml

data = """
01.D 	Fourier A: Constant Level 	Synthetic 		790 	25 	📥 	🔍
02.D 	Fourier A: Increasing Trend 	Synthetic 		790 	25 	📥 	🔍
03.D 	Fourier A: Decreasing Trend 	Synthetic 		790 	25 	📥 	🔍
04.D 	Fourier B: Constant Level 	Synthetic 		790 	38 	📥 	🔍
05.D 	Fourier B: Increasing Trend 	Synthetic 		790 	38 	📥 	🔍
06.D 	Fourier B: Decreasing Trend 	Synthetic 		790 	38 	📥 	🔍
07.D 	Fourier C: Constant Level 	Synthetic 		790 	34 	📥 	🔍
08.D 	Fourier C: Increasing Trend 	Synthetic 		790 	34 	📥 	🔍
09.D 	Fourier C: Decreasing Trend 	Synthetic 		790 	34 	📥 	🔍
10.D 	Fourier D: Constant Level 	Synthetic 		790 	38 	📥 	🔍
11.D 	Fourier D: Increasing Trend 	Synthetic 		790 	38 	📥 	🔍
12.D 	Fourier D: Decreasing Trend 	Synthetic 		790 	38 	📥 	🔍
13.D 	Seasonal Dependence: Constant Level 	Synthetic 		2200 	25 	📥 	🔍
14.D 	Seasonal Dependence: Increasing Trend 	Synthetic 		2200 	25 	📥 	🔍
15.D 	Seasonal Dependence: Decreasing Trend 	Synthetic 		2200 	25 	📥 	🔍
16.D 	Multiplicative Seasonality 	Synthetic 		590 	14 	📥 	🔍
17.D 	High Frequency 	Synthetic 		550 	63 	📥 	🔍
18.S 	CCA: Constant Level 	Synthetic 		1000 	12 	📥 	🔍
19.S 	CCA: Seasonal Patterns 	Synthetic 		1000 	30 	📥 	🔍
20.S 	CCA: Increasing Trend 	Synthetic 		1000 	12 	📥 	🔍
21.S 	CCA: Decreasing Trend 	Synthetic 		1000 	12 	📥 	🔍
22.S 	CCA: Upward Shift 	Synthetic 		1000 	12 	📥 	🔍
23.S 	CCA: Downward Shift 	Synthetic 		1000 	12 	📥 	🔍
24.S 	CCB: Constant Level 	Synthetic 		1000 	30 	📥 	🔍
25.S 	CCB: Double Seasonality 	Synthetic 		1000 	30 	📥 	🔍
26.S 	CCB: Increasing Trend 	Synthetic 		1000 	30 	📥 	🔍
27.S 	CCB: Decreasing Trend 	Synthetic 		1000 	30 	📥 	🔍
28.S 	CCB: Upward Shift 	Synthetic 		1000 	30 	📥 	🔍
29.S 	CCB: Downward Shift 	Synthetic 		1000 	30 	📥 	🔍
30.S 	SDN: Constant Level 	Synthetic 		2200 	25 	📥 	🔍
31.S 	SDN: Increasing Trend 	Synthetic 		2200 	25 	📥 	🔍
32.S 	SDN: Decreasing Trend 	Synthetic 		2200 	25 	📥 	🔍
33.C 	Logistic Map 	Synthetic 		550 	4 	📥 	🔍
34.C 	Hénon Map 	Synthetic 		3000 	3 	📥 	🔍
35.C 	Mackey-Glass System 	Synthetic 		3000 	7 	📥 	🔍
36.C 	Lorenz System 	Synthetic 		3000 	25 	📥 	🔍
37.C 	Rössler System 	Synthetic 		3000 	14 	📥 	🔍
38.C 	Chaotic Signals: A 	Synthetic 		550 	22 	📥 	🔍
39.C 	Chaotic Signals: B 	Synthetic 		550 	7 	📥 	🔍
40.C 	ECGSYN 	Synthetic 		3000 	60 	📥 	🔍
41.A 	Fortaleza 	Real 	Annual 	149 	6 	📥 	🔍
42.A 	Manchas 	Real 	Annual 	176 	11 	📥 	🔍
43.A 	Super Bowl 	Real 	Annual 	22 		📥 	🔍
44.D 	Atmosfera: Temperatura 	Real 	Daily 	365 	7 	📥 	🔍
45.D 	Atmosfera: Umidade Relativa do Ar 	Real 	Daily 	365 	7 	📥 	🔍
46.D 	Banespa 	Real 	Daily 	1499 	7 	📥 	🔍
47.D 	CEMIG 	Real 	Daily 	1499 	7 	📥 	🔍
48.D 	IBV 	Real 	Daily 	1499 	7 	📥 	🔍
49.D 	Patient Demand 	Real 	Daily 	821 	7 	📥 	🔍
50.D 	Petrobras 	Real 	Daily 	1499 	7 	📥 	🔍
51.D 	Poluição: PM10 	Real 	Daily 	365 	7 	📥 	🔍
52.D 	Poluição: SO2 	Real 	Daily 	365 	7 	📥 	🔍
53.D 	Poluição: CO 	Real 	Daily 	365 	7 	📥 	🔍
54.D 	Poluição: O3 	Real 	Daily 	365 	7 	📥 	🔍
55.D 	Poluição: NO2 	Real 	Daily 	365 	7 	📥 	🔍
56.D 	Star 	Real 	Daily 	600 	7 	📥 	🔍
57.D 	Stock Market: Amsterdam 	Real 	Daily 	3128 	7 	📥 	🔍
58.D 	Stock Market: Frankfurt 	Real 	Daily 	3128 	7 	📥 	🔍
59.D 	Stock Market: London 	Real 	Daily 	3128 	7 	📥 	🔍
60.D 	Stock Market: Hong Kong 	Real 	Daily 	3128 	7 	📥 	🔍
61.D 	Stock Market: Japan 	Real 	Daily 	3128 	7 	📥 	🔍
62.D 	Stock Market: Singapore 	Real 	Daily 	3128 	7 	📥 	🔍
63.D 	Stock Market: New York 	Real 	Daily 	3128 	7 	📥 	🔍
64.D 	Truck 	Real 	Daily 	45 	7 	📥 	🔍
65.M 	Bebida 	Real 	Monthly 	187 	12 	📥 	🔍
66.M 	CBE: Chocolate 	Real 	Monthly 	396 	12 	📥 	🔍
67.M 	CBE: Beer 	Real 	Monthly 	396 	12 	📥 	🔍
68.M 	CBE: Electricity Production 	Real 	Monthly 	396 	12 	📥 	🔍
69.M 	Chicken 	Real 	Monthly 	187 	12 	📥 	🔍
70.M 	Consumo 	Real 	Monthly 	154 	12 	📥 	🔍
71.M 	Darwin 	Real 	Monthly 	1400 	12 	📥 	🔍
72.M 	Dow Jones 	Real 	Monthly 	641 	12 	📥 	🔍
73.M 	Energia 	Real 	Monthly 	141 	12 	📥 	🔍
74.M 	Global 	Real 	Monthly 	1800 	12 	📥 	🔍
75.M 	ICV 	Real 	Monthly 	126 	12 	📥 	🔍
76.M 	IPI 	Real 	Monthly 	187 	12 	📥 	🔍
77.M 	Latex 	Real 	Monthly 	199 	12 	📥 	🔍
78.M 	Lavras 	Real 	Monthly 	384 	12 	📥 	🔍
79.M 	Maine 	Real 	Monthly 	128 	12 	📥 	🔍
80.M 	MPrime 	Real 	Monthly 	707 	12 	📥 	🔍
81.M 	OSVisit 	Real 	Monthly 	228 	12 	📥 	🔍
82.M 	Ozônio 	Real 	Monthly 	180 	12 	📥 	🔍
83.M 	PFI 	Real 	Monthly 	115 	12 	📥 	🔍
84.M 	Reservoir 	Real 	Monthly 	864 	12 	📥 	🔍
85.M 	STemp 	Real 	Monthly 	1896 	12 	📥 	🔍
86.M 	Temperatura: Cananéia 	Real 	Monthly 	120 	12 	📥 	🔍
87.M 	Temperatura: Ubatuba 	Real 	Monthly 	120 	12 	📥 	🔍
88.M 	USA 	Real 	Monthly 	130 	12 	📥 	🔍
89.M 	Wine: Fortified White 	Real 	Monthly 	187 	12 	📥 	🔍
90.M 	Wine: Dry White 	Real 	Monthly 	187 	12 	📥 	🔍
91.M 	Wine: Sweet White 	Real 	Monthly 	187 	12 	📥 	🔍
92.M 	Wine: Red 	Real 	Monthly 	187 	12 	📥 	🔍
93.M 	Wine: Rose 	Real 	Monthly 	187 	12 	📥 	🔍
94.M 	Wine: Sparkling 	Real 	Monthly 	187 	12 	📥 	🔍
95.Q 	Beer 	Real 	Quarterly 	32 	4 	📥 	🔍
96.I 	ECG: A 	Real 	0.5s Intervals 	1800 	60 	📥 	🔍
97.I 	ECG: B 	Real 	0.5s Intervals 	1800 	60 	📥 	🔍
98.I 	Laser 	Real 	1.0s Intervals 	1000 	8 	
"""

lines = data.strip().split('\n')

# Parsing data and creating YAML structure
yaml_data = {}
for line in lines[:97]:
    parts = line.split("\t")
    name = '_'.join(parts[1].replace(':', '_').split()).lower()  # Replace spaces and ':' with '_'
    size = parts[-4]  # The size is the fourth element from the end
    yaml_data[name] = {
        'size': int(size),
        'icmc_usp': {
            'name': name
        }
    }
    print(yaml_data)

yaml_file_path = 'icmc_usp.yaml'

with open(yaml_file_path, 'w') as file:
    yaml.dump(yaml_data, file, sort_keys=False)