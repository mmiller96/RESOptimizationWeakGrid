import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
folder_path_results = os.path.join(current_directory, 'results')
folder_path_network = os.path.join(current_directory, 'network')
full_file_path =  os.path.join(folder_path_network, 'Historical_data_corrected.csv')
file_path = os.path.join(folder_path_results, "Res4.pdf")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], 
    # Optionally specify the LaTeX preamble to include specific packages
    "text.latex.preamble": r"\usepackage{amsmath}",
})
size = 30
size_values = 24

# PV generation probability distribution
data_df = pd.read_csv(full_file_path)
NOCT = 40
eta_conv_PV = 0.95
p_coef = -0.35

# Define the function to calculate c_PV
def coeff_PV(T, G, NOCT, eta_conv_PV, p_coef):
    T_GT = T + (NOCT - 20) * G / 800
    Delta_T = T_GT - 25
    Delta_P = Delta_T * p_coef
    eta_T = 1 + (Delta_P/100)
    eta_G = G/1000
    return eta_T * eta_G * eta_conv_PV, eta_T, eta_G

c_PV_values = []
for index, row in data_df.iterrows():
    c_PV, eta_T, eta_G = coeff_PV(row['T[°C]'], row['G [W/m2]'], NOCT, eta_conv_PV, p_coef)
    c_PV_values.append(c_PV)
c_PV = np.array(c_PV_values)

plt.figure(figsize=(10, 6))
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=size_values)
plt.hist(c_PV, bins=20, color='blue', edgecolor='black', alpha=0.7, density=True)
plt.xlabel('$\epsilon^{\mathrm{PV}}$', fontsize=size)
plt.ylabel('Probability density', fontsize=size)
plt.tight_layout()
plt.savefig(file_path)
plt.show()