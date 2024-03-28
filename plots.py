from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], 
    # Optionally specify the LaTeX preamble to include specific packages
    "text.latex.preamble": r"\usepackage{amsmath}",
})
size = 30
size_values = 24
probs_file_path = r'/home/markus/state_of_research/PSCC_2024/Code/pscc-2024-hydorgen-pv-planning/probs_both_20.csv'
full_file_path =  r'/home/markus/state_of_research/PSCC_2024/Code/pscc-2024-hydorgen-pv-planning/Historical_data_corrected.csv'
def create_2D_plot():
    full_df = pd.read_csv(full_file_path)
    probs_df = pd.read_csv(probs_file_path)


    # Extracting temperature and solar radiation columns for the full dataset
    full_temperature = full_df['T[°C]']
    full_solar_radiation = full_df['G [W/m2]']

    # Calculate the KDE for the full dataset to get probability densities
    data_points = np.vstack([full_temperature, full_solar_radiation])
    kde = gaussian_kde(data_points)

    # Evaluate the KDE on the grid of the full dataset
    x_grid = np.linspace(full_temperature.min(), full_temperature.max(), 100)
    y_grid = np.linspace(full_solar_radiation.min(), full_solar_radiation.max(), 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Normalize Z to get probabilities
    Z /= Z.max()

    plt.figure(figsize=(12, 8))
    # Plot the density
    contourf =plt.contourf(X, Y, Z, levels=50, cmap='Reds')

    marker_sizes = probs_df['prob'] * 1000 
    # Overlay the sample points
    scatter = plt.scatter(probs_df['T[°C]'], probs_df['G [W/m2]'], s=marker_sizes, c='blue', marker='x', label='Mean values of scenarios $\mu_s$')

    # Annotate each sample with its probability
    for i, row in probs_df.iterrows():
        if i == 1:
            x = 12
            y = 19
        elif i == 2: # 3 link
            x = 25
            y = 8
        elif i == 7:# 2 links
            x = -15
            y = 7
        elif i == 10:  # 1 links
            x = -28
            y = -8
        elif i == 17:
            x = 20
            y = 0
        elif i == 3:
            x = 15
            y = 5
        elif i == 4:
            x = -10
            y = 5
        else:
            x,y = 5, 5
        plt.annotate(f"{row['prob']:.2f}", (row['T[°C]'], row['G [W/m2]']), textcoords="offset points", xytext=(x,y), ha='center', fontsize=20, color='black')

    cbar = plt.colorbar(contourf)
    plt.tick_params(axis='both', which='major', labelsize=size_values*1.2)
    cbar.set_label('Probability density', fontsize=size_values*1.2)
    cbar.ax.tick_params(labelsize=size_values*1.2) 
    #plt.title('Probability Distribution with Annotated Sample Points')
    plt.tick_params(axis='both', which='major', labelsize=size_values*1.2)
    plt.xlabel('Temperature [°C]', fontsize=size*1.2)
    plt.ylabel('Solar radiation [W/m²]', fontsize=size*1.2)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(r'/home/markus/state_of_research/PSCC_2024/Code/pscc-2024-hydorgen-pv-planning/prob_T_r_GMM.pdf')
    plt.show()
# 2D-probability distribution plot
create_2D_plot()


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
plt.savefig(r'/home/markus/state_of_research/PSCC_2024/Code/pscc-2024-hydorgen-pv-planning/c_PV_prob_density.pdf')
plt.show()
