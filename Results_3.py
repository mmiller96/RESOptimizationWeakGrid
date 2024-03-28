from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
folder_path_results = os.path.join(current_directory, 'results')
folder_path_network = os.path.join(current_directory, 'network')
folder_path_data = os.path.join(current_directory, 'data')
file_path = os.path.join(folder_path_results, "Res3.pdf")
probs_file_path = os.path.join(folder_path_data, 'probs_20.csv')
full_file_path =  os.path.join(folder_path_network, 'Historical_data_corrected.csv')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], 
    # Optionally specify the LaTeX preamble to include specific packages
    "text.latex.preamble": r"\usepackage{amsmath}",
})
size = 30
size_values = 24

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
plt.savefig(file_path)
plt.show()
# 2D-probability distribution plot



