import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import os

def forward_transformation(data_all):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_all)
    scaled_T_G = scaled_data[:, :2]
    data_for_pca = scaled_data[:, 2:]
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data_for_pca)
    explained_variance = pca.explained_variance_ratio_
    scaler_pca = StandardScaler()
    scaled_pca_data = scaler_pca.fit_transform(pca_data)
    final_data = np.hstack([scaled_T_G, scaled_pca_data])
    return final_data, explained_variance, scaler, scaler_pca, pca

def inverse_transformation(Z_low, scaler, scaler_pca, pca):
    inverse_pca_low = scaler_pca.inverse_transform(Z_low[:,2:])
    inverse_pca_data = pca.inverse_transform(inverse_pca_low)
    data_new = scaler.inverse_transform(np.hstack([Z_low[:, :2], inverse_pca_data]))
    return data_new


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    #older_path = os.path.join(current_directory, 'sd', 'network')
    file_path_network = os.path.join(current_directory, 'network', "Historical_data_corrected.csv")
    folder_path_data = os.path.join(current_directory, 'data')
    if not os.path.exists(folder_path_data):
        os.makedirs(folder_path_data)
    data = pd.read_csv(file_path_network, index_col=0)
    data_used = data.iloc[:,4:].copy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_used)
    weights = np.ones(data_scaled.shape[1])
    weight_weather = 20         # it is possible to give T and G higher priority. Higher value means more accurate T and G clustering but worse load clustering.
    weights[2:] = weight_weather*weights[2:]/(weights[2:].sum())            
    X_weighted = data_scaled * weights
    Z, explained_variance, scaler, scaler_pca, pca = forward_transformation(data_used)
    for n_clusters in [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 30]:
            gmm = GaussianMixture(n_components=n_clusters, random_state=0)
            gmm.fit(Z)
            labels = gmm.predict(Z)
            centroids = gmm.means_
            X_centroid = inverse_transformation(centroids, scaler, scaler_pca, pca)
            i=0
            plt.scatter(X_centroid[:,i], X_centroid[:,i+1], color='r')
            plt.scatter(data_used.values[:,i], data_used.values[:,i+1], color='b', alpha=0.025)
            plt.show()
            df_probs = pd.DataFrame(X_centroid, columns=data.iloc[:,4:].columns)
            cluster_sizes = [np.sum(labels == cluster_num) for cluster_num in range(labels.max()+1)]
            probs = [size / len(Z) for size in cluster_sizes]
            df_probs['prob'] = np.array(probs)
            name = "probs_" + str(n_clusters) + ".csv"
            probs_path = os.path.join(folder_path_data, name)
            df_probs.to_csv(probs_path, index=None)