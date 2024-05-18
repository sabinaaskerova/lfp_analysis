import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

class Perform_PCA:
    @staticmethod
    def elbow_criteria(path, n_components=14, noise_channels_removed=False, plot_scree=False):
        if noise_channels_removed:
            data = np.loadtxt(path)
        else:
            data = np.loadtxt(path)[:, 1:-1] # exclude the first and the last channels
        data_centered = data - np.mean(data, axis=0)

        pca = PCA(n_components=n_components)
        pca.fit_transform(data_centered)
        explained_variance = pca.explained_variance_

        diff = np.diff(explained_variance)
        diff2 = np.diff(diff)
        elbow_point = np.argmax(diff2) + 1

        if plot_scree:
            n_components = range(1, len(pca.explained_variance_) + 1)
            plt.figure(figsize=(6, 6))
            plt.plot(n_components,explained_variance, 'o-', linewidth=2)
            plt.title('Screen Plot')
            plt.xlabel('Number of Components')
            plt.ylabel('Explained Variance')
            plt.show()

            print(f'Elbow point is at {elbow_point+1} components')

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        print("Cumulative explained variance for "+path+" : "+ cumulative_variance * 100)

        return elbow_point+1

    @staticmethod
    def define_number_components(path, n_components=14, noise_channels_removed=False, threshold=0.95):
        if noise_channels_removed:
            data = np.loadtxt(path)
        else:
            data = np.loadtxt(path)[:, 1:-1]
        data_centered = data - np.mean(data, axis=0)

        pca = PCA(n_components=n_components)
        pca.fit_transform(data_centered)

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(cumulative_variance >= threshold) + 1

        print(f'Number of components that explain at least {threshold*100}% of the variance: {num_components}')
        print("Cumulative explained variance: ", cumulative_variance * 100)

        return num_components

    @staticmethod
    def perform_pca(path, n_components, animal, ana, cond, noise_channels_removed=False,result_folder="pca_14/"):
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        if noise_channels_removed:
            data = np.loadtxt(path)
        else:
            data = np.loadtxt(path)[:, 1:-1] # exclude the first and the last channels
        data_centered = data - np.mean(data, axis=0)
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(data_centered)
        np.savetxt(result_folder+animal+'_'+ana+'_'+cond+"_eigenvectors.dat", pca.components_)
        data_new = data_centered@pca.components_.T 
        print(data_new.shape) 
        np.savetxt(result_folder+animal+'_'+ana+'_'+cond+"_new.dat", data_new)
        #quit()
        return pca

    @staticmethod
    def visualize_pc(animal, ana, cond, result_folder="pca_14/"):
        data_new = np.loadtxt(result_folder+animal+'_'+ana+'_'+cond+"_new.dat")
        num_times = data_new.shape[0]
        plt.title("PCA for signal:"+animal+'_'+ana+'_'+cond)
        for i in range(3):
            plt.plot(range(num_times), data_new[:,i])
            plt.legend(['PC1', 'PC2','PC3'])
        
        graphs_path = result_folder+"graphs/"
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)
        plt.savefig(graphs_path+animal+'_'+ana+'_'+cond+".png")
        print("Saved the graph to "+graphs_path+animal+'_'+ana+'_'+cond+".png")
        plt.show()
