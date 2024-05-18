import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from perform_pca import Perform_PCA

def get_target_directory():
    return os.getcwd()

def change_directory(target_directory):
    os.chdir(target_directory)

class Perform_ICA:
    @staticmethod
    def perform_ica(data, n_components=14):
        ica = FastICA(n_components=n_components, random_state=42, whiten="arbitrary-variance")
        V = ica.fit_transform(data.T)
        return ica, V
    
    @staticmethod
    def define_artifact_mode(components):
        # returns the index of the mode with the highest variance
        return np.argmax(np.var(components, axis=1))
    
    @staticmethod
    def remove_modes(modes, mixing_matrix, components,  V, mean_):
        
        new_components = components.copy()
        new_mixing_matrix = mixing_matrix.copy()
        new_V = V.copy()
        new_mean = mean_.copy()
        for mode_number in modes:
            if mode_number >= 0 and mode_number < components.shape[0]:
                new_components[mode_number, :] = 0
                new_mixing_matrix[:, mode_number] = 0
                new_V[:, mode_number] = 0
                new_mean[mode_number] = 0
        return new_components, new_mixing_matrix, new_V, new_mean
    
    @staticmethod
    def recreate_signal(mixing_, mean_, V):
        return np.dot(V, mixing_.T) + mean_
        
    @staticmethod
    # using kurtosis
    def define_number_components(path, n_components=13, plot=False):
        data = np.loadtxt(path)[:, 1:-1]
        ica = FastICA(n_components=n_components, random_state=42)
        ica.fit(data.T)
        components = ica.components_
        print(components.shape)

        kurtosis_values = np.zeros(n_components)
        for i in range(n_components):
            kurtosis_values[i] =stats.kurtosis(components[i], fisher=False)

        plt.figure(figsize=(5, 3))
        plt.plot(np.arange(1, n_components + 1), kurtosis_values, marker='o', linestyle='-')
        plt.title('Kurtosis of Independent Components')
        plt.xlabel('Component Index')
        plt.ylabel('Kurtosis')
        plt.grid(True)
        max_kurtosis_indexes = np.argsort(kurtosis_values)[-3:][::-1]
        plt.show()
        return components, max_kurtosis_indexes
    
    @staticmethod
    def extract_ica_components(components, index_list, ana, animal, cond, result_folder="ica/"):
        index_list = [i - 1 for i in index_list]
        extracted_components = components.T[:, index_list]
        df_post = pd.DataFrame(extracted_components)
        stats_post = df_post.describe()
        print(stats_post)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        result_path = result_folder+animal+'_'+ana+'_'+cond+'_ica_kurtosis.dat'
        df_post.to_csv(result_path, sep='\t', index=False)
        print("ICA components written to file "+result_path)
        return extracted_components

    @staticmethod
    def visualize_components(extracted_components, animal, ana, cond, all=True, original_signal=False, result_folder="ica/"):
        path = animal+'_'+ana+'_'+cond+'/'
        graphs_path = result_folder+path+"graphs/"
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)
        if all:
            plt.plot(extracted_components, label='ICA')
            labels = range(0, 601, 100)
            plt.xticks(np.linspace(0, len(extracted_components), len(labels)), labels)
            plt.savefig(graphs_path+animal+'_'+ana+'_'+cond+".png")
        else:
            for i in range(0, extracted_components.shape[1]):
                plt.figure(figsize=(8, 4))
                plt.plot(extracted_components[:,i])
                if original_signal:
                    plt.title(f'{animal} {ana} {cond} Signal {i+1}')
                else:
                    plt.title(f'{animal} {ana} {cond} Independent component  {i+1} ')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')

                labels = range(0, 601, 100)
                plt.xticks(np.linspace(0, len(extracted_components[i]), len(labels)), labels)
                plt.savefig(graphs_path+animal+'_'+ana+'_'+cond+'_'+str(i+1)+".png")
        
        
        
        plt.show()


    

    