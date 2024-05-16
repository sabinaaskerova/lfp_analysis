import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from sys import platform


class Perform_ICA:
    @staticmethod
    def perfrorm_ica(data, n_components=14):
        ica = FastICA(n_components=n_components, random_state=42)
        ica.fit(data.T)
        return ica.components_
    
    @staticmethod
    def remove_mode(mode_number, components):
        return np.delete(components, mode_number, axis=0)
    
    @staticmethod
    def recreate_signal(components):
        X = np.linalg.pinv(components)
        return np.dot(X, components)
        
    
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
    def visualize_components(extracted_components, all=True):
        if all:
            plt.plot(extracted_components, label='ICA')
            labels = range(0, 601, 100)
            plt.xticks(np.linspace(0, len(extracted_components), len(labels)), labels)
        else:
            for i in range(0, extracted_components.shape[1]):
                plt.figure(figsize=(8, 4))
                plt.plot(extracted_components[:,i])
                plt.title(f'Independent component  {i+1}')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')

            labels = range(0, 601, 100)
            plt.xticks(np.linspace(0, len(extracted_components[i]), len(labels)), labels)
            plt.show()


    