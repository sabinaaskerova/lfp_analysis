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
        ica = FastICA(n_components=n_components, random_state=42, whiten=True)
        S_ = ica.fit_transform(data.T)
        return ica, S_
    
    @staticmethod
    def remove_mode(mode_number, mixing_matrix, components,  S_):
        return np.delete(components, mode_number, axis=0), np.delete(mixing_matrix, mode_number, axis=1), np.delete(S_, mode_number, axis=1)
    @staticmethod
    def recreate_signal(mixing_, mean_, S_):
        return np.dot(S_, mixing_.T) + mean_
        
    
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
    def visualize_components(extracted_components, all=True, original_signal=False):
        if all:
            plt.plot(extracted_components, label='ICA')
            labels = range(0, 601, 100)
            plt.xticks(np.linspace(0, len(extracted_components), len(labels)), labels)
        else:
            for i in range(0, extracted_components.shape[1]):
                plt.figure(figsize=(8, 4))
                plt.plot(extracted_components[:,i])
                if original_signal:
                    plt.title(f'Signal {i+1}')
                else:
                    plt.title(f'Independent component  {i+1}')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')

            labels = range(0, 601, 100)
            plt.xticks(np.linspace(0, len(extracted_components[i]), len(labels)), labels)
            plt.show()


if __name__ == "__main__":
    target_directory = get_target_directory()
    original_data_path = "/data/"
    fmin=0.1
    fmax=80
    
    fmin_delta = 0.1
    fmax_delta = 4.0
    fmin_gamma=20
    fmax_gamma=60
    fbands = [fmin_delta,fmax_delta,fmin_gamma,fmax_gamma]
    
    sliding_window_length=10
    overlap = 0.05#0.1
    df = 1/sliding_window_length
    
    pca_14_path =target_directory+"/pca_14_sw%.2f_ol%.2f/"%(sliding_window_length,overlap)
    psd_path = target_directory+"/delta_psd_pca_sw%.2f_ol%.2f/"%(sliding_window_length,overlap)
    psd_path_gamma = target_directory+"/gamma_psd_pca_sw%.2f_ol%.2f/"%(sliding_window_length,overlap)
    ica_14_path =target_directory+"/ica_14_sw%.2f_ol%.2f/"%(sliding_window_length,overlap)
    
    
    animal_list = ['20110810B', '20110817A', '20110817B', '20110824A', '20110824B', '20110913A']
    ana_list = ['low', 'high', 'medium']
    cond_list = ['pre', 'post', 'stim']
    index = 1
    animal = animal_list[index]
    ana = ana_list[index]
    cond = cond_list[index]
    # ICA
    path_data = target_directory+original_data_path+"ascii_out_"+animal+"_"+ana+"_"+cond+".dat"
    data = np.loadtxt(path_data)[:, 1:-1]
    ica_model, S_ = Perform_ICA.perform_ica(data, n_components=13)
    
    components = ica_model.components_
    mixing_matrix = ica_model.mixing_
    whiten_matrix = ica_model.whitening_
    mean_ = ica_model.mean_
    
    # print(f"components.shape {components.shape}")
    # print(f"mixing_matrix.shape {mixing_matrix.shape}")
    # print(f"whiten_matrix.shape {whiten_matrix.shape}")
    # print(f"mean_.shape {mean_.shape}")

    new_components, new_mixing_matrix, new_S = Perform_ICA.remove_mode(6, mixing_matrix ,components,  S_) # remove 7th mode
    
    # print(f"new_components.shape {new_components.shape}")
    # print(f"new_mixing_matrix.shape {new_mixing_matrix.shape}")
    # print(f"new_S.shape {new_S.shape}")

    # Perform_ICA.visualize_components(components.T, all=False)    
    
    new_signal = Perform_ICA.recreate_signal(mixing_=new_mixing_matrix,mean_=mean_, S_=new_S)
    print(new_signal.shape)
    # Perform_ICA.visualize_components(new_signal.T, all=False, original_signal=True)
    
    

    