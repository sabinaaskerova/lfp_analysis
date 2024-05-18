from calculate_psd import Calculate_PSD
from perform_pca import Perform_PCA
from mean_fluctuations import PSD_Fluctuations
import os
from perform_ica import Perform_ICA
import numpy as np
import matplotlib.pyplot as plt

def get_target_directory():
    return os.getcwd()

def change_directory(target_directory):
    os.chdir(target_directory)

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
    
    for animal in animal_list:
        for ana in ana_list:
            for cond in cond_list:
   
                # ICA
                path_data = target_directory+original_data_path+"ascii_out_"+animal+"_"+ana+"_"+cond+".dat"
                data = np.loadtxt(path_data)[:, 1:-1]
                ica_model, V = Perform_ICA.perform_ica(data, n_components=13)
                
                components = ica_model.components_
                mixing_matrix = ica_model.mixing_
                mean_ = ica_model.mean_
                
                Perform_ICA.visualize_components(components.T,animal,ana,cond, all=False, original_signal=True)

   
    # new_components, new_mixing_matrix, new_V, new_mean = Perform_ICA.remove_mode(artifact, mixing_matrix ,components,  V, mean_) # remove artifact mode
    # new_signal = Perform_ICA.recreate_signal(mixing_=new_mixing_matrix,mean_=new_mean, V=new_V)
    # Perform_ICA.visualize_components(new_signal.T, all=True, original_signal=True)