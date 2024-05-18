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
    reconstructed_signal_path = target_directory+"/reconstructed_signal/"
    if not os.path.exists(reconstructed_signal_path):
        os.makedirs(reconstructed_signal_path)
    
    
    animal_list = ['20110810B', '20110817A', '20110817B', '20110824A', '20110824B', '20110913A']
    ana_list = ['low', 'high', 'medium']
    cond_list = ['pre', 'post', 'stim']
    artifact_mode_list = [[10], [6], [2], [8], [7], [], [12], [], [12], [],
                            [5], [], [4, 12], [], [], [9], [6, 12],
                            [], [], [], [], [9], [11], [5], [1, 13],
                            [], [12], [2, 11], [], [], [], [2], [6],
                            [], [], [7, 8], [6, 8], [10], [], [], [1],
                            [], [7], [9], [1, 5, 9], [], [6, 13],
                            [7, 8], [6], [5, 6, 7], [], [4], [], []]
    counter = 0
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
                
                # Perform_ICA.visualize_components(components.T,animal,ana,cond, all=False, original_signal=True)
   
                new_components, new_mixing_matrix, new_V= Perform_ICA.remove_modes(artifact_mode_list[counter], mixing_matrix ,components,  V) # remove artifact mode
                counter += 1
                new_signal = Perform_ICA.recreate_signal(mixing_=new_mixing_matrix,mean_=mean_, V=new_V)
                
                np.savetxt(reconstructed_signal_path+animal+'_'+ana+'_'+cond+"_new.dat", new_signal.T)
                # Perform_ICA.visualize_components(new_signal.T, animal, ana, cond, all=True, original_signal=True, result_folder="cleaned_signal_ica/")