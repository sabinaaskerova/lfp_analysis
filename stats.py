
from calculate_psd import Calculate_PSD
from perform_pca import Perform_PCA
from mean_fluctuations import PSD_Fluctuations
import os
from perform_ica import Perform_ICA
import numpy as np
import csv

class AnalysisResults:
    def __init__(self):
        self.results = []

    def add_result(self, animal, ana, cond, band_type, mean_val, median_val):
        self.results.append([animal, ana, cond, band_type, mean_val, median_val])

    def save_results(self, filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['animal', 'ana', 'cond', 'frequency_band', 'mean_value', 'median_value'])
            writer.writerows(self.results)



def get_target_directory():
    return os.getcwd()

def change_directory(target_directory):
    os.chdir(target_directory)

if __name__ == "__main__":
    target_directory = get_target_directory()
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
    
    pca_14_path =target_directory+"/pca_cleaned_sw%.2f_ol%.2f/"%(sliding_window_length,overlap)
    psd_path = target_directory+"/delta_psd_pca_cleaned_sw%.2f_ol%.2f/"%(sliding_window_length,overlap)
    psd_path_gamma = target_directory+"/gamma_psd_pca_cleaned_sw%.2f_ol%.2f/"%(sliding_window_length,overlap)
    ica_14_path = "/reconstructed_signal/"
    
    
    animal_list = ['20110810B', '20110817A', '20110817B', '20110824A', '20110824B', '20110913A']
    ana_list = ['low', 'high', 'medium']
    cond_list = ['pre', 'post', 'stim']
    
    results = AnalysisResults()
    for animal in animal_list:
        for ana in ana_list:
            for cond in cond_list:
                
                print(animal,ana,cond)
                path_data = target_directory+ica_14_path+animal+"_"+ana+"_"+cond+"_new.dat"
                n_comp = Perform_PCA.define_number_components(path_data, noise_channels_removed=True)
                Perform_PCA.perform_pca(path_data, n_comp, animal, ana, cond, noise_channels_removed=True, result_folder=pca_14_path)
                
                # delta band
                time,freqs,PSD = Calculate_PSD.compute_PSD_all_portions(file=animal,data_path=pca_14_path,
                                                    ana=ana, stype=cond, 
                                                    fmin=fmin, fmax=fmax,result_folder=psd_path,
                                                    sliding_window_length=sliding_window_length, overlap=overlap, 
                                                    n_channels=14, portion_length=100, 
                                                    log=True, mean=False)
                freqs_delta,PSD_delta,mean_delta,variance_delta = Calculate_PSD.bands(freqs,PSD,fmin_delta,fmax_delta)
                freqs_gamma,PSD_gamma,mean_gamma, variance_gamma = Calculate_PSD.bands(freqs,PSD,fmin_gamma,fmax_gamma)
                
                mean_delta_val = np.mean(mean_delta[0, :])  
                median_delta_val = np.median(mean_delta[0, :])
                mean_gamma_val = np.mean(mean_gamma[0, :])  
                median_gamma_val = np.median(mean_gamma[0, :])

                results.add_result(animal, ana, cond, 'delta', mean_delta_val, median_delta_val)
                results.add_result(animal, ana, cond, 'gamma', mean_gamma_val, median_gamma_val)
                
                results_file = 'mean_fluct_stats.csv'
                results.save_results(results_file)
                
                

                
                
            