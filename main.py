
from calculate_psd import Calculate_PSD
from perform_pca import Perform_PCA
from mean_fluctuations import PSD_Fluctuations
import os
from perform_ica import Perform_ICA

def get_target_directory():
    return os.getcwd()

def change_directory(target_directory):
    os.chdir(target_directory)

if __name__ == "__main__":
    target_directory = get_target_directory()
    original_data_path = "/data/"
    # Calculate PSD on PCA components
    #pca_14_path =target_directory+"/pca_14/"
    #psd_path = target_directory+"/delta_psd_pca/"
    #psd_path_gamma = target_directory+"/gamma_psd_pca/"
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
                # PCA
                path_data = target_directory+original_data_path+"ascii_out_"+animal+"_"+ana+"_"+cond+".dat"
                n_comp = Perform_PCA.define_number_components(path_data)
                Perform_PCA.perform_pca(path_data, n_comp, animal, ana, cond, result_folder=pca_14_path)
                
                # delta band
                time,freqs,PSD = Calculate_PSD.compute_PSD_all_portions(file=animal,data_path=pca_14_path,
                                                    ana=ana, stype=cond, 
                                                    fmin=fmin, fmax=fmax,result_folder=psd_path,
                                                    sliding_window_length=sliding_window_length, overlap=overlap, 
                                                    n_channels=14, portion_length=100, 
                                                    log=True, mean=False)
                # ICA
                n_comp_ica = Perform_ICA.define_number_components(path_data)
                Perform_ICA.extract_ica_components(n_comp_ica, [1,2], ana, animal, cond, result_folder=ica_14_path)
                #animal, ana,stype,time,freqs,PSD,psd_file_prefix, channel_no=1, plot=False
                freqs_delta,PSD_delta,mean_delta = Calculate_PSD.bands(freqs,PSD,fmin_delta,fmax_delta)
                freqs_gamma,PSD_gamma,mean_gamma = Calculate_PSD.bands(freqs,PSD,fmin_gamma,fmax_gamma)
                Calculate_PSD.plot_colored_psd(animal,ana,cond,"delta", 
                                time,freqs_delta,PSD_delta,psd_path, channel_no=1, plot=False)
                Calculate_PSD.plot_colored_psd(animal,ana,cond,"gamma", 
                                time,freqs_gamma,PSD_gamma,psd_path, channel_no=1, plot=False)
                

                #quit()
                string = animal+"_"+ana+"_"+cond+'_'+str(fmin)+'_'+str(fmax)
                
                path_psd = psd_path+"ascii_out_"+string+"/"
                #means = PSD_Fluctuations.mean_fluctuations(path_psd)
                band_string = "delta"
                PSD_Fluctuations.plot_mean_fluctuations(mean_delta.T, string,band_string, path = psd_path,num_channels=2, plot=False)
                band_string = "gamma"
                PSD_Fluctuations.plot_mean_fluctuations(mean_gamma.T, string, band_string,path = psd_path,num_channels=2, plot=False)
                
#               # gamma band
#               Calculate_PSD.compute_PSD_all_portions(file=animal,data_path=pca_14_path,
#                                                   ana=ana, stype=cond, 
#                                                   fmin=fmin_gamma, fmax=fmax_gamma, result_folder=psd_path_gamma,
#                                                   sliding_window_length=sliding_window_length, overlap=overlap, 
#                                                   n_channels=14, portion_length=10, 
#                                                   log=True, mean=False)
#               Calculate_PSD.plot_colored_psd(data_path=pca_14_path,animal=animal,ana=ana, stype=cond, 
#                               fmin=fmin_gamma, fmax=fmax_gamma, sliding_window_length=sliding_window_length, 
#                               df=df, psd_file_prefix=psd_path_gamma, channel_no=1, plot=False)
#
#               path_psd_gamma = psd_path_gamma+"ascii_out_"+animal+"_"+ana+"_"+cond+'_'+str(fmin_gamma)+'_'+str(fmax_gamma)+"/"
#               means_gamma = PSD_Fluctuations.mean_fluctuations(path_psd_gamma)
#               PSD_Fluctuations.plot_mean_fluctuations(means=means_gamma, animal=animal, ana=ana, cond=cond, fmin=fmin_gamma, fmax=fmax_gamma,
#                                                       path = psd_path_gamma,num_channels=2)
            