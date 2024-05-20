import numpy as np
import matplotlib.pyplot as plt
import os
from sys import platform

class PSD_Fluctuations:
    @staticmethod
    def mean_fluctuations(path):
        files = os.listdir(path)
        files.sort()
        means = []
        for file in files:
            if file[-4:] != ".dat":
                continue
            psd_data = np.loadtxt(path+file)[:,1:3] # two first PC, assuming there are at least two PC
            psd_variance = np.mean(psd_data, axis=0)
            means.append(psd_variance)
        return np.array(means)

    @staticmethod
    def variance_fluctuations(path):
        files = os.listdir(path)
        files.sort()
        variances = []
        for file in files:
            if file[-4:] != ".dat":
                continue
            psd_data = np.loadtxt(path+file)[:,1:3] # two first PC, assuming there are at least two PC
            psd_variance = np.var(psd_data, axis=0)
            variances.append(psd_variance)
        return np.array(variances)

    @staticmethod
    def plot_mean_fluctuations(means, string, band_str, path, num_channels=2, plot=True):
        fig, axs = plt.subplots(num_channels, figsize=(10, 10))
        for i in range(num_channels):
            axs[i].plot(means[:,i])
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Mean')
            axs[i].set_title('Mean of PSD Fluctuations over Time '+ string+band_str)
            axs[i].grid(True)
            num_ticks = 7  
            custom_labels = np.linspace(0, 600, num_ticks).astype(int)  # Labels from 0 to 600
            tick_positions = np.linspace(0, len(means[:,i]) - 1, num_ticks)  # Corresponding positions in original scale

            axs[i].set_xticks(tick_positions)
            axs[i].set_xticklabels(custom_labels)
       
        graphs_path = path+'ascii_out_'+string+"/graphs/"
        print(graphs_path)
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)
        plt.savefig(graphs_path+string+band_str+"mean_fluctuations_.png")
        print("Saved the graph to "+graphs_path+string+band_str+".png")
        
        if plot==True:
            plt.show()
    
    @staticmethod
    def plot_variance_fluctuations(variances, string, band_str, path, num_channels=2, plot=True):
        fig, axs = plt.subplots(num_channels, figsize=(10, 10))
        for i in range(num_channels):
            axs[i].plot(variances[:,i])
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Variance of frequencies')
            axs[i].set_title('Variance of PSD Fluctuations over Time '+ string+band_str)
            axs[i].grid(True)
            num_ticks = 7  
            custom_labels = np.linspace(0, 600, num_ticks).astype(int)  # Labels from 0 to 600
            tick_positions = np.linspace(0, len(variances[:,i]) - 1, num_ticks)  # Corresponding positions in original scale

            axs[i].set_xticks(tick_positions)
            axs[i].set_xticklabels(custom_labels)
        graphs_path = path+'ascii_out_'+string+"/graphs/"
        print(graphs_path)
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)
        plt.savefig(graphs_path+string+band_str+"variance_fluctuations_.png")
        print("Saved the graph to "+graphs_path+string+band_str+".png")
        if plot==True:
            plt.show()