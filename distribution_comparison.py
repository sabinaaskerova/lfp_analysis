import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def read_data(filename):
    return pd.read_csv(filename)

def filter_data(df, cond, frequency_band, ana):
    new_df = df[(df['cond'] == cond) & 
                (df['frequency_band'] == frequency_band) & 
                (df['ana'] == ana)]
    return new_df

def compute_statistics(data):
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    return mean_val, median_val, std_val

def plot_and_save_bar(data, labels, mean_values, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    bar_positions = np.arange(len(data))
    plt.bar(bar_positions, data, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    for i, v in enumerate(data):
        plt.text(i, v + 0.02, f'{mean_values[i]:.2f}', ha='center', va='bottom')

    plt.xticks(bar_positions, labels) 
    
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    filename = 'mean_fluct_stats.csv'
    df = read_data(filename)
    
    cond_list = ['pre', 'post', 'stim']
    frequency_band_list = ['gamma', 'delta']
    ana_list = ['low', 'high', 'medium']
    animal_list = ['20110810B', '20110817A', '20110817B', '20110824A', '20110824B', '20110913A']

    output_dir = 'bar_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    statistics_list = []
    for cond in cond_list:
        for frequency_band in frequency_band_list:
            for ana in ana_list:
                filtered_df = filter_data(df, cond, frequency_band, ana)
                
                mean_values = filtered_df['mean_value']
                mean_values_array = mean_values.to_numpy()

                mean_val, median_val, std_val = compute_statistics(mean_values)
                statistics = {
                    'Condition': cond,
                    'Frequency Band': frequency_band,
                    'Ana': ana,
                    'Mean': mean_val,
                    'Median': median_val,
                    'Standard Deviation': std_val
                }
                statistics_list.append(statistics)

                print(f"Condition: {cond}, Band: {frequency_band}, Ana: {ana}")
                print(f"Mean: {mean_val}")
                print(f"Median: {median_val}")
                print(f"Standard Deviation: {std_val}")

                plot_title = f'Bar Plot of Mean Values\nCondition: {cond}, Band: {frequency_band}, Ana: {ana}'
                plot_filename = os.path.join(output_dir, f'{frequency_band}_{ana}_{cond}_bar.png')
                plot_and_save_bar(mean_values_array, 
                                  animal_list,
                                  mean_values_array,
                                  plot_title,
                                  'Animal',
                                  'Mean value of mean PSD values over time (dB/Hz)',
                                  plot_filename)
    
    statistics_df = pd.DataFrame(statistics_list)
    statistics_df.to_csv('statistics_summary.csv', index=False)
