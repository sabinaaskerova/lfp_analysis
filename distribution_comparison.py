import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(filename):
    return pd.read_csv(filename)

def filter_data(df, cond, frequency_band, ana):
    new_df = df[(df['cond'] == cond) & 
                (df['frequency_band'] == frequency_band) & 
                (df['ana'] == ana)]
    print(new_df)
    return new_df

def compute_statistics(data):
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    return mean_val, median_val, std_val

def plot_bar(data, labels, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    bar_positions = np.arange(len(data))
    plt.bar(bar_positions, data, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    for i, v in enumerate(data):
        plt.text(i, v + 0.02, f'{labels[i]}\n{v:.2f}', ha='center', va='bottom')
    
    plt.show()

if __name__ == "__main__":
    # Read the data
    filename = 'mean_fluct_stats.csv'
    df = read_data(filename)
    
    # Define conditions
    cond = 'pre'
    frequency_band = 'delta'  # 'gamma' or 'delta'
    ana = 'low'  # 'low', 'high', or 'medium'

    # Filter data based on conditions
    filtered_df = filter_data(df, cond, frequency_band, ana)

    # Gather the mean values for the 6 animals
    mean_values = filtered_df['mean_value']
    mean_values_array = mean_values.to_numpy()
    print(mean_values_array)
    
    # Create animal indexes (1 to 6)
    animal_indexes = range(1, len(mean_values_array) + 1)

    # Compute statistics
    mean_val, median_val, std_val = compute_statistics(mean_values)
    
    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Standard Deviation: {std_val}")

    plot_bar(mean_values_array, 
             animal_indexes,
             f'Bar Plot of Mean Values\nCondition: {cond}, Band: {frequency_band}, Ana: {ana}',
             'Animal',
             'Mean value of mean PSD values over time')
