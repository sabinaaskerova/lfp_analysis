import pandas as pd
from scipy.stats import ttest_ind

def read_data(filename):
    return pd.read_csv(filename)

def filter_data(df, cond, frequency_band):
    return df[(df['cond'] == cond) & (df['frequency_band'] == frequency_band)]

if __name__ == "__main__":
    filename = 'mean_fluct_stats.csv'
    df = read_data(filename)

    delta_pre = filter_data(df, 'pre', 'delta')['mean_value']
    delta_post = filter_data(df, 'post', 'delta')['mean_value']

    t_statistic_delta, p_value_delta = ttest_ind(delta_pre, delta_post, equal_var=False)
    print("Delta Band (Welch's t-test):")
    print("T-statistic:", t_statistic_delta)
    print("P-value:", p_value_delta)

    gamma_pre = filter_data(df, 'pre', 'gamma')['mean_value']
    gamma_post = filter_data(df, 'post', 'gamma')['mean_value']

    t_statistic_gamma, p_value_gamma = ttest_ind(gamma_pre, gamma_post, equal_var=False)
    print("\nGamma Band (Welch's t-test):")
    print("T-statistic:", t_statistic_gamma)
    print("P-value:", p_value_gamma)
