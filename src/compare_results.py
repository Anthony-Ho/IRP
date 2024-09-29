import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Local Imports
from experiment_config import result_dir

# Redefining the Cohen's d function
def cohen_d(group1, group2):
    # Mean difference
    diff_mean = np.mean(group1) - np.mean(group2)
    # Pooled standard deviation
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    # Calculate Cohen's d
    return diff_mean / pooled_std

# Modified function to include print statements showing t-values, p-values, and d-values for Group1 and Group2
def perform_tests_and_cohen_d_with_print(data, metric, group):
    baseline_data = data[(data['strategy'] == 'baseline') & (data['data_group'] == group)][metric]
    naive_data = data[(data['strategy'] == 'naive') & (data['data_group'] == group)][metric]
    ewc_data = data[(data['strategy'] == 'ewc') & (data['data_group'] == group)][metric]
    replay_data = data[(data['strategy'] == 'replay') & (data['data_group'] == group)][metric]

    # Perform t-tests between baseline and each CL strategy
    t_naive, p_naive = ttest_ind(baseline_data, naive_data, equal_var=False)
    t_ewc, p_ewc = ttest_ind(baseline_data, ewc_data, equal_var=False)
    t_replay, p_replay = ttest_ind(baseline_data, replay_data, equal_var=False)

    # Calculate Cohen's d
    d_naive = cohen_d(baseline_data, naive_data)
    d_ewc = cohen_d(baseline_data, ewc_data)
    d_replay = cohen_d(baseline_data, replay_data)

    # Print the results for each strategy
    print(f"--- {group} Results for {metric} ---")
    print(f"Naive vs Baseline: t-value = {t_naive:.4f}, p-value = {p_naive:.4f}, Cohen's d = {d_naive:.4f}")
    print(f"EWC vs Baseline: t-value = {t_ewc:.4f}, p-value = {p_ewc:.4f}, Cohen's d = {d_ewc:.4f}")
    print(f"Replay vs Baseline: t-value = {t_replay:.4f}, p-value = {p_replay:.4f}, Cohen's d = {d_replay:.4f}")
    print("\n")

    return {
        'metric': metric,
        'group': group,
        't_naive': t_naive, 'p_naive': p_naive, 'd_naive': d_naive,
        't_ewc': t_ewc, 'p_ewc': p_ewc, 'd_ewc': d_ewc,
        't_replay': t_replay, 'p_replay': p_replay, 'd_replay': d_replay
    }

# Perform t-tests and Cohen's d comparisons for each RL agent across CL strategies on Group1 and Group2
def perform_agent_comparisons(data, metric, agent):
    agent_data = data[data['rl_model'] == agent]
    
    results = []
    for group in ['Group1', 'Group2']:
        results.append(perform_tests_and_cohen_d_with_print(agent_data, metric, group))
    
    return results


# Reloading the combined results CSV file
combined_results = pd.read_csv(os.path.join(result_dir, 'combined_results.csv'))

# Perform tests and Cohen's d with print statements for Group1 and Group2
test_results_with_print = []
for group in ['Group1', 'Group2']:
    test_results_with_print.append(perform_tests_and_cohen_d_with_print(combined_results, 'Cumulative Return', group))

# Convert the results into a DataFrame
test_results_with_print_df = pd.DataFrame(test_results_with_print)


# Run the comparison for Cumulative Return across all RL agents (PPO, A2C, DDPG)
rl_agents = ['PPO', 'A2C', 'DDPG']
all_agent_results = []

for agent in rl_agents:
    print(f"===== {agent} Results for Cumulative Return =====")
    all_agent_results.append(perform_agent_comparisons(combined_results, 'Cumulative Return', agent))

# Convert the results into a DataFrame for further analysis
all_agent_results_df = pd.DataFrame(all_agent_results)


# Filter data for cumulative return across all strategies
plot_data = combined_results[['Cumulative Return', 'strategy', 'data_group']]

# Create a boxplot to visualize cumulative return for each strategy on Group1 and Group2
plt.figure(figsize=(10, 6))
sns.boxplot(x='strategy', y='Cumulative Return', hue='data_group', data=plot_data)

# Add titles and labels
plt.title('Cumulative Return Comparison for All Strategies on Group1 and Group2', fontsize=14)
plt.xlabel('Strategy', fontsize=12)
plt.ylabel('Cumulative Return', fontsize=12)
plt.legend(title='Data Group')

# Display the plot
plt.tight_layout()
plt.show()

