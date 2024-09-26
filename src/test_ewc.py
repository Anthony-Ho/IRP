# Assuming your code structure in src/
import sys
import os
import pandas as pd

# Add src/ to the system path
sys.path.append('/workspace/IRP/src')

# Import your iteration functions
from experiment_config import tic_list, result_dir, model_dir, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS
from training import train_baseline_agents, train_naive_strategy, train_ewc_agents, train_replay_agents
from data_processing import split_collect_stock_data_from_csv
from experiment_utils import update_combination_status
from envs import PortfolioAllocationEnv
from performance import test_agent_performance, calculate_performance_metrics

iteration, group1, group2, df1, df2 = split_collect_stock_data_from_csv(tic_list=tic_list)

# Split the dataset based on the 'Date' or first level of the multi-index
train_df1 = df1.loc[(df1.index.get_level_values(0) >= '2010-01-01') & (df1.index.get_level_values(0) <= '2019-12-31')]
validation_df1 = df1.loc[(df1.index.get_level_values(0) >= '2020-01-01') & (df1.index.get_level_values(0) <= '2021-12-31')]
trade_df1 = df1.loc[(df1.index.get_level_values(0) >= '2022-01-01') & (df1.index.get_level_values(0) <= '2023-12-31')]

train_df2 = df2.loc[(df2.index.get_level_values(0) >= '2010-01-01') & (df2.index.get_level_values(0) <= '2019-12-31')]
validation_df2 = df2.loc[(df2.index.get_level_values(0) >= '2020-01-01') & (df2.index.get_level_values(0) <= '2021-12-31')]
trade_df2 = df2.loc[(df2.index.get_level_values(0) >= '2022-01-01') & (df2.index.get_level_values(0) <= '2023-12-31')]

# Step 1: Baseline Training
print(f"Iteration {iteration}: Training Baseline Agents")
ppo_baseline, a2c_baseline, ddpg_baseline = train_baseline_agents(
    model_dir, train_df1, group1, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS
)

# Step 3: EWC Strategy Training
print(f"Iteration {iteration}: Training EWC Strategy Agents")
ppo_ewc, a2c_ewc, ddpg_ewc = train_ewc_agents(
    model_dir, train_df1, train_df2, group1, group2, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS
)