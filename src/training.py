import os
from stable_baselines3 import PPO, A2C, DDPG

#Local Import
from envs import PortfolioAllocationEnv
from continual_learning import *


# Updated function to train PPO, A2C, and DDPG agents on group1 data
def train_baseline_agents(model_dir, train_df1, group1, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS, transaction_fee_rate=0.001, initial_balance=100000):
    """
    Train PPO, A2C, and DDPG agents on group1 data and save the models.

    Parameters:
    - model_dir: The directory to save the trained models.
    - train_df1: DataFrame containing group1 data for training.
    - group1: List of tickers in group1.
    - iteration: The iteration number (int) for naming the model files.
    - PPO_PARAMS: Hyperparameters for training the PPO agent.
    - A2C_PARAMS: Hyperparameters for training the A2C agent.
    - DDPG_PARAMS: Hyperparameters for training the DDPG agent.
    - transaction_fee_rate: Transaction fee rate for the environment.
    - initial_balance: Initial balance for the portfolio.

    Returns:
    - ppo_model, a2c_model, ddpg_model: Trained PPO, A2C, and DDPG models.
    """
    
    # Create a unique group name based on the iteration number
    group_name = f"baseline_{iteration}"
    
    # Create the environment for group1 data
    env = PortfolioAllocationEnv(df=train_df1, initial_balance=initial_balance, tic_list=group1, transaction_fee_rate=transaction_fee_rate)

    # Train PPO agent
    ppo_model = PPO("MlpPolicy", env, verbose=1, **PPO_PARAMS)
    ppo_model.learn(total_timesteps=50000)  # Adjust timesteps if needed
    ppo_model.save(os.path.join(model_dir, f"ppo_{group_name}"))
    
    # Train A2C agent
    a2c_model = A2C("MlpPolicy", env, verbose=1, **A2C_PARAMS)
    a2c_model.learn(total_timesteps=80000)  # Adjust timesteps if needed
    a2c_model.save(os.path.join(model_dir, f"a2c_{group_name}"))
    
    # Train DDPG agent
    ddpg_model = DDPG("MlpPolicy", env, verbose=1, **DDPG_PARAMS)
    ddpg_model.learn(total_timesteps=50000)  # Adjust timesteps if needed
    ddpg_model.save(os.path.join(model_dir, f"ddpg_{group_name}"))

    # Return trained models
    return ppo_model, a2c_model, ddpg_model

# Function to train agents using the naive strategy on group2
def train_naive_strategy(model_dir, train_df2, group2, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS):
    """
    Train PPO, A2C, and DDPG agents using the naive continual learning strategy.

    Parameters:
    - model_dir: The directory where the pretrained models are saved.
    - train_df2: DataFrame containing group2 data for training.
    - group2: List of tickers in group2.
    - iteration: The iteration number (int) for naming the model files.
    - PPO_PARAMS: Hyperparameters for training the PPO agent.
    - A2C_PARAMS: Hyperparameters for training the A2C agent.
    - DDPG_PARAMS: Hyperparameters for training the DDPG agent.

    Returns:
    - naive_ppo, naive_a2c, naive_ddpg: Trained PPO, A2C, and DDPG models.
    """
    
    # Create a unique group name based on the iteration number
    baseline_name = f"baseline_{iteration}"
    group_name = f"navie_{iteration}"

    # Create the environment for group2 data
    group2_env = PortfolioAllocationEnv(df=train_df2, initial_balance=100000, tic_list=group2, transaction_fee_rate=0.001)

    # Load and train PPO agent using naive strategy
    naive_ppo = PPO.load(os.path.join(model_dir, f"ppo_{baseline_name}"), env=group2_env)
    naive_ppo.learn(total_timesteps=50000)
    naive_ppo.save(os.path.join(model_dir, f"ppo_{group_name}"))

    # Load and train A2C agent using naive strategy
    naive_a2c = A2C.load(os.path.join(model_dir, f"a2c_{baseline_name}"), env=group2_env)
    naive_a2c.learn(total_timesteps=80000)
    naive_a2c.save(os.path.join(model_dir, f"a2c_{group_name}"))

    # Load and train DDPG agent using naive strategy
    naive_ddpg = DDPG.load(os.path.join(model_dir, f"ddpg_{baseline_name}"), env=group2_env)
    naive_ddpg.learn(total_timesteps=50000)
    naive_ddpg.save(os.path.join(model_dir, f"ddpg_{group_name}"))

    # Return the trained agents
    return naive_ppo, naive_a2c, naive_ddpg

# Function to train PPO, A2C, and DDPG agents using the EWC strategy
def train_ewc_agents(model_dir, train_df1, train_df2, group1, group2, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS, lambda_ewc=0.4):
    """
    Train PPO, A2C, and DDPG agents using the EWC strategy.

    Parameters:
    - model_dir: The directory where the pretrained models are saved.
    - train_df1: DataFrame containing group1 data for calculating Fisher Information matrix.
    - train_df2: DataFrame containing group2 data for training with EWC.
    - group1: List of tickers in group1.
    - group2: List of tickers in group2.
    - iteration: The iteration number (int) for naming the model files.
    - PPO_PARAMS: Hyperparameters for training the PPO agent.
    - A2C_PARAMS: Hyperparameters for training the A2C agent.
    - DDPG_PARAMS: Hyperparameters for training the DDPG agent.
    - lambda_ewc: Regularization strength for EWC.

    Returns:
    - ewc_ppo, ewc_a2c, ewc_ddpg: Trained PPO, A2C, and DDPG models using EWC.
    """
    
    # Create a unique group name based on the iteration number
    baseline_name = f"baseline_{iteration}"
    group_name = f"ewc_{iteration}"

    # Create environments for group1 and group2
    group1_env = PortfolioAllocationEnv(df=train_df1, initial_balance=100000, tic_list=group1, transaction_fee_rate=0.001)
    group2_env = PortfolioAllocationEnv(df=train_df2, initial_balance=100000, tic_list=group2, transaction_fee_rate=0.001)

    ## PPO Agent with EWC ##
    ppo_model_group1 = PPO.load(os.path.join(model_dir, f"ppo_{baseline_name}"))

    # Create DataLoader and initialize EWC for PPO
    train_group1_dataloader = create_dataloader(
        agent=ppo_model_group1,
        env=group1_env,
        desired_num_observations=10000,
        batch_size=64,
        use_agent_policy=True
    )
    ewc_ppo = EWC(agent=ppo_model_group1, dataloader=train_group1_dataloader, lambda_=lambda_ewc)

    # Create the EWC_PPO agent
    ewc_ppo_agent = EWC_PPO(
        policy=ppo_model_group1.policy.__class__,
        env=group2_env,
        ewc=ewc_ppo,
        verbose=1
    )

    # Load pre-trained policy weights and train the EWC agent
    ewc_ppo_agent.policy.load_state_dict(ppo_model_group1.policy.state_dict())
    ewc_ppo_agent.learn(total_timesteps=50000)
    ewc_ppo_agent.save(os.path.join(model_dir, f"ppo_{group_name}"))

    ## A2C Agent with EWC ##
    a2c_model_group1 = A2C.load(os.path.join(model_dir, f"a2c_{baseline_name}"))

    # Create DataLoader and initialize EWC for A2C
    train_group1_dataloader = create_dataloader(
        agent=a2c_model_group1,
        env=group1_env,
        desired_num_observations=10000,
        batch_size=64,
        use_agent_policy=True
    )
    ewc_a2c = EWC(agent=a2c_model_group1, dataloader=train_group1_dataloader, lambda_=lambda_ewc)

    # Create the EWC_A2C agent
    ewc_a2c_agent = EWC_A2C(
        policy=a2c_model_group1.policy.__class__,
        env=group2_env,
        ewc=ewc_a2c,
        verbose=1
    )

    # Load pre-trained policy weights and train the EWC agent
    ewc_a2c_agent.policy.load_state_dict(a2c_model_group1.policy.state_dict())
    ewc_a2c_agent.learn(total_timesteps=80000)
    ewc_a2c_agent.save(os.path.join(model_dir, f"a2c_{group_name}"))

    ## DDPG Agent with Deterministic EWC ##
    ddpg_model_group1 = DDPG.load(os.path.join(model_dir, f"ddpg_{baseline_name}"))

    # Create DataLoader for DDPG and initialize Deterministic EWC for the actor network
    train_group1_dataloader = create_dataloader(
        agent=ddpg_model_group1,
        env=group1_env,
        desired_num_observations=10000,
        batch_size=64,
        use_agent_policy=True
    )
    ewc_ddpg = Deterministic_EWC(agent=ddpg_model_group1, dataloader=train_group1_dataloader, lambda_=lambda_ewc)

    # Create the EWC_DDPG agent and load the pre-trained networks
    ewc_ddpg_agent = EWC_DDPG(
        policy=ddpg_model_group1.policy.__class__,
        env=group2_env,
        ewc=ewc_ddpg,
        verbose=1,
    )

    ewc_ddpg_agent.actor.load_state_dict(ddpg_model_group1.actor.state_dict())
    ewc_ddpg_agent.critic.load_state_dict(ddpg_model_group1.critic.state_dict())
    ewc_ddpg_agent.actor_target.load_state_dict(ddpg_model_group1.actor.state_dict())
    ewc_ddpg_agent.critic_target.load_state_dict(ddpg_model_group1.critic.state_dict())

    # Train the EWC_DDPG agent
    ewc_ddpg_agent.learn(total_timesteps=50000)
    ewc_ddpg_agent.save(os.path.join(model_dir, f"ddpg_{group_name}"))

    # Return trained agents
    return ewc_ppo_agent, ewc_a2c_agent, ewc_ddpg_agent

# Function to train PPO, A2C, and DDPG agents using the Buffer Replay strategy
def train_replay_agents(model_dir, train_df1, train_df2, group1, group2, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS):
    from stable_baselines3 import PPO, A2C, DDPG

    """
    Train PPO, A2C, and DDPG agents using the Buffer Replay strategy.

    Parameters:
    - model_dir: The directory where the pretrained models are saved.
    - train_df1: DataFrame containing group1 data for calculating Fisher Information matrix.
    - train_df2: DataFrame containing group2 data for training with Buffer Replay.
    - group1: List of tickers in group1.
    - group2: List of tickers in group2.
    - iteration: The iteration number (int) for naming the model files.
    - PPO_PARAMS: Hyperparameters for training the PPO agent.
    - A2C_PARAMS: Hyperparameters for training the A2C agent.
    - DDPG_PARAMS: Hyperparameters for training the DDPG agent.
    - lambda_ewc: Regularization strength for Buffer Replay.

    Returns:
    - replay_ppo, replay_a2c, replay_ddpg: Trained PPO, A2C, and DDPG models using Buffer Replay.
    """
    
    # Create a unique group name based on the iteration number
    baseline_name = f"baseline_{iteration}"
    group_name = f"replay_{iteration}"

    replay_ppo = perform_replay_training(
        train_df1=train_df1,
        train_df2=train_df2,
        tic_list_group1=group1,
        tic_list_group2=group2,
        total_timesteps=50000,
        agent_class=PPO,
        agent_filename=f"ppo_{baseline_name}",
        model_dir=model_dir,
        model_filename=f"ppo_{group_name}",
        reinforcement_interval=15000,
        reinforcement_steps=3000,
        transaction_fee_rate=0.001,
        initial_balance=100000
    )

    replay_a2c = perform_replay_training(
        train_df1=train_df1,
        train_df2=train_df2,
        tic_list_group1=group1,
        tic_list_group2=group2,
        total_timesteps=80000,
        agent_class=A2C,
        agent_filename=f"a2c_{baseline_name}",
        model_dir=model_dir,
        model_filename=f"a2c_{group_name}",
        reinforcement_interval=15000,
        reinforcement_steps=3000,
        transaction_fee_rate=0.001,
        initial_balance=100000
    )

    replay_ddpg = perform_replay_training(
        train_df1=train_df1,
        train_df2=train_df2,
        tic_list_group1=group1,
        tic_list_group2=group2,
        total_timesteps=50000,
        agent_class=DDPG,
        agent_filename=f"ddpg_{baseline_name}",
        model_dir=model_dir,
        model_filename=f"ddpg_{group_name}",
        reinforcement_interval=15000,
        reinforcement_steps=3000,
        transaction_fee_rate=0.001,
        initial_balance=100000
    )

    return replay_ppo, replay_a2c, replay_ddpg