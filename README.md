# Continual Learning in Reinforcement Learning-Based Portfolio Management: A Study on Adaptability and Knowledge Retention

## Overview

This project explores the application of Continual Learning (CL) techniques to Reinforcement Learning (RL) agents in the context of portfolio optimization. Using RL algorithms (PPO, A2C, and DDPG) and CL strategies (Naive, EWC, Replay Buffer), this research evaluates the agents' adaptability and stability across different stock groups. This artefact includes the source code, Docker configurations, and supporting resources for reproducing the experiment.

## Project Structure
```plaintext
.
├── archives/                    # Archived data and backups
├── data/                        # Directory for stock data and combinations files
├── docker/                      # Docker configurations and Dockerfile
├── models/                      # Output directory for saved trained models
├── results/                     # Output directory for experiment results
├── src/                         # Source code for running experiments
│   ├── __init__.py
│   ├── combine_results.sh
│   ├── compare_results.py
│   ├── continual_learning.py
│   ├── controller.py
│   ├── data_processing.py
│   ├── envs.py
│   ├── experiment_config.py
│   ├── experiment_utils.py
│   ├── performance.py
│   ├── requirements.txt
│   ├── run_experiment_with_validation.py
│   ├── training.py
│   └── viking-crl.job
├── .gitignore
├── LICENSE
├── README.md                    # This README file
└── run_docker_experiment.sh
```

## Prerequisites

- Docker with NVIDIA GPU support and drivers installed
- Python 3.9+ if running locally
- NVIDIA Docker Toolkit for GPU support in Docker

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Anthony-Ho/IRP.git
   cd IRP
   ```

2. **Build the Docker Image**: Navigate to the docker directory and build the Docker image:
    ```bash
    docker build -t rl-portfolio-optimization -f docker/Dockerfile .
    ```
3. **Prepare Data and Results Directories**: Ensure the data/ directory contains necessary data files, including the `combinations.csv` file for running the experiment.

4. **Run Experiments with Shell Script**: The project includes a shell script for running experiments with Docker. Use this to specify the combination file and the number of iterations.

    ```bash
    ./run_docker_experiment.sh [combination_file] [iterations]
    ```
    - `combination_file`: Optional. The filename to the combination file within `data/`. Defaults to `combinations.csv`.  The combination file must be located in `data/`.
    - `iterations`: Optional. Number of iterations to run. Defaults to 1.
    
    For example:
    ```bash
    ./run_docker_experiment.sh combinations-viking.csv 5
    ```
## Running Experiments
### Running a Single Experiment
You can run the experiment manually with Docker by using the `run_experiment_with_validation.py` script:

```bash
docker run --gpus all -it \
  -v "$PWD/src:/workspace/IRP/src" \
  -v "$PWD/data:/workspace/IRP/data" \
  -v "$PWD/results:/workspace/IRP/results" \
  rl-portfolio-optimization --combination_file "combinations-viking.csv"
```

**Running Multiple Iterations**

The shell script `run_docker_experiment.sh` simplifies running multiple experiments in a loop. Adjust the number of iterations as needed.

## Files and Resources
- `requirements.txt`: List of dependencies for the Python environment.
- `run_experiment_with_validation.py`: Main experiment script, which trains and validates models using specified combination files.
- `data_processing.py`, `continual_learning.py`, `training.py`: Core modules for data preparation, continual learning strategies, and model training.
- `viking-crl.job`: Example job script for running the experiment on the Viking HPC cluster.

## Additional Notes
GPU Support: The Docker setup assumes NVIDIA GPU support. Ensure drivers and CUDA are correctly configured.
Output: Experiment results are saved in the `results/` directory. Performance metrics and returns are logged in CSV files for each iteration.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments
This project builds upon the FinRL library and other related works in Reinforcement Learning and Financial Modeling.






