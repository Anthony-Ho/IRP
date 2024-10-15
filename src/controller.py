import os
import subprocess
import tarfile
import pandas as pd
import glob
import sys

# Local Import
from experiment_config import tic_list, result_dir, model_dir
from run_experiment import run_experiment
from run_experiment_with_validation import run_experiment_with_validation

archive_dir = '../archives'

# Ensure archive directory exists
if not os.path.exists(archive_dir):
    os.makedirs(archive_dir)

def archive_models(iteration):
    """
    Archive the models created in the current iteration into a tar file and clean up the models directory.
    """
    # Get the model files for this iteration
    model_files = glob.glob(os.path.join(model_dir, f'*_{iteration}.zip'))
    
    if not model_files:
        print(f"No models found for iteration {iteration}")
        return

    # Create the tar file for archiving
    tar_filename = os.path.join(archive_dir, f'models_iteration_{iteration}.tar.gz')
    with tarfile.open(tar_filename, 'w:gz') as tar:
        for model_file in model_files:
            tar.add(model_file, arcname=os.path.basename(model_file))
            os.remove(model_file)  # Remove the file after archiving

    print(f"Archived models for iteration {iteration} to {tar_filename}")

def run_controller(num_iterations):
    """
    Main controller function to sequentially run experiments, archive models, and collect results.
    
    Parameters:
    - num_iterations: The number of iterations to run
    """
    for i in range(num_iterations):
        # Call run_experiment() to run the experiment and get the iteration number
        iteration = run_experiment_with_validation(combination_file='combinations-viking-retest.csv')

        if iteration is None:
            print("No more combinations left to run.")
            break

        # Archive models for the completed iteration
        archive_models(iteration)


if __name__ == '__main__':
    # Get the number of iterations to run from the command-line argument
    if len(sys.argv) != 2:
        print("Usage: python controller.py <number_of_iterations>")
        sys.exit(1)

    num_iterations = int(sys.argv[1])
    run_controller(num_iterations)