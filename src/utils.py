import os
import random
import pandas as pd

# Local Import
from  config import data_dir

def prebuild_group_combinations(tic_list, num_iterations, output_file='combinations.csv'):
    """
    Pre-build random group1 and group2 combinations and save to a CSV file.
    Ensures group1 and group2 are sorted and that no identical group1 is added to the combinations.
    
    Parameters:
    - tic_list: List of stock tickers (e.g., DJI stocks).
    - num_iterations: Number of random group1 and group2 combinations to generate.
    - output_file: File to save the generated combinations.

    Returns:
    - None (Writes the combinations to a CSV file).
    """
    combinations = []
    seen_group1 = set()  # Set to track unique group1 combinations

    for iteration in range(num_iterations):
        while True:  # Loop to ensure unique group1
            random.shuffle(tic_list)
            mid_point = len(tic_list) // 2
            group1 = sorted(tic_list[:mid_point])  # Sort group1
            group2 = sorted(tic_list[mid_point:])  # Sort group2
            
            # Convert group1 to tuple so it can be stored in a set
            group1_tuple = tuple(group1)
            
            # If group1 is unique, break out of the loop
            if group1_tuple not in seen_group1:
                seen_group1.add(group1_tuple)
                break
        
        # Add the valid combination to the list with initial status "untrained"
        combinations.append({
            'iteration': iteration,
            'group1': group1,
            'group2': group2,
            'status': "untrained"  # Status can be "untrained", "training", or "trained"
        })

    # Save to CSV
    pd.DataFrame(combinations).to_csv(os.path.join(data_dir, output_file), index=False)
    print(f"Unique group combinations with status saved to {output_file}")

def get_next_combination(csv_file='combinations.csv', container_id=None):
    """
    Get the next untrained group1 and group2 combination from a CSV file and mark it as "training".
    
    Parameters:
    - csv_file: Path to the CSV file storing the combinations.
    - container_id: ID of the container or process taking up the combination (optional).
    
    Returns:
    - group1, group2: Lists of tickers for group1 and group2.
    """
    # Load the CSV file
    df = pd.read_csv(os.path.join(data_dir, csv_file))

    # Find the next untrained combination
    untrained_row = df[df['status'] == 'untrained'].iloc[0]

    # Extract group1 and group2 from the row
    group1 = eval(untrained_row['group1'])  # Convert string representation of list to actual list
    group2 = eval(untrained_row['group2'])  # Convert string representation of list to actual list

    # Mark this combination as "training"
    df.loc[df['iteration'] == untrained_row['iteration'], 'status'] = 'training'

    # Save the updated CSV
    df.to_csv(os.path.join(data_dir, csv_file), index=False)

    return untrained_row['iteration'], group1, group2

