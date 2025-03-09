import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

#model = "attention"	#"astar"
#filename = "/share/$GROUP/$USER/GenAI-for-Systems-Gym/attention/experiments/evaluate_astar_parrot_attention/" + model + ".pkl"
filename = "attention.pkl"

def load_metadata(filename):
    """
    Load metadata from a pickle file and return it as a list of dictionaries.
    
    Args:
    - filename (str): Path to the pickle file.

    Returns:
    - list: List of dictionaries containing metadata if file exists, else an empty list.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print("File not found.")
        return []
    

def process_metadata(metadata_list, num_repetitions=30):
    # Initialize a list to store the filtered data
    filtered_data = []
    
    # Get the current entry (the last entry in the metadata list)
    target = len(metadata_list)-1
    next_entry = None
    
    # Check that access_history has at least one entry
    # if len(curr_entry["access_history"]) == 0:
    #     print("Access history is empty.")
    #     return filtered_data

    
    # Loop 30 times to perform the search and appending
    for i in range(30):
        # Get the address and pc from the first entry in access_history
        # first_access = curr_entry["access_history"][0]
        # target_address, target_pc = first_access  # Unpack tuple assuming it's (address, pc)
        
        
        # Search from the bottom to find the first matching entry for pc and address
        # for entry in reversed(metadata_list[:metadata_list.index(curr_entry)]):  # Exclude the current entry from search
        #     if entry["pc"] == target_pc and entry["address"] == target_address:
        #         # Find the index of the minimum value in the 'scores' list
        #         min_score_index = np.argmin(entry["scores"])
        #         # Append the corresponding attention weight from 'curr_entry' to filtered_data
        #         filtered_data.append(curr_entry["attention_weights"][min_score_index])
        #         next_entry = entry
        #         break  # Stop once the first match is found from the bottom
        curr_entry = metadata_list[target-i]
        min_score_index = np.argmin(curr_entry["scores"])
        filtered_data.append(curr_entry["attention_weights"][min_score_index])
        

    return filtered_data

def plot_heatmap(filtered_data):
    # Stack the list of arrays into a 2D numpy array
    heatmap_data = np.vstack(filtered_data)
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, aspect='auto', cmap='gray')
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Source Offset")
    plt.ylabel("Target Offset")
    plt.title("2D Heatmap of Attention Weights")
    output_file_path = "./attention_plot.png"
#    output_file_path = "C:/Users/Avit/Documents/NCSU Notes/Fall 24/592/Proj/attention plots/" + model + ".png"
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    plt.show()

# Load the metadata (assuming load_metadata is defined)
metadata_list = load_metadata(filename)

# Process the metadata to get the filtered data
filtered_data = process_metadata(metadata_list)

# Print the filtered data
plot_heatmap(filtered_data)

    
