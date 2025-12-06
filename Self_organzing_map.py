import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# if/else indicator in pandas library version
try: 
    # upload training set
    train_df = pd.read_csv('iris-data.csv', header=None)  # straightforwardly get data
    iris = train_df.iloc[:, :].values   # read data according to the number of columns and rows

    # upload validation set
    val_df = pd.read_csv('iris-labels.csv', header=None)
    label = val_df.iloc[:,:].values
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("please check 'iris-data.csv' and 'iris-labels.csv' are in the SCRIPT DIRECTORY.")
    exit()


max_iris = np.max(iris)
standardized_iris = iris / max_iris
print("Data standardization completed.")

eta_0 = 0.1
d_eta = 0.01  # decay rate for learning rate

sigma_0 = 10
d_sigma = 0.05  # decay rate for neighborhood radius

epochs = 10 

bath_size = 1

output_grid_rows = 40
output_grid_cols = 40
input_features_sim = 4 # number of features in input data  

weights = np.random.rand(output_grid_rows, output_grid_cols, input_features_sim) # initialize weights randomly

num_patterns = 150  # number of data samples

def find_i0j0(sample_x, current_weights):
    difference = sample_x - current_weights
    distances = np.sum(difference ** 2, axis=-1)
    i0, j0 = np.unravel_index(np.argmin(distances), distances.shape)
    return i0, j0

def plot_i0j0_locations(ax, title, data, labels, weights, output_rows, output_cols):
    cmap = ListedColormap(['red', 'green', 'blue'])
    i0j0_locations= {0: [], 1: [], 2: []}

    for i in range(data.shape[0]):
        current_sample = data[i]
        label = int(labels[i])
        i0, j0 = find_i0j0(current_sample, weights)
        i0j0_locations[label].append((j0, i0))  # i is row index, j is column index

    ax.set_title(title)
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_xlim(-0.5, output_cols - 0.5) 
    ax.set_ylim(-0.5, output_rows - 0.5) 
    ax.set_aspect('equal', adjustable='box') 

    for label_val, color_name in zip([0, 1, 2], ['red', 'green', 'blue']):
        locs = np.array(i0j0_locations[label_val])
        if locs.size > 0: 
            ax.scatter(locs[:, 0], locs[:, 1], s=50, c=color_name, label=f'Class {label_val}') 

    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))

fig, axes = plt.subplots(1, 2, figsize=(18, 8)) 
plot_i0j0_locations(axes[0], "Winning Neurons (Initial Weights)", standardized_iris, label, weights, output_grid_rows, output_grid_cols)

print("\ntraning starting")
initial_weights = np.copy(weights) # avoid data overwrite

neuron_positions = np.array(np.indices((output_grid_rows, output_grid_cols))).transpose(1, 2, 0)

for epoch in range(epochs):

    eta = eta_0 * np.exp(-epoch * d_eta)
    sigma = sigma_0 * np.exp(-epoch * d_sigma)

    shuffled_indices = np.random.permutation(num_patterns)  # avoid neuron network learning data sequence

    for i in shuffled_indices:
        x = standardized_iris[i]
        difference = x - weights
        distances = np.sum(difference ** 2, axis=-1)
        i0, j0 = np.unravel_index(np.argmin(distances), distances.shape)

        r_i0 = np.array([i0,j0])

        distance_to_i0j0_squared = np.sum((neuron_positions - r_i0)**2, axis=-1)
        
        if sigma == 0:
            h = np.zeros((output_grid_rows, output_grid_cols))
            h[i0][j0] = 1
        else:
            h = np.exp(-distance_to_i0j0_squared/(2*sigma**2))

        d_weights = eta * h[:,:,np.newaxis] * difference

        weights += d_weights

    print(f"Epoch {epoch + 1}/{epochs} completed.")

print("Training completed.")

plot_i0j0_locations(axes[1], "Winning Neurons (Final Weights)", standardized_iris, label, weights, output_grid_rows, output_grid_cols)

plt.tight_layout(rect=[0, 0, 0.9, 1]) 
plt.suptitle("SOM Clustering of Iris Dataset", y=1.02, fontsize=16) 
plt.show()
