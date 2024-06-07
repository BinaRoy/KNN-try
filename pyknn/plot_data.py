import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data(filepath):
    
    with h5py.File(filepath, 'r') as file:
        data = file['dat'][:]  # Load the dataset named 'dat'
    
    # # Assuming the data is in shape (samples, features)
    # features = np.delete(data, label_index, axis=1)  # Remove the label column from the data
    # labels = data[:, label_index]  # Use the specified column as labels
    
    # return features, labels

    # # Assuming the data is stored in a CSV format without headers
    # data = np.loadtxt(filepath, delimiter=',')
    return data

def plot_data(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Coordinates
    xs = data[0, :]
    ys = data[1, :]
    zs = data[2, :]
    # Time as color
    time = data[3, :]

    scatter = ax.scatter(xs, ys, zs, c=time, cmap='viridis')
    fig.colorbar(scatter, ax=ax, label='Time')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.title('3D Spatial-Temporal Data Visualization')
    plt.show()

if __name__ == "__main__":
    import sys
    filepath = '/home/tkvr85/AMD/pyknn/data/filaments/filaments_clean.h5'
    data = load_data(filepath)
    plot_data(data)

