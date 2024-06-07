import h5py
import numpy as np

def load_data(filepath):
    """
    Load data from an HDF5 file, assuming the last feature (default) as labels.
    
    Parameters:
    - filepath: str, path to the .h5 file.
    - label_index: int, the index of the feature to be used as labels.
    
    Returns:
    - features: np.ndarray, features for each sample.
    - labels: np.ndarray, labels for each sample.
    """
    with h5py.File(filepath, 'r') as file:
        data = file['dat'][:]  # Load the dataset named 'dat'
    
    # Assuming the data is in shape (samples, features)
    # features = np.delete(data, 3 ,axis=0) 
    # print("Features is:",features)
    labels = data[3,:]  # Use the specified column as labels
    print("Labels is :",labels)

    return data, labels


