import numpy as np
import pandas as pd
import h5py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from memory_profiler import memory_usage
import time


def load_h5_data(filepath, dataset_name='data'):
    with h5py.File(filepath, 'r') as file:
        data = file[dataset_name][:]
    print("Data shape :", data.shape)
    
    # 确保数据以样本为列（features in rows）
    # 我们需要转置数据，因为通常机器学习库期望样本为行，特征为列
    data = data.T  # Transpose to make samples as columns if needed
    
    # 前三列作为特征
    features = data[:, :3]  # Take first three rows as features
    # 最后一列作为标签
    labels = data[:, -1]    # Take last row as labels
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    
    return features, labels

def preprocess_labels(labels):
    """
    Convert continuous time values into discrete class labels.
    """
    label_series = pd.Series(labels).astype('category')
    return label_series.cat.codes

def run_knn(X_train, y_train, n_neighbors=3, algorithm='auto'):
    """
    Train KNN classifier and measure performance.
    """
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    memory_usage_start = memory_usage(max_usage=True)  # This is a float
    knn.fit(X_train, y_train)
    memory_usage_end = memory_usage(max_usage=True)  # This is also a float
    end_time = time.time()

    return {
        'time_taken': end_time - start_time,
        'memory_used': memory_usage_end - memory_usage_start,  # Correct calculation
        'classifier': knn
    }

def main():
    filepath = '/home/tkvr85/AMD/pyknn/data/filaments/filaments_clean.h5'  # Update the path to your .h5 file
    dataset_name = 'dat'  # Update the dataset name if different

    features, times = load_h5_data(filepath, dataset_name)
    labels = preprocess_labels(times)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    results = run_knn(X_train, y_train, n_neighbors=5, algorithm='kd_tree')

    # Make predictions
    predictions = results['classifier'].predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Time taken: {results['time_taken']:.4f} seconds")
    print(f"Memory used: {results['memory_used']:.4f} MiB")

if __name__ == "__main__":
    main()
