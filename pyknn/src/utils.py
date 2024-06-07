import pandas as pd

def save_results(accuracy, tree_time, tree_memory, total_time, total_memory, leaf_size):
    results = pd.DataFrame({
        'Accuracy': [accuracy],
        'Tree Build Time (s)': [tree_time],
        'Tree Memory Usage (MiB)': [tree_memory],
        'Total Time (s)': [total_time],
        'Total Memory Usage (MiB)': [total_memory],
        'Leaf Size': [leaf_size]
    })
    results.to_csv('knn_performance.csv', index=False)
    print("Results saved to knn_performance.csv")
