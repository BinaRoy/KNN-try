from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from memory_profiler import memory_usage
import time

def train_model(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')
    tree_start_time = time.time()
    tree_start_memory = memory_usage(max_usage=True)
    knn.fit(X_train, y_train)
    tree_end_time = time.time()
    tree_end_memory = memory_usage(max_usage=True)
    tree_time = tree_end_time - tree_start_time
    tree_memory = tree_end_memory - tree_start_memory
    return knn, tree_time, tree_memory

def test_model(knn, X_test, y_test):
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

