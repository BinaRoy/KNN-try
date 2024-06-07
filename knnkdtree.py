# 导入必要的库
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage

# 记录整个程序的开始时间和内存
start_time = time.time()
start_memory = memory_usage(max_usage=True)

# 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化KNN模型
knn = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')

# 记录构建树的开始时间和内存
tree_start_time = time.time()
tree_start_memory = memory_usage(max_usage=True)

# 训练模型
knn.fit(X_train, y_train)

# 记录构建树的结束时间和内存，并计算差值
tree_end_time = time.time()
tree_end_memory = memory_usage(max_usage=True)
tree_time = tree_end_time - tree_start_time
tree_memory = tree_end_memory - tree_start_memory

# 进行预测
predictions = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)

# 记录整个程序的结束时间和内存，并计算差值
end_time = time.time()
end_memory = memory_usage(max_usage=True)
total_time = end_time - start_time
total_memory = end_memory - start_memory

# 输出时间和内存使用情况
print(f"构建树的时间: {tree_time} 秒")
print(f"构建树的内存消耗: {tree_memory} MiB")
print(f"整个程序的运行时间: {total_time} 秒")
print(f"整个程序的内存消耗: {total_memory} MiB")

# 输出准确率
print(f"预测准确率为：{accuracy * 100:.2f}%")
