#include <iostream>
#include <vector>
#include <chrono>
#include "KNN.h"
#include "utils.h"

int main() {
    // 数据加载和预处理
    std::vector<std::vector<double>> data;
    std::vector<int> labels;
    loadData(data, labels);  // 假设这个函数从文件中加载数据和标签

    // 数据集分割
    std::vector<std::vector<double>> trainData, testData;
    std::vector<int> trainLabels, testLabels;
    splitData(data, labels, trainData, trainLabels, testData, testLabels, 0.8);

    // 创建 KNN 实例
    int k = 3;
    KNN knn(k);

    // 开始构建 KD 树，并测量时间
    auto start = std::chrono::high_resolution_clock::now();
    knn.train(trainData, trainLabels);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "KD Tree build time: " << elapsed.count() << " seconds.\n";

    // 测试模型
    start = std::chrono::high_resolution_clock::now();
    double accuracy = knn.test(testData, testLabels);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Query time: " << elapsed.count() << " seconds.\n";
    std::cout << "Accuracy: " << accuracy * 100 << "%\n";

    // KD树统计信息
    int size;
    double buildTime;
    knn.tree.getStats(size, buildTime); // 假设这个方法能获取树的大小和构建时间
    std::cout << "KD Tree size: " << size << " nodes\n";
    std::cout << "KD Tree build time recorded in class: " << buildTime << " seconds\n";

    return 0;
}
