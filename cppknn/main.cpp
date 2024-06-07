#include <iostream>
#include "KNN.h"
#include "utils.h"

int main() {
    // 加载数据
    std::vector<std::vector<double>> data;
    std::vector<int> labels;
    loadData(data, labels);

    // 数据集分割
    std::vector<std::vector<double>> trainData, testData;
    std::vector<int> trainLabels, testLabels;
    splitData(data, labels, trainData, trainLabels, testData, testLabels, 0.8);

    // 创建KNN模型
    KNN knn(3); // 使用k=3
    knn.train(trainData, trainLabels);

    // 测试模型
    double accuracy = knn.test(testData, testLabels);
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}

