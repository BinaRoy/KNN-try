#include "KNN.h"

KNN::KNN(int k) : k(k) {}

void KNN::train(const std::vector<std::vector<double>>& trainData, const std::vector<int>& trainLabels) {
    tree.build(trainData, trainLabels);
}

double KNN::test(const std::vector<std::vector<double>>& testData, const std::vector<int>& testLabels) {
    int correct = 0;
    for (size_t i = 0; i < testData.size(); i++) {
        int predicted = tree.nearestNeighbor(testData[i]);
        if (predicted == testLabels[i]) correct++;
    }
    return static_cast<double>(correct) / testData.size();
}

