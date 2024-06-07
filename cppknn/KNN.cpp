#include "KNN.h"
#include "utils.h"
#include <limits>
#include <map>

KNN::KNN(int k) : k(k) {}

void KNN::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    trainData = data;
    trainLabels = labels;
}

int KNN::predict(const std::vector<double>& sample) {
    std::vector<std::pair<double, int>> distances;
    for (size_t i = 0; i < trainData.size(); i++) {
        double dist = euclideanDistance(sample, trainData[i]);
        distances.push_back(std::make_pair(dist, trainLabels[i]));
    }
    sort(distances.begin(), distances.end());

    // 选择最近的k个邻居
    std::map<int, int> counts;
    for (int i = 0; i < k; i++) {
        counts[distances[i].second]++;
    }

    // 找出出现次数最多的标签
    int maxCount = 0;
    int mostFrequentLabel = -1;
    for (auto& count : counts) {
        if (count.second > maxCount) {
            maxCount = count.second;
            mostFrequentLabel = count.first;
        }
    }

    return mostFrequentLabel;
}

double KNN::test(const std::vector<std::vector<double>>& testData, const std::vector<int>& testLabels) {
    int correctCount = 0;
    for (size_t i = 0; i < testData.size(); i++) {
        int predicted = predict(testData[i]);
        if (predicted == testLabels[i]) {
            correctCount++;
        }
    }
    return static_cast<double>(correctCount) / testData.size();
}
