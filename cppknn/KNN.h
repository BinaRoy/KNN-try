#ifndef KNN_H
#define KNN_H

#include <vector>

class KNN {
public:
    KNN(int k);
    void train(const std::vector<std::vector<double>>& trainData, const std::vector<int>& trainLabels);
    int predict(const std::vector<double>& sample);
    double test(const std::vector<std::vector<double>>& testData, const std::vector<int>& testLabels);

private:
    int k;
    std::vector<std::vector<double>> trainData;
    std::vector<int> trainLabels;
};

#endif