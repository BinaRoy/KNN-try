#ifndef KNN_H
#define KNN_H

#include "KDTree.h"

class KNN {
public:
    KNN(int k);
    void train(const std::vector<std::vector<double>>& trainData, const std::vector<int>& trainLabels);
    double test(const std::vector<std::vector<double>>& testData, const std::vector<int>& testLabels);
private:
    int k;
    KDTree tree;
};

#endif

