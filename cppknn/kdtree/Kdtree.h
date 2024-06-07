#ifndef KDTREE_H
#define KDTREE_H

#include <vector>
#include <memory>

struct KDNode {
    std::vector<double> point;
    int label;
    std::shared_ptr<KDNode> left;
    std::shared_ptr<KDNode> right;
    KDNode(std::vector<double> pt, int lbl) : point(pt), label(lbl), left(nullptr), right(nullptr) {}
};

class KDTree {
public:
    KDTree();
    void build(const std::vector<std::vector<double>>& points, const std::vector<int>& labels);
    int nearestNeighbor(const std::vector<double>& point);
    void getStats(int& size, double& buildTime);

private:
    std::shared_ptr<KDNode> root;
    int dimensions;
    std::shared_ptr<KDNode> buildRec(const std::vector<std::pair<std::vector<double>, int>>& data, int depth);
    int nnRec(const std::shared_ptr<KDNode>& node, const std::vector<double>& point, double& bestDist, int depth);
    double distanceSquared(const std::vector<double>& point1, const std::vector<double>& point2);
};

#endif

