#ifndef PUCT_H
#define PUCT_H

#include <iostream>
#include <vector>
#include <iterator>
#include <limits>
#include <tuple>
#include <string>
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {
    
struct PUCTNode {
    Action action;
    float prior;
    int visit_count;
    float cum_value;
    float pucb_value;
    std::vector<PUCTNode *> children;
};

void printNode(PUCTNode &node);
void writeNode(const PUCTNode &node, int turn_number, float duration, std::string file_name);

class PUCT {
private:
    float puct_c;
    int budget;
    VPNetModel &model;
public:
    PUCT(float c, int b, VPNetModel &model)
        : puct_c(c), budget(b), model(model) {}
    Action search(std::unique_ptr<open_spiel::State> &state, int turn_number, bool verbose, std::string output_file);
    PUCTNode * select_node(std::vector<PUCTNode *> &children);
    PUCTNode * sample_pucb(std::vector<PUCTNode *> &children, int N);
    float traverse(std::unique_ptr<open_spiel::State> &root, PUCTNode &root_node);
    std::tuple<std::vector<PUCTNode *>, double> expand(std::unique_ptr<open_spiel::State> &root);
    void delete_tree(PUCTNode &root_node);
    float get_c() { return puct_c; }
    int get_budget() { return budget; }

};

}
}
}

#endif
