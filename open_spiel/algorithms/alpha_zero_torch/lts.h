#ifndef LTS_H
#define LTS_H

#include <iostream>
#include <vector>
#include <string>
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

struct LTSNode {
    Action action;
    bool visited;
    int depth;
    float actor_rp;
    float eventual_rp;
    float cost;
    float pred_val;
    float minimax_val;
    bool terminal;
    std::vector<LTSNode *> children;
};

void printNode(const LTSNode &node);
void writeNode(const LTSNode &node, int turn_number, float duration, std::string file_name);

class LTS {
private:
    int budget;
    int search_count;
    bool terminate;
    float current_bound;
    float next_bound;
    VPNetModel &model;
public:
    LTS(int b, VPNetModel &model)
        : budget(b), model(model) {}
    std::vector<LTSNode *> expand(std::unique_ptr<open_spiel::State> &root, LTSNode &root_node);
    float traverse(std::unique_ptr<open_spiel::State> &root, LTSNode &root_node);
    void build(std::unique_ptr<open_spiel::State> &root, LTSNode &root_node);
    float minimax(LTSNode &root_node, float bound);
    LTSNode * select_best(std::vector<LTSNode *> &children);
    Action search(std::unique_ptr<open_spiel::State> &state, int turn_number, bool verbose, std::string output_file);
    void delete_tree(LTSNode &root_node);
};

}
}
}

#endif
