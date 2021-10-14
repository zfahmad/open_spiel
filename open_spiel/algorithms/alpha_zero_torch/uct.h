#ifndef UCT_H
#define UCT_H

#include <iostream>
#include <vector>
#include <iterator>
#include <limits>
#include <string>
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {
    
struct UCTNode {
    Action action;
    float visit_count;
    float cum_value;
    float ucb_value;
    std::vector<UCTNode> children;
};

void printNode(UCTNode &node);
void writeNode(const UCTNode &node, int turn_number, float duration, std::string file_name);

class UCT {
private:
    float uct_c;
    int budget;
public:
    UCT(float c, int b)
        : uct_c(c), budget(b) {}
    Action search(std::unique_ptr<open_spiel::State> &state, int turn_number, bool verbose=false, std::string output_file="");
    UCTNode * select_lcb(std::vector<UCTNode> &children, int N);
    UCTNode * sample_ucb(std::vector<UCTNode> &children, int N);
    float traverse(std::unique_ptr<open_spiel::State> &root, UCTNode &root_node);
    std::vector<UCTNode> expand(std::unique_ptr<open_spiel::State> &root);
    float rollout(std::unique_ptr<open_spiel::State> &root);
    float get_c() { return uct_c; }
    int get_budget() { return budget; }
};

}
}
}

#endif
