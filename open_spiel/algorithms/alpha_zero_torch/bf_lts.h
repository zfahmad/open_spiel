#ifndef BF_LTS_H
#define BF_LTS_H

#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"

namespace open_spiel::algorithms::torch_az {

class BFSNode {
public:
    Action action;
    std::unique_ptr<open_spiel::State> state;
    BFSNode *parent;
    int depth;
    float actor_rp;
    float eventual_rp;
    float cost;
    float pred_val;
    float minimax_val;
    bool terminal;
    std::vector<BFSNode*> children;
    BFSNode() {
        action = NULL;
        state = nullptr;
        parent = nullptr;
        depth = 0;
        actor_rp = 0.0;
        eventual_rp = 0.0;
        cost = 0.0;
        pred_val = -std::numeric_limits<float>::infinity();
        minimax_val = -std::numeric_limits<float>::infinity();
        terminal = false;
    };
};

struct MyComparator {
    bool operator() (BFSNode *arg_1, BFSNode *arg_2) {
        return arg_1->cost > arg_2->cost;
    }
};

void printNode(const BFSNode &node);
void writeNode(const BFSNode &node, int turn_number, float duration, std::string file_name);

class BFLTS {
private:
    int budget;
    VPNetModel &model;
    std::shared_ptr<const Game> &game;
public:
    BFLTS(std::shared_ptr<const Game> &game, int budget, VPNetModel &model)
        : game(game), budget(budget), model(model) {}
    std::priority_queue<BFSNode*, std::vector<BFSNode*>, MyComparator> pq;
    void generate_children(BFSNode &root_node);
    BFSNode * search(std::unique_ptr<open_spiel::State> &state, int turn_number, bool verbose, std::string output_file);
    BFSNode * select_best(std::vector<BFSNode *> &children);
    BFSNode * build_tree(std::unique_ptr<open_spiel::State> &state);
    void delete_tree(BFSNode *root_node);
    float minimax(BFSNode &root_node);
};

bool operator< (const BFSNode &node_1, const BFSNode &node_2);

bool operator> (const BFSNode &node_1, const BFSNode &node_2);

}

#endif
