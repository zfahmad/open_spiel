#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <chrono>

#include "open_spiel/algorithms/alpha_zero_torch/lts.h"
using namespace std::chrono;

namespace open_spiel {
namespace algorithms {
namespace torch_az {

void printNode(const LTSNode &node) {
    std::cout << "Action: " << node.action << std::endl
              << "Depth: " << node.depth << std::endl
              << "Minimax Value: " << node.minimax_val << std::endl
              << "Predicted Value: " << node.pred_val << std::endl
              << "Actor RP: " << node.actor_rp << std::endl
              << "Eventual RP: " << node.eventual_rp << std::endl
              << "Cost: " << node.cost << std::endl
              << "Terminal: " << node.terminal << "\n" << std::endl;
}

void writeNode(const LTSNode &node, int turn_number, float duration, std::string file_name="") {
    std::ofstream os_stream;
    os_stream.open(file_name, std::ios::app);

    os_stream << "LTS at turn: " << turn_number << "\n" << std::endl;
    os_stream << "Player: " << turn_number % 2 << std::endl;
    os_stream << "Elapsed time: " << duration << "\n" << std::endl;
    for (auto child = node.children.begin(); child < node.children.end(); child++) {
        os_stream << "Action: " << (*child).action << std::endl
                  << "Depth: " << (*child).depth << std::endl
                  << "Minimax Value: " << (*child).minimax_val << std::endl
                  << "Predicted Value: " << (*child).pred_val << std::endl
                  << "Actor RP: " << (*child).actor_rp << std::endl
                  << "Eventual RP: " << (*child).eventual_rp << std::endl
                  << "Cost: " << (*child).cost << std::endl
                  << "Terminal: " << (*child).terminal << "\n" << std::endl;
    }
    os_stream << "=====================================\n\n" << std::endl;
    os_stream.close();
}

std::vector<LTSNode> LTS::expand(std::unique_ptr<open_spiel::State> &root, LTSNode &root_node) {
     std::vector<Action> actions;
     actions = root->LegalActions();
     std::vector<Action>::iterator action;
     std::vector<LTSNode> children;
     VPNetModel::InferenceInputs inputs = {actions, root->ObservationTensor()};
     std::vector<VPNetModel::InferenceOutputs> outputs;
     outputs = model.Inference(std::vector{inputs});
     float num_actions = actions.size();

     // std::cout << outputs[0].policy << std::endl;

     for (int i = 0; i < num_actions; i++) {
         LTSNode *child = new LTSNode;
         child->action = outputs[0].policy[i].first;
         child->visited = false;
         child->actor_rp = root_node.eventual_rp + log(outputs[0].policy[i].second);
         child->eventual_rp = root_node.actor_rp + log(1.0 / num_actions);
         child->depth = root_node.depth + 1;
         child->cost = log(child->depth) - std::max(child->actor_rp, child->eventual_rp);
         child->minimax_val = 99.0;
         child->terminal = false;
         children.push_back(*child);
     }

     root_node.pred_val = outputs[0].value;
     if (root->CurrentPlayer() == 1) {
         root_node.pred_val = -root_node.pred_val;
     }

     return children;
}

float LTS::traverse(std::unique_ptr<open_spiel::State> &root, LTSNode &root_node) {
    float value = -std::numeric_limits<float>::infinity();
    float child_val;
    std::unique_ptr<open_spiel::State> next_state;
    // std::cout << root << std::endl;

    if (root->IsTerminal()) {
        // std::cout << "Terminal" << std::endl;
        if (root->Returns()[0] == 0.0) {
            value = 0.0;
        }
        else {
            value = -1.0;
        }
        // std::cout << "Returning" << std::endl;
    } else {
        if (!root_node.visited) {
            // std::cout << "Is terminal: " << root->IsTerminal() << std::endl;
            root_node.children = expand(root, root_node);
            search_count++;
            root_node.visited = true;
            if (search_count > budget) {
                terminate = true;
                return root_node.minimax_val;
            }
        }

        bool has_children = false;
        for (auto child = root_node.children.begin(); child < root_node.children.end(); child++) {
            // std::cout << (*child).cost << " " << current_bound << std::endl;
            if (child->cost > current_bound)
                next_bound = std::min(next_bound, child->cost);
            if ((*child).cost <= current_bound && terminate == false) {
                has_children = true;
                next_state = root->Clone();
                next_state->ApplyAction((*child).action);
                child_val = traverse(next_state, *child);
                // traverse(next_state, *child, model);
                value = std::max(child_val, value);
            }
        }

        if (has_children == false) {
            value = root_node.pred_val;
        }
    }
    if (terminate == true)
        return root_node.minimax_val;

    root_node.minimax_val = -value;

    return -value;
}

void LTS::build(std::unique_ptr<open_spiel::State> &root, LTSNode &root_node) {
    std::unique_ptr<open_spiel::State> next_state;
    float value;

    if (root->IsTerminal()) {
        if (root->Returns()[0] == 0.0) {
            value = 0.0;
        }
        else {
            value = -1.0;
        }
        root_node.terminal = true;
        root_node.minimax_val = -value;
    } else {
        if (root_node.visited == false) {
            root_node.children = expand(root, root_node);
            search_count++;
            if (search_count > budget)
                terminate = true;
            root_node.visited = true;
        }

        for (auto child = root_node.children.begin(); child < root_node.children.end(); child++) {
            if (child->cost > current_bound)
                next_bound = std::min(next_bound, child->cost);
            if ((*child).cost <= current_bound && terminate == false) {
                next_state = root->Clone();
                next_state->ApplyAction((*child).action);
                build(next_state, *child);
            }
        }
    }
}

float LTS::minimax(LTSNode &root_node, float bound) {
    float child_val, value = -std::numeric_limits<float>::infinity();
    bool has_children = false;
    // printNode(root_node);

    if (root_node.terminal) {
        // std::cout << "Reached terminal" << std::endl;
        value = root_node.minimax_val;
        return value;
    }

    for (auto child = root_node.children.begin(); child < root_node.children.end(); child++) {
        // std::cout << "Looking at children" << std::endl;
        if ((*child).cost <= bound) {
            has_children = true;
            child_val = minimax(*child, bound);
            value = std::max(child_val, value);
        }
    }

    if (has_children == false) {
        value = root_node.pred_val;
    }

    root_node.minimax_val = -value;
    return -value;
}

LTSNode * LTS::select_best(std::vector<LTSNode> &children) {
    std::vector<LTSNode *> best_children, best_child;
    std::vector<LTSNode>::iterator child;
    float max_val = -std::numeric_limits<float>::infinity();

    for (child = children.begin(); child < children.end(); child++) {
        if ((*child).minimax_val > max_val) {
            best_children.clear();
            best_children.push_back(&(*child));
            max_val = (*child).minimax_val;
        } else if ((*child).minimax_val == max_val) {
            best_children.push_back(&(*child));
        }
    }

    std::sample(best_children.begin(), best_children.end(), 
            std::back_inserter(best_child), 1,
            std::mt19937{std::random_device{}()});

    return best_child.front();
}

Action LTS::search(std::unique_ptr<open_spiel::State> &state, int turn_number, bool verbose=false, std::string output_file="") {
    auto start = high_resolution_clock::now();
    LTSNode root_node;
    root_node.action = NULL;
    root_node.depth = 1;
    root_node.actor_rp = 0.0;
    root_node.eventual_rp = 0.0;
    root_node.cost = 0.0;
    root_node.visited = false;
    root_node.terminal = false;
    std::unique_ptr<open_spiel::State> root;
    std::vector<LTSNode> current_candidates;
    current_bound = 0.0;
    // current_bound = std::numeric_limits<float>::infinity();
    next_bound = std::numeric_limits<float>::infinity();
    terminate = false;
    float bound, value;

    search_count = 0;
//     for (int i = 0; i < 4; i++) {
    while (next_bound != current_bound && !terminate) {
        root = state->Clone();
//        build(root, root_node);
        value = traverse(root, root_node);
        if (verbose) {
            std::cout << "Current Bound: " << current_bound << " "
                      << "Next Bound: " << next_bound << " " 
                      << "Num Leaves: " << search_count << std::endl;
        }
        if (!terminate) {
            bound = current_bound;
            current_candidates = root_node.children;
        }

        current_bound += log(sqrt(2));
        if (next_bound > current_bound)
            current_bound = next_bound;
        next_bound = std::numeric_limits<float>::infinity();
        // for (auto child = root_node.children.begin(); child < root_node.children.end(); child++) {
        //     printNode((*child));
        // }
    }

//    if (verbose)
//        std::cout << "Running minimax on LTS tree with bound: " << bound << std::endl;
//    root = state->Clone();
//    value = minimax(root_node, bound);
    auto stop = high_resolution_clock::now();
    if (verbose) {
        std::cout << "Finished search over with minimax value: " << -value << std::endl;
        for (auto child = current_candidates.begin(); child < current_candidates.end(); child++) {
            printNode((*child));
        }
    }

    LTSNode *selection = select_best(current_candidates);

    auto duration = duration_cast<seconds>(stop - start);
    writeNode(root_node, turn_number, duration.count(), output_file);

    return (*selection).action;
}

}
}
}
