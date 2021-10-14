#include <iostream>
#include <vector>
#include <iterator>
#include <limits>
#include <random>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <chrono>
#include <string>

#include "open_spiel/algorithms/alpha_zero_torch/puct.h"
using namespace std::chrono;

namespace open_spiel {
namespace algorithms {
namespace torch_az {

void printNode(PUCTNode &node) {
    std::cout << "Action: " << node.action << std::endl
              << "Prior: " << node.prior << std::endl
              << "Exp. Value: " << node.cum_value / node.visit_count << std::endl
              << "UCB Value: " << node.pucb_value << std::endl
              << "Visits: " << node.visit_count << "\n" << std::endl;
}

void writeNode(const PUCTNode &node, int turn_number, float duration, std::string file_name="") {
    std::ofstream os_stream;
    os_stream.open(file_name, std::ios::app);

    os_stream << "PUCT at turn: " << turn_number << "\n" << std::endl;
    os_stream << "Player: " << turn_number % 2 << std::endl;
    os_stream << "Elapsed time: " << duration << "\n" << std::endl;
    for (auto child = node.children.begin(); child < node.children.end(); child++) {
        os_stream << "Action: " << (*child).action << std::endl
                  << "Prior: " << (*child).prior << std::endl
                  << "Exp. Value: " << (*child).cum_value / (*child).visit_count << std::endl
                  << "UCB Value: " << (*child).pucb_value << std::endl
                  << "Visits: " << (*child).visit_count << "\n" << std::endl;
    }
    os_stream << "=====================================\n\n" << std::endl;
    os_stream.close();
}

PUCTNode * PUCT::sample_pucb(std::vector<PUCTNode> &children, int N) {
    std::vector<PUCTNode *> best_children, best_child;
    std::vector<PUCTNode>::iterator child;
    float max_value = -std::numeric_limits<float>::infinity();

    for (child = children.begin(); child < children.end(); child++) {
        if ((*child).visit_count == 0) {
            (*child).pucb_value = std::numeric_limits<float>::infinity();
        } else {
            (*child).pucb_value = (*child).cum_value / (*child).visit_count
                + puct_c * (*child).prior * sqrt(N / (*child).visit_count);
        }

        if ((*child).pucb_value > max_value) {
            best_children.clear();
            best_children.push_back(&(*child));
            max_value = (*child).pucb_value;
        } else if ((*child).pucb_value == max_value) {
            best_children.push_back(&(*child));
        }
    }

    std::sample(best_children.begin(), best_children.end(), 
            std::back_inserter(best_child), 1,
            std::mt19937{std::random_device{}()});

    return best_child.front();
}

float PUCT::traverse(std::unique_ptr<open_spiel::State> &root, PUCTNode &root_node) {
    if (root->IsTerminal()) {
        // std::cout << "Terminal" << std::endl;
        double value;
        if (root->Returns()[0] == 0.0) {
            value = 0.0;
        }
        else {
            value = -1.0;
        }
        root_node.visit_count++;
        root_node.cum_value += -value;

        return -value;
    } else if (root_node.visit_count == 0) {

        auto [children, value] = expand(root);

        root_node.children = children;
        root_node.cum_value += -value;
        root_node.visit_count++;
        
        return -value;
    } else {
        double value;
        // std::cout << "Select" << std::endl;
        PUCTNode *next_node = sample_pucb(root_node.children, root_node.visit_count);
        root->ApplyAction((*next_node).action);
        value = traverse(root, *next_node);
        root_node.cum_value += -value;
        root_node.visit_count++;

        return -value;
    }

    return 0.0;
}

std::tuple<std::vector<PUCTNode>, double> PUCT::expand(std::unique_ptr<open_spiel::State> &root) {
     // std::cout << root << std::endl;
     std::vector<Action> actions;
     actions = root->LegalActions();
     std::vector<Action>::iterator action;
     std::vector<PUCTNode> children;
     VPNetModel::InferenceInputs inputs = {actions, root->ObservationTensor()};
     std::vector<VPNetModel::InferenceOutputs> outputs;
     outputs = model.Inference(std::vector{inputs});
     int num_actions = actions.size();
     double value;

     // std::cout << actions << std::endl;

     for (int i = 0; i < num_actions; i++) {
         PUCTNode *child = new PUCTNode;
         child->action = outputs[0].policy[i].first;
         child->prior = outputs[0].policy[i].second;
         child->visit_count = 0;
         child->cum_value = 0;
         children.push_back(*child);
     }

     value = outputs[0].value;
     if (root->CurrentPlayer() == 1) {
         value = -value;
     }

     return {children, value};
}

PUCTNode * PUCT::select_node(std::vector<PUCTNode> &children) {
    std::vector<PUCTNode *> best_children, best_child;
    std::vector<PUCTNode>::iterator child;
    float max_count = -std::numeric_limits<float>::infinity();

    for (child = children.begin(); child < children.end(); child++) {
        if ((*child).visit_count > max_count) {
            best_children.clear();
            best_children.push_back(&(*child));
            max_count = (*child).visit_count;
        } else if ((*child).visit_count == max_count) {
            best_children.push_back(&(*child));
        }
    }

    std::sample(best_children.begin(), best_children.end(), 
            std::back_inserter(best_child), 1,
            std::mt19937{std::random_device{}()});

    return best_child.front();
}

Action PUCT::search(std::unique_ptr<open_spiel::State> &state, int turn_number, bool verbose, std::string output_file) {
    auto start = high_resolution_clock::now();
    PUCTNode root_node;
    std::unique_ptr<open_spiel::State> root;
    root_node.visit_count = 0;
    root_node.cum_value = 0;

    for (int i = 0; i < budget; i++) {
        root = state->Clone();
        traverse(root, root_node);
    }

    PUCTNode *selection = select_node(root_node.children);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    if (verbose) {
        printNode(*selection);
    }
    writeNode(root_node, turn_number, duration.count(), output_file);

    return (*selection).action;
}

}
}
}
