#include <iostream>
#include <vector>
#include <iterator>
#include <limits>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
using namespace std::chrono;

#include "open_spiel/algorithms/alpha_zero_torch/uct.h"

namespace open_spiel::algorithms::torch_az {

// UCT::UCT(float c, int b) {
//     uct_c = c;
//     budget = b;
// };

void printNode(UCTNode &node) {
    std::cout << "Action: " << node.action << std::endl
              << "Exp. Value: " << node.cum_value / node.visit_count << std::endl
              << "UCB Value: " << node.ucb_value << std::endl
              << "Visits: " << node.visit_count << "\n" << std::endl;
}

void writeNode(const UCTNode &node, int turn_number, float duration, std::string file_name="") {
    std::ofstream os_stream;
    os_stream.open(file_name, std::ios::app);

    os_stream << "UCT at turn: " << turn_number << std::endl;
    os_stream << "Player: " << turn_number % 2 << std::endl;
    os_stream << "Elapsed time: " << duration << "\n" << std::endl;
    for (auto child = node.children.begin(); child < node.children.end(); child++) {
        os_stream << "Action: " << (*child)->action << std::endl
                  << "Exp. Value: " << ((*child)->cum_value / (*child)->visit_count) << std::endl
                  << "Cum. Value: " << (*child)->cum_value << std::endl
                  << "UCB Value: " << (*child)->ucb_value << std::endl
                  << "Visits: " << (*child)->visit_count << "\n" << std::endl;
    }
    os_stream << "=====================================\n\n" << std::endl;
    os_stream.close();
}

UCTNode * UCT::sample_ucb(std::vector<UCTNode *> &children, int N) {
    std::vector<UCTNode *> best_children, best_child;
    std::vector<UCTNode *>::iterator child;
    float max_value = -std::numeric_limits<float>::infinity();

    for (child = children.begin(); child < children.end(); child++) {
        if ((*child)->visit_count == 0.0) {
            (*child)->ucb_value = std::numeric_limits<float>::infinity();
        } else {
            (*child)->ucb_value = (*child)->cum_value / (*child)->visit_count
                + uct_c * sqrt(2 * log(N) / (*child)->visit_count);
        }

        if ((*child)->ucb_value > max_value) {
            best_children.clear();
            best_children.push_back((*child));
            max_value = (*child)->ucb_value;
        } else if ((*child)->ucb_value == max_value) {
            best_children.push_back((*child));
        }
    }

    std::sample(best_children.begin(), best_children.end(), 
            std::back_inserter(best_child), 1,
            std::mt19937{std::random_device{}()});

    return best_child.front();
}

float UCT::traverse(std::unique_ptr<open_spiel::State> &root, UCTNode &root_node) {
    if (root->IsTerminal()) {
        // std::cout << "Terminal" << std::endl;
        float value;
        if (root->Returns()[0] == 0.0) {
            value = 0.0;
        }
        else {
            value = -1.0;
        }
        root_node.visit_count += 1.0;
        root_node.cum_value += -value;

        return -value;
    } else if (root_node.visit_count == 0.0) {
        float value;

        root_node.children = expand(root);
        value = rollout(root);

        root_node.cum_value = -value;
        root_node.visit_count += 1.0;
        
        return -value;
    } else {
        float value;
        // std::cout << "Select" << std::endl;
        UCTNode *next_node = sample_ucb(root_node.children, root_node.visit_count);
        root->ApplyAction((*next_node).action);
        value = traverse(root, *next_node);
        root_node.cum_value += -value;
        root_node.visit_count += 1.0;

        return -value;
    }

    return 0.0;
}

std::vector<UCTNode *> UCT::expand(std::unique_ptr<open_spiel::State> &root) {
     // std::cout << root << std::endl;
     std::vector<Action> actions;
     actions = root->LegalActions();
     std::vector<Action>::iterator action;
     std::vector<UCTNode *> children;

     // std::cout << actions << std::endl;

     for (action = actions.begin(); action < actions.end(); action++) {
         UCTNode *child = new UCTNode;
         child->action = *action;
         child->visit_count = 0.0;
         child->cum_value = 0.0;
         children.push_back(child);
     }

     return children;
}

float UCT::rollout(std::unique_ptr<open_spiel::State> &root) {
    std::vector<Action> actions, random_action;
    float rv;

    if (root->IsTerminal()) {
        // std::cout << root << std::endl;
        if (root->Returns()[0] == 0.0) {
            return 0.0;
        }
        else {
            return -1.0;
        }
    } else {
        // std::cout << root << std::endl;
        actions = root->LegalActions();
        std::sample(actions.begin(), actions.end(), 
                std::back_inserter(random_action), 1,
                std::mt19937{std::random_device{}()});
        root->ApplyAction(random_action.front());
        rv = rollout(root);
        return -rv;
    }
}

UCTNode * UCT::select_lcb(std::vector<UCTNode *> &children, int N) {
    std::vector<UCTNode *> best_children, best_child;
    std::vector<UCTNode *>::iterator child;
    float child_lcb, max_lcb = -std::numeric_limits<float>::infinity();

    for (child = children.begin(); child < children.end(); child++) {
        child_lcb = (*child)->cum_value / (*child)->visit_count
            - sqrt(2 * log(N) / (*child)->visit_count);

        if (child_lcb > max_lcb) {
            best_children.clear();
            best_children.push_back((*child));
            max_lcb = child_lcb;
        } else if (child_lcb == max_lcb) {
            best_children.push_back((*child));
        }
    }

    std::sample(best_children.begin(), best_children.end(), 
            std::back_inserter(best_child), 1,
            std::mt19937{std::random_device{}()});

    return best_child.front();
}

Action UCT::search(std::unique_ptr<open_spiel::State> &state, int turn_number, bool verbose, std::string output_file) {
    auto start = high_resolution_clock::now();
    Action selected_action;
    UCTNode root_node;
    std::unique_ptr<open_spiel::State> root;
    root_node.visit_count = 0.0;
    root_node.cum_value = 0.0;
        for (auto child = root_node.children.begin(); child < root_node.children.end(); child++)
            printNode((**child));

    for (int i = 0; i < budget; i++) {
        root = state->Clone();
        traverse(root, root_node);
    }

    UCTNode *selection = select_lcb(root_node.children, budget);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    if (verbose) {
        for (auto child = root_node.children.begin(); child < root_node.children.end(); child++)
            printNode((**child));

        printNode(*selection);
    }
    writeNode(root_node, turn_number, duration.count(), output_file);

    selected_action = (*selection).action;

    delete_tree(root_node);

    return selected_action;
}

void UCT::delete_tree(UCTNode &root_node) {
    UCTNode *temp;
    if (root_node.children.empty()) {
        return;
    } else {
        for (auto child = root_node.children.begin(); child < root_node.children.end(); child++) {
            delete_tree(**child);
            temp =  (*child);
            delete temp;
        }
    }
}

}
