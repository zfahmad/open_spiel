#include <iostream>
#include <vector>
#include <chrono>
#include "open_spiel/algorithms/alpha_zero_torch/bf_lts.h"

using namespace std::chrono;
namespace open_spiel::algorithms::torch_az {

void printNode(const BFSNode &node) {
    std::cout << "Action: " << node.action << std::endl
              << "Depth: " << node.depth << std::endl
              << "Minimax Value: " << node.minimax_val << std::endl
              << "Predicted Value: " << node.pred_val << std::endl
              << "Actor RP: " << node.actor_rp << std::endl
              << "Eventual RP: " << node.eventual_rp << std::endl
              << "Cost: " << node.cost << "\n" << std::endl;
}

void writeNode(const BFSNode &node, int turn_number, float duration, std::string file_name="") {
    std::ofstream os_stream;
    os_stream.open(file_name, std::ios::app);

    os_stream << "BFS at turn: " << turn_number << "\n" << std::endl;
    os_stream << "Player: " << turn_number % 2 << std::endl;
    os_stream << "Elapsed time: " << duration << "\n" << std::endl;
    for (auto child = node.children.begin(); child < node.children.end(); child++) {
        os_stream << "Action: " << (*child)->action << std::endl
                  << "Depth: " << (*child)->depth << std::endl
                  << "Minimax Value: " << (*child)->minimax_val << std::endl
                  << "Predicted Value: " << (*child)->pred_val << std::endl
                  << "Actor RP: " << (*child)->actor_rp << std::endl
                  << "Eventual RP: " << (*child)->eventual_rp << std::endl
                  << "Cost: " << (*child)->cost << std::endl
                  << "Terminal: " << (*child)->terminal << "\n" << std::endl;
    }
    os_stream << "=====================================\n\n" << std::endl;
    os_stream.close();
}

bool operator< (const BFSNode &node_1, const BFSNode &node_2) {
    return node_1.cost < node_2.cost;
}

bool operator> (const BFSNode &node_1, const BFSNode &node_2) {
    return node_1.cost > node_2.cost;
}

void BFLTS::generate_children(std::unique_ptr<open_spiel::State> &root, BFSNode &root_node) {
    std::vector<Action> actions;
    actions = root->LegalActions();
    std::vector<Action>::iterator action;
    VPNetModel::InferenceInputs inputs = {actions, root->ObservationTensor()};
    std::vector<VPNetModel::InferenceOutputs> outputs;
    outputs = model.Inference(std::vector{inputs});
    float num_actions = actions.size();
    std::string state_str = root->ToString();
    std::unique_ptr<open_spiel::State> new_state;

    for (int i = 0; i < num_actions; i++) {
        BFSNode *child = new BFSNode;
        child->action = outputs[0].policy[i].first;
        new_state = root->Clone();
        new_state->ApplyAction(child->action);
        child->state_str = new_state->Serialize();
        child->parent = &root_node;
        child->actor_rp = root_node.eventual_rp + log(outputs[0].policy[i].second);
        child->eventual_rp = root_node.actor_rp + log(1.0 / num_actions);
        child->depth = root_node.depth + 1;
        child->cost = log(child->depth) - std::max(child->actor_rp, child->eventual_rp);
        child->minimax_val = NULL;
        child->pred_val = NULL;
        child->terminal = false;
        pq.push(*child);
    }

    root_node.pred_val = outputs[0].value;
    if (root->CurrentPlayer() == 1) {
        root_node.pred_val = -root_node.pred_val;
    }
}

BFSNode * BFLTS::select_best(std::vector<BFSNode *> &children) {
    std::vector<BFSNode *> best_children, best_child;
    std::vector<BFSNode *>::iterator child;
    float max_val = -std::numeric_limits<float>::infinity();

    for (child = children.begin(); child < children.end(); child++) {
        if ((*child)->minimax_val > max_val) {
            best_children.clear();
            best_children.push_back((*child));
            max_val = (*child)->minimax_val;
        } else if ((*child)->minimax_val == max_val) {
            best_children.push_back((*child));
        }
    }

    std::sample(best_children.begin(), best_children.end(),
                std::back_inserter(best_child), 1,
                std::mt19937{std::random_device{}()});

    return best_child.front();
}

Action BFLTS::search(std::unique_ptr<open_spiel::State> &state, int turn_number, bool verbose, std::string output_file) {
    auto start = high_resolution_clock::now();
    float value;
    BFSNode *tree;
    tree = build_tree(state);
    value = minimax(*tree);
    std::cout << "Value returned: " << -value << std::endl;
    if (verbose) {
        std::cout << "Finished search over with minimax value: " << -value << std::endl;
        for (auto child = tree->children.begin(); child < tree->children.end(); child++) {
            printNode((**child));
        }
    }
    auto stop = high_resolution_clock::now();
    BFSNode *selection = select_best(tree->children);
    auto selected_action = selection->action;
    auto duration = duration_cast<milliseconds>(stop - start);
    writeNode(*tree, turn_number, duration.count(), output_file);
    delete_tree(tree);
    return selected_action;
}

BFSNode * BFLTS::build_tree(std::unique_ptr<open_spiel::State> &state) {
    std::unique_ptr<open_spiel::State> root;
    root = state->Clone();
    int i = 0;
    BFSNode root_node;
    BFSNode *tree, *node;
    root_node.action = NULL;
    root_node.state_str = root->Serialize();
    root_node.parent = nullptr;
    root_node.depth = 1;
    root_node.actor_rp = 0.0;
    root_node.eventual_rp = 0.0;
    root_node.cost = 0.0;
    root_node.terminal = false;
    pq.push(root_node);

//    std::cout << "In tree: " << std::endl;
//    for (int i = 0; i < budget; i++) {
    while (!pq.empty() && i < budget) {
        node = new BFSNode;
        *node = pq.top();
        pq.pop();
//        printNode(*node);
//        std::cout << root << std::endl;
        root = game->DeserializeState(node->state_str);
//        std::cout << root << std::endl;
        if (!root->IsTerminal()) {
            generate_children(root, *node);
        } else {
            float value;
            if (root->Returns()[0] == 0.0) {
                value = 0.0;
            }
            else {
                value = -1.0;
            }
            node->minimax_val = -value;
        }
        if (node->parent != nullptr)
            node->parent->children.push_back(node);
        else
            tree = node;
        i++;
    }
    return tree;
}

float BFLTS::minimax(BFSNode &root_node) {
    float child_val, value = -std::numeric_limits<float>::infinity();
    bool has_children = false;
    // printNode(root_node);

    if (root_node.terminal) {
        value = root_node.minimax_val;
//        std::cout << value << std::endl;
        return value;
    }

    if (root_node.children.empty()) {
        value = root_node.pred_val;
    } else {
        for (auto child = root_node.children.begin(); child < root_node.children.end(); child++) {
            child_val = minimax(**child);
            value = std::max(child_val, value);
        }
    }

    root_node.minimax_val = -value;
//    std::cout << value << std::endl;
    return -value;
}

void BFLTS::delete_tree(BFSNode *root_node) {
    BFSNode *temp;
    if (root_node->children.empty()) {
        return;
    } else {
        for (auto child = root_node->children.begin(); child < root_node->children.end(); child++) {
            delete_tree(*child);
            temp =  (*child);
            delete temp;
        }
    }
}

}
