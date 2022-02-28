//
// Created by Zaheen Ahmad on 2022-01-19.
//

#include "open_spiel/algorithms/alpha_zero_torch/bf2lts.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"


namespace open_spiel::algorithms::torch_az {
    int MIN_GC_LIMIT = 5;

    int MemoryUsedMb(int nodes) {
        return nodes * sizeof(SearchNode) / (1 << 20);
    }

    void PrintNode(const SearchNode& node) {
        std::cout << "Action: " << node.action << std::endl
                  //              << "State: " << node.state << std::endl
                  << "Depth: " << node.depth << std::endl
                  << "Minimax Value: " << node.minimax_val << std::endl
                  << "Predicted Value: " << node.prediction_val << std::endl
                  << "Actor RP: " << node.actor_rp << std::endl
                  << "Eventual RP: " << node.eventual_rp << std::endl
                  << "Cost: " << node.cost << "\n" << std::endl;
    }

    bool SearchNode::CompareFinal(const SearchNode& b) const {
        double out = (outcome.empty() ? 0 : outcome[player]);
        double out_b = (b.outcome.empty() ? 0 : b.outcome[b.player]);
        if (out != out_b) {
            return out < out_b;
        }
        return minimax_val < b.minimax_val;
    }

    SearchNode& SearchNode::BestChild() {
        std::vector<SearchNode *> best_children, best_child;
        std::vector<SearchNode *>::iterator child;
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

        return *best_child.front();
    }

    BF2LTSBot::BF2LTSBot(const Game &game, std::shared_ptr<VPNetEvaluator> evaluator, int max_simulations,
                   int seed, bool verbose)
        : max_simulations_{max_simulations},
          //max_nodes_((max_memory_mb << 20) / sizeof(SearchNode) + 1),
          nodes_(0),
          gc_limit_(MIN_GC_LIMIT),
          verbose_(verbose),
          rng_(seed),
          evaluator_(evaluator) {
        GameType game_type = game.GetType();
        if (game_type.reward_model != GameType::RewardModel::kTerminal)
            SpielFatalError("Game must have terminal rewards.");
        if (game_type.dynamics != GameType::Dynamics::kSequential)
            SpielFatalError("Game must have sequential turns.");
    }

    std::vector<SearchNode *> BF2LTSBot::GenerateChildren(SearchNode &parent_node) {
        ActionsAndProbs legal_actions = evaluator_->Prior(*parent_node.state);
        parent_node.prediction_val = evaluator_->Evaluate(*parent_node.state)[parent_node.player];
        int num_actions = legal_actions.size();
        std::vector<SearchNode *> generated_children;

//        std::cout << legal_actions << std::endl;

        for (int i = 0; i < num_actions; i++) {
//            std::cout << legal_actions[i].first << " " << legal_actions[i].second << std::endl;
            float actor_rp = parent_node.eventual_rp + log(legal_actions[i].second);
            float eventual_rp = parent_node.actor_rp + log(1.0 / num_actions);
            int child_depth = parent_node.depth + 1;
            float cost = log(child_depth) - std::max(actor_rp, eventual_rp);
            std::unique_ptr<State> working_state = parent_node.state->Clone();
            working_state->ApplyAction(legal_actions[i].first);
            generated_children.push_back(CreateSearchNode(*working_state, &parent_node, child_depth, actor_rp,
                                                          eventual_rp, legal_actions[i].first, cost));
        }

//        for (auto child = generated_children.begin(); child < generated_children.end(); child++) {
//            PrintNode(**child);
//        }

        return generated_children;
    }

    void BF2LTSBot::BuildSearchTree(SearchNode *root_node) {

        int iterations = 0;
        pq.emplace(root_node);
        while (!pq.empty() && iterations < max_simulations_) {
            // Pop a node from the priority queue to add to tree

            auto node = pq.top();
            pq.pop();

            // Check if node is terminal state. Generate children of node if it is not
            // and place children into queue.

            PrintNode(*node);

            if (node->state->IsTerminal()) {
                node->minimax_val = node->state->Returns()[node->player];
                node->terminal = true;
            } else {
                auto generated_children = GenerateChildren(*node);
                for (auto child = generated_children.begin(); child < generated_children.end(); child++)
                    pq.emplace(*child);
            }

            // Add popped node into tree.
            if (node->parent != NULL)
                node->parent->children.push_back(node);

            iterations++;
        }
    }

    double BF2LTSBot::MinimaxSearch(SearchNode *root_node) {
        if (root_node->terminal)
            return -root_node->minimax_val;

        double child_value, value = -std::numeric_limits<float>::infinity();

        if (root_node->children.empty()) {
            value = root_node->prediction_val;
        } else {
            for (SearchNode *child: root_node->children) {
                child_value = MinimaxSearch(child);
                value = std::max(child_value, value);
            }
        }

        root_node->minimax_val = -value;
        return -value;
    }

    void BF2LTSBot::TraverseTree(SearchNode *root_node) {
        PrintNode(*root_node);

        for (SearchNode *node: root_node->children)
            TraverseTree(node);
    }

    SearchNode * BF2LTSBot::CreateSearchNode(const State& state, SearchNode *parent, int depth, float actor_rp,
                                             float eventual_rp, Action action, float cost) {
        SearchNode *node = new SearchNode;
        node->action = action;
        node->player = state.CurrentPlayer();
        node->parent = parent;
        node->state = state.Clone();
        node->depth = depth;
        node->actor_rp = actor_rp;
        node->eventual_rp = eventual_rp;
        node->prediction_val = NULL;
        node->minimax_val = NULL;
        node->cost = cost;
        node->terminal = false;

        return node;
    }

    void BF2LTSBot::GarbageCollect(SearchNode *node) {
        if (node->children.empty()) {
            return;
        }
        for (SearchNode *child: node->children) {
            GarbageCollect(child);
            delete child;
        }
    }

    SearchNode * BF2LTSBot::BF2LTSearch(const State& state) {
        auto root_node = CreateSearchNode(state);
        BuildSearchTree(root_node);
        MinimaxSearch(root_node);
        return root_node;
    }

    BF2LTSBot::BF2LTSBot() {}

    Action BF2LTSBot::Step(const State& state) {return state.LegalActions()[0];}
}