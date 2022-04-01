//
// Created by Zaheen Ahmad on 2022-01-19.
//

#ifndef OPEN_SPIEL_BF2LTS_H
#define OPEN_SPIEL_BF2LTS_H

#include <stdint.h>

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <queue>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"


namespace open_spiel::algorithms::torch_az {

    class Evaluator {
    public:
        virtual ~Evaluator() = default;

        // Return a value of this state for each player.
        virtual std::vector<double> Evaluate(const State& state) = 0;

        // Return a policy: the probability of the current player playing each action.
        virtual ActionsAndProbs Prior(const State& state) = 0;
    };

    struct SearchNode {
        Action action;
        std::unique_ptr<open_spiel::State> state;
        Player player;
        SearchNode *parent;
        int depth;
        float actor_rp;
        float eventual_rp;
        float cost;
        double prediction_val;
        double minimax_val;
        std::vector<double> outcome;
        bool terminal;
        std::vector<SearchNode *> children;

        SearchNode() {};
        bool CompareFinal(const SearchNode &b) const;
        SearchNode& BestChild();
    };

    struct MyComparator {
        bool operator() (SearchNode *arg_1, SearchNode *arg_2) {
            return arg_1->cost > arg_2->cost;
        }
    };

    class BF2LTSBot : public Bot {
    public:
        BF2LTSBot(const Game& game, std::shared_ptr<VPNetEvaluator> evaluator,
                int max_simulations,
                //int64_t max_memory_mb,  // Max memory use in megabytes.
                int seed, bool verbose);
        BF2LTSBot();
        ~BF2LTSBot() = default;
        SearchNode * BF2LTSearch(const State& state);
        SearchNode * CreateSearchNode(const State& state, SearchNode *parent=NULL, int depth=1,
                                      float actor_rp=0.0, float eventual_rp=0.0, Action action=kInvalidAction,
                                      float cost=0.0);
        std::vector<SearchNode *> GenerateChildren(SearchNode &parent_node);
        void BuildSearchTree(SearchNode *root_node);
        double MinimaxSearch(SearchNode *root_node);
        void TraverseTree(SearchNode *root_node);
        Action Step(const State& state);
        void GarbageCollect(SearchNode *node);

    private:
        int max_simulations_;
        int max_nodes_;
        std::shared_ptr<VPNetEvaluator> evaluator_;
        std::mt19937 rng_;
        bool verbose_;
        int nodes_;
        int gc_limit_;

    };

    bool operator< (const SearchNode &node_1, const SearchNode &node_2);

    bool operator> (const SearchNode &node_1, const SearchNode &node_2);

}

#endif //OPEN_SPIEL_BF2LTS_H
