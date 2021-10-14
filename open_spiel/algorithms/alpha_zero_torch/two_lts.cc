#include <iostream>
#include <vector>
#include <iterator>
#include <tuple>
#include <limits>

#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/games/connect_four.h"


namespace open_spiel {
namespace algorithms {
namespace torch_az {

    std::tuple<int, double, double, int> two_lts(std::unique_ptr<open_spiel::State> &root, 
            int depth, double bound, double cprp, double nprp, double nlb, int num_visited, 
            VPNetModel *model) {

        std::vector<Action> actions;
        std::vector<Action>::iterator action;
        std::unique_ptr<open_spiel::State> next_state;
        actions = root->LegalActions();
        VPNetModel::InferenceInputs inputs = {root->LegalActions(), root->ObservationTensor()};
        std::vector<VPNetModel::InferenceOutputs> outputs;
        outputs = model->Inference(std::vector{inputs});
        float num_actions = outputs[0].policy.size();
        double value = -std::numeric_limits<double>::infinity();
        num_visited++;
        double denom, cost, np, cp;
        bool has_children = false;
        int best_action;

        // std::cout << root->CurrentPlayer() << std::endl;
        // std::cout << root << std::endl;
        // std::cout << outputs[0].value << std::endl;
        //
        if (bound == 1) {
            std::cout << "Initial State Policy: " << outputs[0].policy << std::endl;
        }

        if (root->IsTerminal()) {
            //std::cout << "Terminal: " << root->Returns() << std::endl;
            if (root->Returns()[0] == 0.0) {
                return {9, 0.0, nlb, num_visited};
            }
            else {
                return {9, 1.0, nlb, num_visited};
            }
        }
        else {
            // std::cout << outputs[0].policy << std::endl;
            for (int i = 0; i < num_actions; i++) {
                cp = log(outputs[0].policy[i].second) + cprp;
                np = log(1 / num_actions) + nprp;
                // std::cout << log(cp) << " " << log(np) << " " << 1 / num_actions << std::endl;
                denom = std::max(cp, np);
                cost = log(depth + 1) - denom;
                // std::cout << cost << std::endl;
                if (cost > bound)
                    nlb = std::min(cost, nlb);
                // std::cout << cost << " " << bound << std::endl;
                if (cost <= bound) {
                    has_children = true;
                    next_state = root->Clone();
                    next_state->ApplyAction(outputs[0].policy[i].first);
                    auto [baction, child_value, new_nlb, num] = two_lts(next_state, depth + 1, bound, 
                            np, cp, nlb, num_visited, model);
                    nlb = new_nlb;
                    num_visited = num;
                    // std::cout << value << " " << child_value << std::endl;
                    if (child_value > value) {
                        value = child_value;
                        best_action = outputs[0].policy[i].first;
                    }
                }
            }
        }

        if (has_children == false) {
            value = outputs[0].value;
            if (root->CurrentPlayer() == 1) {
                value = -value;
            }
            // value = -99;
        }

        return {best_action, -value, nlb, num_visited};
    }

}
}
}


int main(int argc, char **argv) {
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("connect_four");
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();

    std::string graph_def = "vnet.pb";
    std::string path = "/Users/zaheen/projects/open_spiel/open_spiel/algorithms/alpha_zero_torch/";
    open_spiel::algorithms::torch_az::CreateGraphDef(*game, 0.0001, 1, path,
            graph_def, "resnet", 256, 10);
    open_spiel::algorithms::torch_az::VPNetModel *model = new open_spiel::algorithms::torch_az::VPNetModel(*game, 
            path, graph_def, "cpu:0");
    model->LoadCheckpoint(path.append("checkpoint-1500"));
    std::cout << "Model Loaded\n";
    std::tuple<int, double, double, int> stats = std::make_tuple(9, 0, 1, 0);
    double bound = 0;

    for (int i = 1; i < 16; i += 1) {
        stats = open_spiel::algorithms::torch_az::two_lts(state, 1, bound, 0, 0, std::numeric_limits<double>::max(), 
                0, model);
        std::cout << "Action: " << std::get<0>(stats) << " Value: " << -std::get<1>(stats) << " Visited: " 
            << std::get<3>(stats) << " Bound: " << std::get<2>(stats) << std::endl;
        bound = std::get<2>(stats);
    }
}
