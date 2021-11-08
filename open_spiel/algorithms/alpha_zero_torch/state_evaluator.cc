#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/games/connect_four.h"


namespace open_spiel::algorithms::torch_az {

    int EvaluateStates() {
        std::cout << "Testing build." << std::endl;

        std::cout << "Loading game..." << std::endl;
        std::shared_ptr<const open_spiel::Game> game =
                open_spiel::LoadGame("connect_four");
        std::cout << "Connect Four loaded..." << std::endl;

        std::cout << "Loading model..." << std::endl;
        std::string graph_def = "vnet.pb";
        CreateGraphDef(
                *game, 0.0001, 1,
                "/Users/zaheen/projects/open_spiel/open_spiel/algorithms/alpha_zero_torch/",
                graph_def, "resnet", 256, 10);
        VPNetModel model = VPNetModel(*game,
                                      "/Users/zaheen/projects/open_spiel/open_spiel/algorithms/alpha_zero_torch/",
                                      "vnet.pb", "cpu:0");
        model.LoadCheckpoint(
                "/Users/zaheen/projects/open_spiel/open_spiel/algorithms/alpha_zero_torch/checkpoint-1500");
        //model.LoadCheckpoint("/home/zaheen/Documents/az/c4/checkpoint-0");
        std::cout << "Model loaded..." << std::endl;

        std::unique_ptr<open_spiel::State> state = game->NewInitialState();

        std::cout << state->ToString() << std::endl;
        std::cout << state->ObservationTensor() << std::endl;
        std::cout << state->LegalActions() << std::endl;
        VPNetModel::InferenceInputs inputs = {state->LegalActions(), state->ObservationTensor()};
        std::vector<VPNetModel::InferenceOutputs> outputs;
        outputs = model.Inference(std::vector{inputs});
        std::cout << outputs[0].value << " " << outputs[0].policy << std::endl;

        std::string line;
        std::ifstream states_list;
        states_list.open("/Users/zaheen/projects/open_spiel/open_spiel/algorithms/alpha_zero_torch/states_list.txt",
                         std::ios::in);

        if (states_list.is_open()) {
            while (getline(states_list, line)) {
                state = std::unique_ptr<open_spiel::State>(new connect_four::ConnectFourState(game, line));
                std::cout << state->ToString() << std::endl;
                std::cout << state->ObservationTensor() << std::endl;
                std::cout << state->LegalActions() << std::endl;
                VPNetModel::InferenceInputs inputs = {state->LegalActions(), state->ObservationTensor()};
                std::vector<VPNetModel::InferenceOutputs> outputs;
                outputs = model.Inference(std::vector{inputs});
                std::cout << outputs[0].value << "\n";

                for (int i = 0; i < outputs[0].policy.size(); i++)
                    std::cout << outputs[0].policy[i].second << " ";

                std::cout << std::endl;
            }
            states_list.close();
        }
        return 0;
    }
}

int main(int argc, char **argv) {
    open_spiel::algorithms::torch_az::EvaluateStates();
    return 0;
}
