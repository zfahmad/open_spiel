#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#include "open_spiel/algorithms/alpha_zero_torch/uct.h"
#include "open_spiel/algorithms/alpha_zero_torch/puct.h"
#include "open_spiel/algorithms/alpha_zero_torch/lts.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

void UCTvUCT(std::shared_ptr<const Game> &game, int budget, std::string output_file) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    Action action;
    int turn_number = 0;
    UCT player = UCT(0.99, budget);
    bool finish = false;

    std::cout << state << std::endl;
    while(!finish) {
        action = player.search(state, turn_number, false, output_file);
        state->ApplyAction(action); 
        std::cout << state << std::endl;
        turn_number++;
        if (state->IsTerminal())
            finish = true;
    }
}

void PUCTvLTS(std::shared_ptr<const Game> &game, int budget, VPNetModel &model) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    Action action;
    int turn_number = 0;
    PUCT p1 = PUCT(0.99, budget, model);
    LTS p2 = LTS(budget, model);
    bool finish = false;

    std::string output_file = "puct_lts_" + std::to_string(budget) + ".txt";
    std::ofstream os_stream;
    os_stream.open(output_file);

    std::cout << state << std::endl;
    os_stream << state << std::endl;
    while(!finish) {
        if (turn_number % 2)
            action = p2.search(state, turn_number, false, 
                    "puct_lts_" + std::to_string(budget) + "_stats.txt");
        else
            action = p1.search(state, turn_number, false, 
                    "puct_lts_" + std::to_string(budget) + "_stats.txt");
        state->ApplyAction(action); 
        std::cout << state << std::endl;
        os_stream << state << std::endl;
        turn_number++;
        if (state->IsTerminal())
            finish = true;
    }
    os_stream.close();
}

void LTSvPUCT(std::shared_ptr<const Game> &game, int budget, VPNetModel &model) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    Action action;
    int turn_number = 0;
    LTS p1 = LTS(budget, model);
    PUCT p2 = PUCT(0.99, budget, model);
    bool finish = false;

    std::string output_file = "lts_puct_" + std::to_string(budget) + ".txt";
    std::ofstream os_stream;
    os_stream.open(output_file);

    std::cout << state << std::endl;
    os_stream << state << std::endl;
    while(!finish) {
        if (turn_number % 2)
            action = p2.search(state, turn_number, false, 
                    "lts_puct_" + std::to_string(budget) + "_stats.txt");
        else
            action = p1.search(state, turn_number, false, 
                    "lts_puct_" + std::to_string(budget) + "_stats.txt");
        state->ApplyAction(action); 
        std::cout << state << std::endl;
        os_stream << state << std::endl;
        turn_number++;
        if (state->IsTerminal())
            finish = true;
    }
    os_stream.close();
}

}
}
}

int main(int argc, char **argv) {
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("connect_four");
    std::string graph_def = "vnet.pb";
    std::string path = "/Users/zaheen/projects/open_spiel/open_spiel/algorithms/alpha_zero_torch/";
    open_spiel::algorithms::torch_az::CreateGraphDef(*game, 0.0001, 1, path,
            graph_def, "resnet", 256, 10);
    open_spiel::algorithms::torch_az::VPNetModel *model = 
        new open_spiel::algorithms::torch_az::VPNetModel(*game, path, graph_def, "cpu:0");
    model->LoadCheckpoint(path.append("checkpoint-1500"));
    // open_spiel::algorithms::torch_az::UCTvUCT(game, 2048, "test_game.txt");
    for (int b = 2; b < 2049; b *= 2) {
        open_spiel::algorithms::torch_az::PUCTvLTS(game, b, *model);
        open_spiel::algorithms::torch_az::LTSvPUCT(game, b, *model);
    }
}
