#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>

#include "open_spiel/algorithms/alpha_zero_torch/uct.h"
#include "open_spiel/algorithms/alpha_zero_torch/puct.h"
#include "open_spiel/algorithms/alpha_zero_torch/lts.h"
#include "open_spiel/algorithms/alpha_zero_torch/bf_lts.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"

namespace open_spiel::algorithms::torch_az {

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

void evaluatePUCT(std::shared_ptr<const Game> &game, int budget, int model_num, int num_games, VPNetModel &model) {
    std::unique_ptr<open_spiel::State> state;
    PUCT p1 = PUCT(0.99, budget, model);
    UCT p2 = UCT(0.98, 1024);
    int p1_rate[3] = {0, 0, 0};
    int p2_rate[3] = {0, 0, 0};

    std::string output_file = "results/summary/eval_puct_" + std::to_string(budget) + "_" + std::to_string(model_num) + ".txt";
    std::ofstream os_stream;
    os_stream.open(output_file);

    for (int eval_num = 0; eval_num < num_games; eval_num++) {
        std::string game_file = "results/eval_puct_p1_" + std::to_string(budget) + "_" + std::to_string(model_num) + "_" + std::to_string(eval_num) + "_game.txt";
        std::ofstream game_stream;
        game_stream.open(game_file);
        state = game->NewInitialState();
        Action action;
        std::cout << state << std::endl;
        game_stream << state << std::endl;
        int turn_number = 0;
        bool finish = false;


        while (!finish) {
            if (turn_number % 2)
                action = p2.search(state, turn_number, false,
                                   "results/eval_puct_p1_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            else
                action = p1.search(state, turn_number, false,
                                   "results/eval_puct_p1_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            state->ApplyAction(action);
            std::cout << state << std::endl;
            game_stream << state << std::endl;
            turn_number++;
            if (state->IsTerminal()) {
                finish = true;
                if (state->Returns()[0] == 0.0) {
                    p1_rate[2] += 1;
                } else if (state->Returns()[0] == 1.0) {
                    p1_rate[0] += 1;
                } else {
                    p1_rate[1] += 1;
                }
            }
        }
        game_stream.close();
    }
    os_stream << std::to_string(p1_rate[0]) + " " + std::to_string(p1_rate[1]) + " " + std::to_string(p1_rate[2]) + "\n" << std::endl;

    for (int eval_num = 0; eval_num < num_games; eval_num++) {
        std::string game_file = "results/eval_puct_p2_" + std::to_string(budget) + "_" + std::to_string(model_num) + "_" + std::to_string(eval_num) + "_game.txt";
        std::ofstream game_stream;
        game_stream.open(game_file);
        state = game->NewInitialState();
        Action action;
        std::cout << state << std::endl;
        game_stream << state << std::endl;
        int turn_number = 0;
        bool finish = false;


        while (!finish) {
            if (turn_number % 2)
                action = p1.search(state, turn_number, false,
                                   "results/eval_puct_p2_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            else
                action = p2.search(state, turn_number, false,
                                   "results/eval_puct_p2_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            state->ApplyAction(action);
            std::cout << state << std::endl;
            game_stream << state << std::endl;
            turn_number++;
            if (state->IsTerminal()) {
                finish = true;
                if (state->Returns()[0] == 0.0) {
                    p2_rate[2] += 1;
                } else if (state->Returns()[0] == 1.0) {
                    p2_rate[1] += 1;
                } else {
                    p2_rate[0] += 1;
                }
            }
        }
        game_stream.close();
    }
    os_stream << std::to_string(p2_rate[0]) + " " + std::to_string(p2_rate[1]) + " " + std::to_string(p2_rate[2]) + "\n" << std::endl;
    os_stream.close();
}

void evaluateBFLTS(std::shared_ptr<const Game> &game, int budget, int model_num, int num_games, VPNetModel &model) {
    std::unique_ptr<open_spiel::State> state;
    BFLTS p1 = BFLTS(game, budget, model);
    UCT p2 = UCT(0.98, 65536);
    int p1_rate[3] = {0, 0, 0};
    int p2_rate[3] = {0, 0, 0};

    std::string output_file = "results/summary/eval_bflts_" + std::to_string(budget) + "_" + std::to_string(model_num) + ".txt";
    std::ofstream os_stream;
    os_stream.open(output_file);

    for (int eval_num = 0; eval_num < num_games; eval_num++) {
        std::string game_file = "results/eval_bflts_p1_" + std::to_string(budget) + "_" + std::to_string(model_num) + "_" + std::to_string(eval_num) + "_game.txt";
        std::ofstream game_stream;
        game_stream.open(game_file);
        state = game->NewInitialState();
        Action action;
        std::cout << state << std::endl;
        game_stream << state << std::endl;
        int turn_number = 0;
        bool finish = false;


        while (!finish) {
            if (turn_number % 2)
                action = p2.search(state, turn_number, false,
                                   "results/eval_bflts_p1_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            else
                action = p1.search(state, turn_number, false,
                                   "results/eval_bflts_p1_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            state->ApplyAction(action);
            std::cout << state << std::endl;
            game_stream << state << std::endl;
            turn_number++;
            if (state->IsTerminal()) {
                finish = true;
                if (state->Returns()[0] == 0.0) {
                    p1_rate[2] += 1;
                } else if (state->Returns()[0] == 1.0) {
                    p1_rate[0] += 1;
                } else {
                    p1_rate[1] += 1;
                }
            }
        }
        game_stream.close();
    }
    os_stream << std::to_string(p1_rate[0]) + " " + std::to_string(p1_rate[1]) + " " + std::to_string(p1_rate[2]) + "\n" << std::endl;

    for (int eval_num = 0; eval_num < num_games; eval_num++) {
        std::string game_file = "results/eval_bflts_p2_" + std::to_string(budget) + "_" + std::to_string(model_num) + "_" + std::to_string(eval_num) + "_game.txt";
        std::ofstream game_stream;
        game_stream.open(game_file);
        state = game->NewInitialState();
        Action action;
        std::cout << state << std::endl;
        game_stream << state << std::endl;
        int turn_number = 0;
        bool finish = false;


        while (!finish) {
            if (turn_number % 2)
                action = p1.search(state, turn_number, false,
                                   "results/eval_bflts_p2_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            else
                action = p2.search(state, turn_number, false,
                                   "results/eval_bflts_p2_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            state->ApplyAction(action);
            std::cout << state << std::endl;
            game_stream << state << std::endl;
            turn_number++;
            if (state->IsTerminal()) {
                finish = true;
                if (state->Returns()[0] == 0.0) {
                    p2_rate[2] += 1;
                } else if (state->Returns()[0] == 1.0) {
                    p2_rate[1] += 1;
                } else {
                    p2_rate[0] += 1;
                }
            }
        }
        game_stream.close();
    }
    os_stream << std::to_string(p2_rate[0]) + " " + std::to_string(p2_rate[1]) + " " + std::to_string(p2_rate[2]) + "\n" << std::endl;
    os_stream.close();
}

void evaluateLTS(std::shared_ptr<const Game> &game, int budget, int model_num, int num_games, VPNetModel &model) {
    std::unique_ptr<open_spiel::State> state;
    LTS p1 = LTS(budget, model);
    UCT p2 = UCT(0.98, 65536);
    int p1_rate[3] = {0, 0, 0};
    int p2_rate[3] = {0, 0, 0};

    std::string output_file = "results/summary/eval_bflts_" + std::to_string(budget) + "_" + std::to_string(model_num) + ".txt";
    std::ofstream os_stream;
    os_stream.open(output_file);

    for (int eval_num = 0; eval_num < num_games; eval_num++) {
        std::string game_file = "results/eval_lts_p1_" + std::to_string(budget) + "_" + std::to_string(model_num) + "_" + std::to_string(eval_num) + "_game.txt";
        std::ofstream game_stream;
        game_stream.open(game_file);
        state = game->NewInitialState();
        Action action;
        std::cout << state << std::endl;
        game_stream << state << std::endl;
        int turn_number = 0;
        bool finish = false;


        while (!finish) {
            if (turn_number % 2)
                action = p2.search(state, turn_number, false,
                                   "results/eval_lts_p1_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            else
                action = p1.search(state, turn_number, false,
                                   "results/eval_lts_p1_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            state->ApplyAction(action);
            std::cout << state << std::endl;
            game_stream << state << std::endl;
            turn_number++;
            if (state->IsTerminal()) {
                finish = true;
                if (state->Returns()[0] == 0.0) {
                    p1_rate[2] += 1;
                } else if (state->Returns()[0] == 1.0) {
                    p1_rate[0] += 1;
                } else {
                    p1_rate[1] += 1;
                }
            }
        }
        game_stream.close();
    }
    os_stream << std::to_string(p1_rate[0]) + " " + std::to_string(p1_rate[1]) + " " + std::to_string(p1_rate[2]) + "\n" << std::endl;

    for (int eval_num = 0; eval_num < num_games; eval_num++) {
        std::string game_file = "results/eval_lts_p2_" + std::to_string(budget) + "_" + std::to_string(model_num) + "_" + std::to_string(eval_num) + "_game.txt";
        std::ofstream game_stream;
        game_stream.open(game_file);
        state = game->NewInitialState();
        Action action;
        std::cout << state << std::endl;
        game_stream << state << std::endl;
        int turn_number = 0;
        bool finish = false;


        while (!finish) {
            if (turn_number % 2)
                action = p1.search(state, turn_number, false,
                                   "results/eval_lts_p2_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            else
                action = p2.search(state, turn_number, false,
                                   "results/eval_lts_p2_" + std::to_string(budget) + "_" + std::to_string(model_num) +
                                   "_" +
                                   std::to_string(eval_num) + "_stats.txt");
            state->ApplyAction(action);
            std::cout << state << std::endl;
            game_stream << state << std::endl;
            turn_number++;
            if (state->IsTerminal()) {
                finish = true;
                if (state->Returns()[0] == 0.0) {
                    p2_rate[2] += 1;
                } else if (state->Returns()[0] == 1.0) {
                    p2_rate[1] += 1;
                } else {
                    p2_rate[0] += 1;
                }
            }
        }
        game_stream.close();
    }
    os_stream << std::to_string(p2_rate[0]) + " " + std::to_string(p2_rate[1]) + " " + std::to_string(p2_rate[2]) + "\n" << std::endl;
    os_stream.close();
}

}

int main(int argc, char **argv) {
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("connect_four");
    std::string graph_def = "vnet.pb";
    std::string path = std::getenv("C4_MODEL_PATH"); // "/home/zaheen/projects/os/open_spiel/algorithms/alpha_zero_torch/";
    open_spiel::algorithms::torch_az::CreateGraphDef(*game, 0.0001, 1, path,
            graph_def, "resnet", 256, 10);
    open_spiel::algorithms::torch_az::VPNetModel *model = 
        new open_spiel::algorithms::torch_az::VPNetModel(*game, path, graph_def, std::getenv("C4_DEVICE"));
    model->LoadCheckpoint(path.append("/checkpoint-1500"));
    // open_spiel::algorithms::torch_az::UCTvUCT(game, 2048, "test_game.txt");
    for (int b = 16; b < 33; b *= 2) {
        open_spiel::algorithms::torch_az::evaluatePUCT(game, b, 1500, 5, *model);
//        open_spiel::algorithms::torch_az::evaluateLTS(game, b, 1500, 5, *model);
//        open_spiel::algorithms::torch_az::evaluateBFLTS(game, b, 1500, 5, *model);
    }
}
