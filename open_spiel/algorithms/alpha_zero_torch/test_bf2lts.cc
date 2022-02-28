//
// Created by Zaheen Ahmad on 2022-01-21.
//

#include "open_spiel/algorithms/alpha_zero_torch/bf2lts.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel.h"
#include <cstring>

namespace open_spiel::algorithms::torch_az {

    int test_bf2lts() {
        std::shared_ptr<const open_spiel::Game> game =
                open_spiel::LoadGame("connect_four");
        std::string path = "/Users/zaheen/projects";
        std::string graph_def = "vpnet.pb";
        std::string model_path = absl::StrCat(path, "/", graph_def);
        SPIEL_CHECK_TRUE(CreateGraphDef(
                *game, 0.0001, 1, path, graph_def, "resnet", 256, 10));
        DeviceManager device_manager;
        device_manager.AddDevice(VPNetModel(*game, path, graph_def, "cpu:0"));
        device_manager.Get(0, 0)->LoadCheckpoint("/Users/zaheen/projects/open_spiel/open_spiel/algorithms/alpha_zero_torch/checkpoint-1500");
        auto eval = std::make_shared<VPNetEvaluator>(
                &device_manager, 1, 1,
                10, 1 / 16);
        BF2LTSBot ltsbot = BF2LTSBot(*game, std::move(eval), 16, 7335, true);
        std::unique_ptr<State> state = game->NewInitialState();
        SearchNode *a;
        SearchNode b, c, d;
//        b.minimax_val = 3.1;
//        b.state = state->Clone();
//        c.minimax_val = 2.0;
//        c.state = state->Clone();
//        d.minimax_val = 2.9;
//        d.state = state->Clone();
//        a->children.push_back(&b);
//        a->children.push_back(&c);
//        a->children.push_back(&d);

//        SearchNode* best = &a->BestChild();
//        std::cout << best->minimax_val << std::endl;

//        std::cout << a->prediction_val << std::endl;
//        auto children = ltsbot.GenerateChildren(*a);
//        std::cout << a->prediction_val << std::endl;
        a = ltsbot.BF2LTSearch(*state);
        ltsbot.TraverseTree(a);
        ltsbot.GarbageCollect(a);

        return 0;
    }

}

int main(int argc, char** argv) {
    open_spiel::algorithms::torch_az::test_bf2lts();
}