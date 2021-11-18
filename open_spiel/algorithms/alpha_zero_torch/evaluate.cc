#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "open_spiel/algorithms/alpha_zero_torch/uct.h"
#include "open_spiel/algorithms/alpha_zero_torch/puct.h"
#include "open_spiel/algorithms/alpha_zero_torch/lts.h"
#include "open_spiel/algorithms/alpha_zero_torch/bf_lts.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/games/connect_four.h"


namespace open_spiel {
namespace algorithms {
namespace torch_az {

int playGame(std::shared_ptr<const Game> &game) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    std::unique_ptr<open_spiel::State> root;
    UCT mcts = UCT(0.98, 5096);
    UCTNode node;
    std::vector<UCTNode>::iterator child;
    node.visit_count = 16;
    node.cum_value = 7;
    float rv;

    std::cout << mcts.get_c() << " " << mcts.get_budget() << std::endl;
    std::cout << node.visit_count << " " << node.cum_value << std::endl;

    // rv = mcts.rollout(state);
    // std::cout << "Rollout Value: " << rv << std::endl;

    // UCTNode root_node;
    // root_node.visit_count = 0;
    // root_node.cum_value = 0;

    // for (int i = 0; i < 512; i++) {
    //     root = state->Clone();
    //     mcts.traverse(root, root_node);
    // }
    // 
    // for (child = root_node.children.begin(); child < root_node.children.end(); child++) {
    //     printNode(*child);
    // }

//     UCTNode *selection = mcts.select_lcb(root_node.children, 512);
    // printNode(*selection);
//    mcts.search(state, 1, true, "test_file.txt");
//    mcts.search(state, 1, true, "test_file.txt");

    std::string graph_def = "vnet.pb";
    std::string path = std::getenv("C4_MODEL_PATH");// "/Users/zaheen/projects/open_spiel/open_spiel/algorithms/alpha_zero_torch/";
    open_spiel::algorithms::torch_az::CreateGraphDef(*game, 0.0001, 1, path,
            graph_def, "resnet", 256, 10);
    open_spiel::algorithms::torch_az::VPNetModel *model = new open_spiel::algorithms::torch_az::VPNetModel(*game, 
            path, graph_def, std::getenv("C4_DEVICE"));
    model->LoadCheckpoint(path.append("/checkpoint-1500"));
    PUCT pmcts = PUCT(0.98, 32, *model);
//    PUCTNode puct_node;
//    puct_node.visit_count = 0;
//    puct_node.cum_value = 0;
//    std::vector<PUCTNode>::iterator pchild;

    // for (int i = 0; i < 128; i++) {
    //     root = state->Clone();
    //     pmcts.traverse(root, puct_node, *model);
    // }
    // 
    // for (pchild = puct_node.children.begin(); pchild < puct_node.children.end(); pchild++) {
    //     printNode(*pchild);
    // }

//    pmcts.search(state, 0, false, "test_file.txt");
//    pmcts.search(state, 0, false, "test_file.txt");

    state = std::unique_ptr<open_spiel::State>(new connect_four::ConnectFourState(game,
                                                                                  "...x......x......o.....oxox...xooo..oxxxo."));

    std::vector<Action> actions;
    actions = state->LegalActions();
    VPNetModel::InferenceInputs inputs = {actions, state->ObservationTensor()};
    std::vector<VPNetModel::InferenceOutputs> outputs;
    outputs = model->Inference(std::vector{inputs});
    std::cout << outputs[0].value << std::endl;

    state = std::unique_ptr<open_spiel::State>(new connect_four::ConnectFourState(game,
                                                                                  "...x......x......o.o....xox...xooo..oxxxo."));

    actions = state->LegalActions();
    inputs = {actions, state->ObservationTensor()};
    outputs = model->Inference(std::vector{inputs});
    std::cout << outputs[0].value << std::endl;
    LTS lts_search = LTS(32, *model);


    // LTSNode lnode;
    // lnode.depth = 1;
    // lnode.actor_rp = log(1.0);
    // lnode.eventual_rp = log(1.0);
    // lnode.cost = 0.0;

    // root = state->Clone();
    // lnode.children = lts_search.expand(root, lnode, *model);
    // for (auto lchild = lnode.children.begin(); lchild < lnode.children.end(); lchild++) {
    //     printNode(*lchild);
    // }
    state = std::unique_ptr<open_spiel::State>(new connect_four::ConnectFourState(game,
                                                                                  "..........x......o......xox...xooo..oxxxo."));
    lts_search.search(state, 0, true, "test_file.txt");
//    lts_search.search(state, 0, true, "test_file.txt");

    BFLTS bflts = BFLTS(game, 32, *model);
//    bflts.search(state, 0, true, "test_file.txt");
//    bflts.search(state, 0, true, "test_file.txt");

    // root = state->Clone();
    // auto rand_evaluator = std::make_shared<RandomRolloutEvaluator>(1, 1);
    // MCTSBot azmcts(*game, rand_evaluator, 1.0, pow(2, 16), 1000, true, 1, true);
    // auto rnode = azmcts.MCTSearch(*root);
    // for (auto child = rnode->children.begin(); child < rnode->children.end(); child++)
    //     std::cout << (*child).action << " " << ((*child).total_reward / (*child).explore_count) << " " <<  (*child).explore_count << std::endl;

    return 0;
}

}
}
}

int main(int argc, char **argv) {
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("connect_four");
    open_spiel::algorithms::torch_az::playGame(game);
    return 0;
}
