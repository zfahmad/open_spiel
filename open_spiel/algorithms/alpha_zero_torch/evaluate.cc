#include <iostream>
#include <vector>
#include <cmath>

#include "open_spiel/algorithms/alpha_zero_torch/uct.h"
#include "open_spiel/algorithms/alpha_zero_torch/puct.h"
#include "open_spiel/algorithms/alpha_zero_torch/lts.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

int playGame(std::shared_ptr<const Game> &game) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    std::unique_ptr<open_spiel::State> root;
    UCT mcts = UCT(0.98, 8192);
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

    // UCTNode *selection = mcts.select_lcb(root_node.children, 512);
    // printNode(*selection);
    mcts.search(state, 1, false, "test_file.txt");

    std::string graph_def = "vnet.pb";
    std::string path = "/Users/zaheen/projects/open_spiel/open_spiel/algorithms/alpha_zero_torch/";
    open_spiel::algorithms::torch_az::CreateGraphDef(*game, 0.0001, 1, path,
            graph_def, "resnet", 256, 10);
    open_spiel::algorithms::torch_az::VPNetModel *model = new open_spiel::algorithms::torch_az::VPNetModel(*game, 
            path, graph_def, "cpu:0");
    model->LoadCheckpoint(path.append("checkpoint-1500"));
    PUCT pmcts = PUCT(0.98, 128, *model);
    PUCTNode puct_node;
    puct_node.visit_count = 0;
    puct_node.cum_value = 0;
    std::vector<PUCTNode>::iterator pchild;

    // for (int i = 0; i < 128; i++) {
    //     root = state->Clone();
    //     pmcts.traverse(root, puct_node, *model);
    // }
    // 
    // for (pchild = puct_node.children.begin(); pchild < puct_node.children.end(); pchild++) {
    //     printNode(*pchild);
    // }

    pmcts.search(state, 0, false, "test_file.txt");
    
    LTS lts_search = LTS(128, *model);

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
    lts_search.search(state, 0, false, "test_file.txt");

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
