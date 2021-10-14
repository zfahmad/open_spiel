#include <iostream>
#include <vector>
#include <iterator>
#include <tuple>

#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/games/connect_four.h"


namespace open_spiel {
namespace algorithms {
namespace torch_az {

std::tuple<int, int> id_ab_negamax(std::unique_ptr<open_spiel::State> &root, 
        int depth, int alpha, int beta, int num_visited, VPNetModel *model) {

  std::vector<Action> actions;
  std::vector<Action>::iterator action;
  std::unique_ptr<open_spiel::State> next_state;
  actions = root->LegalActions();
  int value = -99;
  num_visited++;

  if (root->IsTerminal()) {
    if (root->Returns()[0] == 0.0) {
      return {0, num_visited};
    }
    else {
      return {1, num_visited};
    }
  }
  if (depth == 0) {
      VPNetModel::InferenceInputs inputs = {root->LegalActions(), root->ObservationTensor()};
      std::vector<VPNetModel::InferenceOutputs> outputs;
      outputs = model->Inference(std::vector{inputs});
      std::cout << "Leaf value: " << outputs[0].value << "\n";
      return {-outputs[0].value, num_visited};
  }
  if (depth != 0) {
    for (action = actions.begin(); action < actions.end(); action++) {
      next_state = root->Clone();
      next_state->ApplyAction(*action);
      auto [child_value, num] = id_ab_negamax(next_state, depth - 1, -beta, -alpha, num_visited, model);
      num_visited = num;
      value = std::max(value, child_value);
      alpha = std::max(alpha, value);

      if (alpha >= beta)
          break;
    }
  }

  return {-value, num_visited};
}

}
}
}

int main(int argc, char **argv) {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("connect_four");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  
  std::string graph_def = "vnet.pb";
  std::string path = "/home/zaheen/projects/os/open_spiel/algorithms/alpha_zero_torch/";
  open_spiel::algorithms::torch_az::CreateGraphDef(*game, 0.0001, 1, path,
                  graph_def, "resnet", 256, 10);
  open_spiel::algorithms::torch_az::VPNetModel *model = new open_spiel::algorithms::torch_az::VPNetModel(*game, 
          path, graph_def, "cuda:0");
  model->LoadCheckpoint(path.append("checkpoint-1500"));
  std::cout << "Model Loaded\n";

  auto [value, num_visited] = open_spiel::algorithms::torch_az::id_ab_negamax(state, 4, -99, 99, 0, model);
  std::cout << "Value: " << -value << " Visited: " << num_visited << std::endl;
}
