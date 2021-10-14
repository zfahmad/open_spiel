#include <iostream>
#include <vector>
#include <iterator>
#include <tuple>

#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/games/connect_four.h"


namespace open_spiel {
namespace algorithms {
namespace torch_az {

std::tuple<int, int> negamax(std::unique_ptr<open_spiel::State> &root, int num_visited) {

  std::vector<Action> actions;
  std::vector<Action>::iterator action;
  std::unique_ptr<open_spiel::State> next_state;
  actions = root->LegalActions();
  int value = -99;
  num_visited++;

  if (root->IsTerminal()){
    if (root->Returns()[0] == 0) {
      return {0, num_visited};
    }
    else {
      return {1, num_visited};
    }
  }

  for (action = actions.begin(); action < actions.end(); action++) {
    next_state = root->Clone();
    next_state->ApplyAction(*action);
    auto [child_value, num] = negamax(next_state, num_visited);
    num_visited = num;
    value = std::max(value, child_value);
  }

  return {-value, num_visited};

}

}
}
}

int main(int argc, char **argv) {

  std::shared_ptr<const open_spiel::Game> game =
    open_spiel::LoadGame("tic_tac_toe");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  auto [value, num_visited] = open_spiel::algorithms::torch_az::negamax(state, 0);
  std::cout << "Value: " << -value << " Visited: " << num_visited << std::endl;

}
