#include <iostream>
#include <vector>
#include "open_spiel/algorithms/alpha_zero_torch/bf_lts.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

void printNode(LTSNode &node) {
    std::cout << "Action: " << node.action << std::endl
              << "Depth: " << node.depth << std::endl
              << "Minimax Value: " << node.minimax_val << std::endl
              << "Actor RP: " << node.actor_rp << std::endl
              << "Eventual RP: " << node.eventual_rp << std::endl
              << "Cost: " << node.cost << "\n" << std::endl;
}

void BT_LTS::enqueue(BFSNode new_item) {
    if (size == 0) {
        QNode *new_node;
        new_node->item = &new_item;
        new_node->prev = NULL;
        new_node->next = NULL;
        head = new_node;
    }
}

void printQueue() {
    if (size == 0) {
        std::cout << "Queue is empty!\n" << std::endl;
    } else {
        BFSNode *current = head;
        do {
           printNode(current->item);
           current = current->next;
        } while (current != NULL)
    }
}

}
}
}
