#ifndef BF_LTS_H
#define BF_LTS_H

#include <iostream>
#include <vector>
#include <string>

namespace open_spiel {
namespace algorithms {
namespace torch_az {

struct BFSNode {
    Action action;
    std::string state_str;
    BFSNode *parent;
    int depth;
    float actor_rp;
    float eventual_rp;
    float cost;
    float pred_val;
    float minimax_val;
    std::vector<BFSNode> children;
};

class BFSQueue {
    struct QNode {
        BFSNode *item;
        BFSNode *next;
        BFSNode *prev;
    }
    private:
        int size;
        int capacity;
        QNode *head;
    public:
        BFSQueue(capacity)
            : size(0), capacity(capacity) {}
        void enqueue(BFSNode new_node);
        BFSNode * pull();
        void pop();
        void printQueue();
}

}
}
}

#endif
