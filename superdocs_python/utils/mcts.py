# based on https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from dataclasses import dataclass

@dataclass
class ExecutionNode:
    changes: dict
    feedback: str
    reward: float
    children: list

    # attributes get added when the ExecutionNode is added to the MCTS object
    descendants: int = -1
    parent: int = -1
    id: int = -1

class MCTS:
    def __init__(self, initial_node):
        self.root = 0
        initial_node.id = 0
        initial_node.descendants = 0
        self.nodes = {0: initial_node}

    def find_best_node(self, curr_node=None):
        if not(curr_node):
            curr_node = self.nodes[self.root]
        best_score = self.nodes[curr_node.id].reward
        best_node = curr_node
        for child_id in self.nodes[curr_node.id].children:
            child_node, child_score = self.find_best_node(curr_node=self.nodes[child_id])
            if child_score > best_score:
                best_score = child_score
                best_node = child_node
        return best_node, best_score

    def update_reward(self, node_id, new_reward):
        if node_id == -1:
            return
        self.nodes[node_id].descendants += 1
        self.nodes[node_id].reward = (self.nodes[node_id].reward * (self.nodes[node_id].descendants) + new_reward) / (self.nodes[node_id].descendants + 1)
        self.update_reward(self.nodes[node_id].parent, new_reward)

    def add_child(self, parent_id, child):
        child.id = len(self.nodes)
        child.parent = parent_id
        child.descendants = 0 # the actual number of nodes to be considered at a given position is the number of descendants +1
        child.children = []        
    
        self.nodes[child.id] = child
        self.nodes[parent_id].children.append(child.id)
        self.update_reward(parent_id, child.reward)