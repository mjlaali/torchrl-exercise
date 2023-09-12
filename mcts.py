from torchrl.envs import EnvBase

from typing import Dict, Optional
from dataclasses import dataclass, field

import torch.nn.functional as F

import torch
import pickle
import tensordict


def puct(p_sa: torch.Tensor, n_sa: torch.Tensor, cput: float) -> torch.Tensor:
    """
    q_sa and p_sa are not clear to me. They seems to need to be positive.
    http://gauss.ececs.uc.edu/Conferences/isaim2010/papers/rosin.pdf
    """
    n = torch.sum(n_sa, dim=-1)
    u_sa = cput * p_sa * torch.sqrt(n) / (1 + n_sa)
    return u_sa


@dataclass
class MCTSNode:
    # prior weights on for every state, action paris p(s, a)
    p_sa: torch.Tensor
    # Q values (i.e. Q(s, a))
    q_sa: torch.Tensor
    # Number of selected actions on each state
    n_sa: torch.Tensor


@dataclass
class TensordictTree:
    done_key: str
    observation_key: str = "observation"
    search_tree: Dict[bytes, MCTSNode] = field(default_factory=dict)

    def is_terminal(self, state: tensordict.TensorDict) -> bool:
        if self.done_key in state.keys():
            return state.get(self.done_key)
        return False

    def get_node_key(self, state: tensordict.TensorDict) -> bytes:
        observation = state[self.observation_key]
        return pickle.dumps(observation.detach().numpy())

    def get_node(self, state: tensordict.TensorDict) -> Optional[MCTSNode]:
        node_key = self.get_node_key(state)

        if node_key in self.search_tree:
            return self.search_tree[node_key]
        return None

    def add_node(self, state: tensordict.TensorDict, node: MCTSNode):
        node_key = self.get_node_key(state)
        assert node_key not in self.search_tree, 'Duplicate node cannot be added to tree'
        self.search_tree[node_key] = node


class AlphaZeroMCTS:
    def __init__(
            self,
            env: EnvBase,
            nnet: tensordict.nn.TensorDictModule,
            cput=1.0,
            max_depth=100,
    ):
        self.env = env
        self.nnet = nnet
        self.cput = cput
        self.max_depth = max_depth

    def search(self, tree: TensordictTree, state: tensordict.TensorDict, depth=0) -> torch.Tensor:
        if tree.is_terminal(state):
            return state[self.env.reward_key]

        if tree.get_node(state) is None:
            node, reward = self.explore(state)
            tree.add_node(node)
            return reward

        node = tree.get_node(state)
        state[self.env.action_key] = self.select_action(node)

        next_state = self.env.step(state)["next"]
        if depth != self.max_depth:
            reward = self.search(tree, next_state.clone(), depth + 1)
        else:
            reward, _ = torch.max(tree.get_node(next_state).q_sa, dim=-1)

        self.update_node(node, state, reward)
        return reward

    def explore(self, state):
        self.nnet(state)
        q_sa = state["action_values"]

        estimated_rewards, _ = torch.max(q_sa, dim=-1)

        node = MCTSNode(
            q_sa,
            q_sa,
            torch.zeros_like(q_sa),
        )

        return node, estimated_rewards

    def select_action(self, node: MCTSNode):
        u = puct(node.p_sa, node.n_sa, self.cput)
        action = torch.argmax(node.q_sa + u, dim=-1)
        return action

    def update_node(self, node: MCTSNode, state: tensordict.TensorDict, reward: torch.Tensor):
        action = state["action"]
        one_hot_action = F.one_hot(action, num_classes=node.q_sa.shape[-1])
        epsilon = torch.Tensor([1e-9])
        node.q_sa = node.q_sa + (reward * one_hot_action - node.q_sa) / (node.n_sa + epsilon)
        node.n_sa = node.n_sa + one_hot_action
