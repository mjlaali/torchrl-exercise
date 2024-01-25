import pickle
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable

import tensordict
import torch
from torch.nn.functional import one_hot
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedDiscreteTensorSpec
from torchrl.data.tensor_specs import OneHotDiscreteTensorSpec
from torchrl.envs import EnvBase
from torchrl.objectives.value import TDLambdaEstimator, ValueEstimatorBase


@dataclass
class MCTSNode:
    # prior weights on for every state, action paris p(s, a)
    p_sa: torch.Tensor
    # Q values (i.e. Q(s, a))
    q_sa: torch.Tensor
    # Number of selected actions on each state
    n_sa: torch.Tensor


class PUCTPolicy:
    def __init__(self, cput: float):
        self.cput = cput

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        u = self.puct(td["p_sa"], td["n_sa"])
        action = torch.argmax(td["q_sa"] + u, dim=-1)
        td["action"] = action
        return td

    def puct(self, p_sa: torch.Tensor, n_sa: torch.Tensor) -> torch.Tensor:
        """
        q_sa and p_sa are not clear to me. They seem to need to be positive.
        http://gauss.ececs.uc.edu/Conferences/isaim2010/papers/rosin.pdf
        """
        n = torch.sum(n_sa, dim=-1)
        u_sa = self.cput * p_sa * torch.sqrt(n) / (1 + n_sa)
        return u_sa


@dataclass
class TensorDictTree:
    nnet: tensordict.nn.TensorDictModule
    value_estimator: ValueEstimatorBase
    done_key: str = "done"
    value_key: str = "q_sa"
    cnt_key: str = "n_sa"
    prior_key: str = "p_sa"

    observation_key: str = "observation"
    search_tree: Dict[bytes, MCTSNode] = field(default_factory=dict)

    def _get_node_key(self, state: tensordict.TensorDictBase) -> bytes:
        observation = state[self.observation_key]
        return pickle.dumps(observation.detach().numpy())

    def _get_node(self, state: tensordict.TensorDictBase) -> Optional[MCTSNode]:
        node_key = self._get_node_key(state)

        if node_key in self.search_tree:
            return self.search_tree[node_key]
        return None

    def augment_state(self, state: tensordict.TensorDictBase) -> TensorDictBase:
        node = self._get_node(state)
        state["env_done"] = state[self.done_key]

        if node is None:
            node = self._create_node(state)
            if len(self.search_tree) > 1:       # we cannot set done for the reset_state, it is not supported
                state[self.done_key] = torch.ones_like(state[self.done_key]).to(torch.bool)

        state[self.value_key] = node.q_sa
        state[self.cnt_key] = node.n_sa
        state[self.prior_key] = node.p_sa
        return state

    def _create_node(self, state):
        node_key = self._get_node_key(state)
        assert node_key not in self.search_tree, 'Duplicate node cannot be added to tree'

        nn_output = self.nnet(state)
        q_sa = nn_output["action_values"]

        node = MCTSNode(
            q_sa,
            q_sa,
            torch.zeros_like(q_sa),
        )

        self.search_tree[node_key] = node
        return node

    def update(self, episode: TensorDictBase):
        next_state_value, _ = torch.max(episode[("next", self.value_key)] * episode["action"], dim=-1, keepdim=True)
        episode[("next", self.value_estimator.value_key)] = next_state_value
        value_estimator_input = episode.unsqueeze(dim=0)
        target_value = self.value_estimator.value_estimate(value_estimator_input)
        target_value = target_value.squeeze(dim=0)

        for idx in range(episode.batch_size[0]):
            state = episode[idx, ...]
            node = self._get_node(state)
            action = state["action"]
            node.q_sa = (node.q_sa * (node.n_sa + 1) + target_value[idx, ...] * action) / (node.n_sa + 1 + action)
            node.n_sa += action


class AlphaZeroMCTS(EnvBase):
    def __init__(
            self,
            env: EnvBase,
            tree: TensorDictTree,
            gamma=1.0,
    ):
        super().__init__(device=env.device, batch_size=env.batch_size)

        self.env = env
        self.gamma = gamma
        self.tree = tree

        self._make_spec()

    def _make_spec(self):
        observations = {k: self.env.observation_spec[k] for k in self.env.observation_spec.keys()}

        assert isinstance(self.env.action_spec, OneHotDiscreteTensorSpec)
        self.action_spec = self.env.action_spec
        self.done_spec = self.env.done_spec

        action_shape = self.action_spec.shape
        observations[self.tree.prior_key] = BoundedTensorSpec(low=0.0, high=1.0, shape=action_shape)
        observations[self.tree.value_key] = BoundedTensorSpec(low=0.0, high=1.0, shape=action_shape)
        observations[self.tree.cnt_key] = UnboundedDiscreteTensorSpec(shape=action_shape)
        observations["env_done"] = self.env.done_spec

        self.observation_spec = CompositeSpec(observations)
        self.reward_spec = self.env.reward_spec

    def _step(self, state: tensordict.TensorDictBase) -> tensordict.TensorDict:
        # state contains both observation, mcts node info, and action
        next_state = self.env.step(state)["next"]
        self.tree.augment_state(next_state)
        return next_state

    def _reset(self, tensor: TensorDictBase, **kwargs) -> TensorDictBase:
        reset_state = self.env.reset(tensor)
        self.tree.augment_state(reset_state)
        return reset_state

    def _set_seed(self, seed: Optional[int]):
        pass

