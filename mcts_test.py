from unittest.mock import MagicMock

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import check_env_specs
from torchrl.envs.libs.gym import GymEnv
from torchrl.objectives.value import TDLambdaEstimate, TDLambdaEstimator

from mcts import TensorDictTree, AlphaZeroMCTS, PUCTPolicy


def test_env_spec():
    env = GymEnv("CliffWalking-v0")

    nnet = MagicMock(return_value={"action_values": torch.Tensor([1 / 4 for _ in range(4)])})
    tree = TensorDictTree(nnet, MagicMock())
    under_test = AlphaZeroMCTS(
        env,
        tree,
    )

    check_env_specs(under_test)


def test_expand():
    env = GymEnv("CliffWalking-v0")
    nnet = MagicMock(return_value={"action_values": torch.Tensor([1 / 4 for _ in range(4)])})
    tree = TensorDictTree(nnet, MagicMock())
    under_test = AlphaZeroMCTS(
        env,
        tree,
    )

    def dummy_policy(td: TensorDictBase):
        td["action"] = 3
        return td

    for i in range(2):
        rollout = under_test.rollout(10, policy=dummy_policy)
        assert rollout.batch_size[-1] == i + 1
        for idx, state in enumerate(rollout):
            assert state[("next", "reward")].item() == -1
            assert state[("next", "done")].item() if idx == i else not state[("next", "done")].item()
            assert not state[("next", "env_done")].item()
            assert "q_sa" in state.keys()
            assert "n_sa" in state.keys()
            assert "p_sa" in state.keys()


def test_puct():
    puct = PUCTPolicy(1.0)
    res = puct(
        TensorDict(
            {
                "q_sa": torch.Tensor([0.4, 0.4, 0.4]),
                "p_sa": torch.Tensor([0.4, 0.4, 0.4]),
                "n_sa": torch.Tensor([5, 0, 10]),
            },
            batch_size=()
        )
    )

    # Explore when the p_sa is the same
    assert res["action"].item() == 1


def test_update_tree():
    action = torch.Tensor([0, 1, 0]).to(torch.int64)
    num_actions = action.shape[-1]

    state = TensorDict(
        {
            "observation": torch.Tensor([1]),
            "done": torch.Tensor([0]).to(torch.bool),
            "action": action
        },
        batch_size=()
    )

    nnet = MagicMock(return_value={"action_values": torch.Tensor([1/3 for _ in range(num_actions)])})

    tree = TensorDictTree(
        nnet, TDLambdaEstimator(gamma=1.0, lmbda=1.0, value_network=None),
    )

    out_state = tree.augment_state(state)

    assert "n_sa" in out_state.keys()
    assert (out_state["n_sa"] == torch.Tensor([0 for _ in range(num_actions)])).all()
    assert "q_sa" in out_state.keys()
    assert "p_sa" in out_state.keys()

    reward = torch.Tensor([1])
    next_state = TensorDict(
        {
            "observation": torch.Tensor([2]),
            "done": torch.Tensor([0]).to(torch.bool),
            "reward": reward
        },
        batch_size=()
    )

    next_state = tree.augment_state(next_state)
    state["next"] = next_state

    path = torch.stack([state], len(state.batch_size)).contiguous()

    tree.update(path)

    updated_state = tree.augment_state(state)
    assert (updated_state["n_sa"] == action).all()
    assert (updated_state["p_sa"] == state["p_sa"]).all()
    assert (updated_state["q_sa"] == (state["p_sa"] + state["action"] * reward) / (state["n_sa"] + 1)).all()

# TODO: now we see that tree get updated, can we see we can solve a RL problem with MCTS?
