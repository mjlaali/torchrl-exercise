import pytest

from mcts import puct, TensordictTree, MCTSNode
from tensordict import TensorDict
import torch


def test_puct():
    res = puct(
        p_sa=torch.Tensor([0.4, 0.2, 0.4]),
        n_sa=torch.Tensor([5, 0, 10]),
        cput=1,
    )

    # Explore when the p_sa is the same
    assert res[0] > res[2], f'res = {res}'


def test_add_node_to_tensordict_tree():
    observation_key = "observation"
    tree = TensordictTree(done_key="done", observation_key=observation_key)
    node = MCTSNode(torch.Tensor([0.1, 0.9]), torch.Tensor([0.1, 0.9]), torch.Tensor([0, 0]))
    state = TensorDict({observation_key: torch.Tensor([1, 2, 3])}, batch_size=())
    tree.add_node(state, node)
    retrieved = tree.get_node(state)
    assert retrieved == node


def test_adding_duplicate_state_raise_exception():
    observation_key = "observation"
    tree = TensordictTree(done_key="done", observation_key=observation_key)
    node = MCTSNode(torch.Tensor([0.1, 0.9]), torch.Tensor([0.1, 0.9]), torch.Tensor([0, 0]))
    state = TensorDict({observation_key: torch.Tensor([1, 2, 3])}, batch_size=())
    tree.add_node(state, node)
    with pytest.raises(AssertionError):
        tree.add_node(state, node)


def test_detecting_terminal_state():
    observation_key = "observation"
    done_key = "done"
    tree = TensordictTree(done_key=done_key, observation_key=observation_key)
    for done in (True, False):
        state = TensorDict({observation_key: torch.Tensor([1, 2, 3]), done_key: torch.Tensor([done])}, batch_size=())
        assert tree.is_terminal(state) == done

