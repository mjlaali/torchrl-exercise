from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import CompositeSpec, BoundedTensorSpec, DiscreteTensorSpec, BinaryDiscreteTensorSpec, \
    OneHotDiscreteTensorSpec
from torchrl.envs import EnvBase, check_env_specs


@dataclass(frozen=True)
class TicTacToeBoard:
    board: np.ndarray

    def is_winner(self, player: int) -> bool:
        player_view = self.board == player
        row_win = np.sum((np.sum(player_view, axis=-1) == 3)) > 0
        col_win = np.sum((np.sum(player_view, axis=-2) == 3)) > 0
        main_diag_win = np.trace(player_view) == 3
        anti_diag_win = np.trace(np.fliplr(player_view)) == 3

        won = np.sum(
            (row_win + col_win + main_diag_win + anti_diag_win) > 0
        ).item()
        return won

    def done(self) -> bool:
        empty_cell = np.sum(self.board == 0) > 0
        return not empty_cell or self.is_winner(1) or self.is_winner(2)


class TicTacToe(EnvBase):
    """
    Environment asks each agent in each turn for a move (even the agent that is not its turn).
    Environment accepts the move from active player and ignore the other player.
    Regardless of which player is active, environment provided a reward for both agents.
    """

    def __init__(self, seed: Optional[int] = None, device: str = 'cpu', *argv, **kwargs):
        super().__init__(*argv, device=device, **kwargs)
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _step(self, state_action: TensorDict):
        turn = state_action["turn"]
        boards = state_action["board"]
        action = state_action[self.action_key]
        is_invalid = torch.sum((turn[:, 1:2] == 1) * action * boards).item()
        if is_invalid == 0:
            next_board = boards + ((turn[:, 1:2] == 1) * action) + ((turn[:, 1:2] == 1) * action * 2)[[1, 0]]
            next_turn = ~turn
            tic_toc_toe_board = TicTacToeBoard(next_board[0, :].reshape(3, 3).numpy())
            reward = torch.Tensor(
                [
                    tic_toc_toe_board.is_winner(i + 1) for i in range(2)
                ]
            ).float() * 1
            reward -= reward[[1, 0]]

            done = torch.Tensor(
                [
                    tic_toc_toe_board.done() for i in range(2)
                ]
            ).long()
        else:
            next_board = boards
            next_turn = turn
            done = state_action["done"]
            reward = turn[:, 1:2] * -1.0

        next_state = TensorDict({
            "board": next_board,
            "reward": reward,
            "turn": next_turn,
            "done": done,
        },
            state_action.shape,
        )
        return next_state

    def _reset(self, tensordict: Optional[TensorDict], **kwargs) -> TensorDict:
        return TensorDict(
            {
                "board": torch.zeros((2, 9)).long(),
                "turn": torch.Tensor([[0, 1], [1, 0]]).bool(),
                "done": torch.zeros((2, 1)).long(),
            },
            batch_size=(),
        )

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _make_spec(self):
        if self.batch_size:
            raise ValueError(f"batch size is not supported, it is {self.batch_size}")
        batch_size = self.batch_size
        self.observation_spec = CompositeSpec(
            {
                "board": DiscreteTensorSpec(n=3, shape=(2, 9)),
                "turn": OneHotDiscreteTensorSpec(n=2, shape=(2, 2))
            },
            shape=batch_size
        )

        self.state_spec = self.observation_spec.clone()
        self.action_spec = OneHotDiscreteTensorSpec(n=9, shape=(2, 9))
        self.reward_spec = BoundedTensorSpec(
            minimum=-10,
            maximum=1,
            dtype=torch.float,
            shape=(2, 1)
        )
        self.done_spec = BinaryDiscreteTensorSpec(n=1, shape=(2, 1))


def test_spec():
    env = TicTacToe()
    check_env_specs(env)


def render(board):
    print(board.numpy())


def human_vs_human():
    env = TicTacToe()
    state_action = env.reset()
    turn = 0
    with torch.no_grad():
        while torch.sum(state_action["done"]).item() == 0:
            render(state_action["board"][0, :].reshape(3, 3))

            action = int(input(f"Next action for player {turn + 1}: "))
            if action > 8 or action < 0:
                break

            action_tensor = torch.zeros((2, 9)).long()
            action_tensor[turn, action] = 1
            state_action["action"] = action_tensor
            next_state = env.step(state_action)["next"]

            print("reward:\n" + str(next_state["reward"].numpy()))
            print()
            state_action = next_state
            turn = state_action["turn"][0, 0].long().item()


if __name__ == '__main__':
    test_spec()
    human_vs_human()
