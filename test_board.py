import pytest
import numpy as np

from tic_toc_toe_multi_agent import TicTacToeBoard


@pytest.mark.parametrize(
    "board, expected_winner, expected_done",
    [
        (
                np.asarray([
                    [1, 1, 1],
                    [0, 2, 0],
                    [2, 0, 2],
                ]),
                1,
                True,
        ),
        (
                np.asarray([
                    [2, 1, 1],
                    [0, 2, 1],
                    [0, 0, 2],
                ]),
                2,
                True,
        ),
        (
                np.asarray([
                    [1, 1, 0],
                    [0, 2, 1],
                    [2, 0, 2],
                ]),
                0,
                False,
        ),
        (
                np.asarray([
                    [1, 2, 0],
                    [1, 2, 0],
                    [1, 0, 2],
                ]),
                1,
                True,
        ),

        (
                np.asarray([
                    [1, 2, 1],
                    [2, 2, 1],
                    [2, 1, 2],
                ]),
                0,
                True,
        ),

    ]
)
def test_board(board: np.ndarray, expected_winner: int, expected_done: bool):
    tic_toc_toe = TicTacToeBoard(board)
    if expected_winner:
        assert tic_toc_toe.is_winner(expected_winner)
    else:
        for player in (1, 2):
            assert not tic_toc_toe.is_winner(player)

    assert tic_toc_toe.done() == expected_done
