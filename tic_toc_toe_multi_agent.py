import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule,
    TensorDictSequential,
    dispatch,
)
from tensordict.utils import NestedKey
from torchrl.collectors import (
    DataCollectorBase,
    SyncDataCollector,
)
from torchrl.data import (
    CompositeSpec,
    BoundedTensorSpec,
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorDictReplayBuffer,
    LazyMemmapStorage,
)
from torchrl.envs import (
    EnvBase,
    check_env_specs,
    ExplorationType,
    set_exploration_type,
    StepCounter,
    Compose,
    TransformedEnv,
)
from torchrl.modules import (
    MultiAgentMLP,
    QValueModule,
    EGreedyModule,
)
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.record import TensorboardLogger
from torchrl.trainers import (
    Trainer,
    ReplayBufferTrainer,
    UpdateWeights,
    Recorder,
    LogReward,
)


@dataclass(frozen=True)
class TicTacToeBoard:
    board: np.ndarray

    def is_winner(self, player=1) -> bool:
        player_view = self.board == 1
        row_win = np.sum((np.sum(player_view, axis=-1) == 3)) > 0
        col_win = np.sum((np.sum(player_view, axis=-2) == 3)) > 0
        main_diag_win = np.trace(player_view) == 3
        anti_diag_win = np.trace(np.fliplr(player_view)) == 3

        won = np.sum((row_win + col_win + main_diag_win + anti_diag_win) > 0).item()
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

    def __init__(
        self, seed: Optional[int] = None, device: str = "cpu", *argv, **kwargs
    ):
        super().__init__(*argv, device=device, **kwargs)
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    @property
    def board_key(self):
        return "agents", "observation"

    def _step(self, state_action: TensorDict):
        board = state_action[self.board_key]
        action = state_action[self.action_key]
        turn = torch.sum(board[0, :] != 0).long().item() % 2
        is_invalid = torch.sum(action[turn, :] * board[turn, :]).long().item()
        done_key = self.done_keys[0]
        assert done_key == "done"

        if is_invalid == 0:
            next_board = torch.zeros_like(board)
            next_board[turn, :] = board[turn, :] + action[turn, :]
            next_board[1 - turn, :] = board[1 - turn, :] + action[turn, :] * 2

            reward = torch.zeros((2,), dtype=torch.float32)
            game = TicTacToeBoard(next_board[turn, :].reshape(3, 3).numpy())
            reward[turn] = int(game.is_winner())
            reward[1 - turn] = 0

            is_done = game.done()

            done = torch.Tensor([is_done])
        else:
            next_board = board
            done = state_action[done_key]
            reward = torch.Tensor([0.0, 0.0])
            reward[turn] = -1.0

        next_state = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": next_board,
                        "reward": reward,
                    },
                    batch_size=torch.Size((2,)),
                ),
                done_key: done.bool(),
            },
            state_action.shape,
        )
        return next_state

    def _reset(self, tensordict: Optional[TensorDict], **kwargs) -> TensorDict:
        return TensorDict(
            {
                "agents": TensorDict(
                    {"observation": torch.zeros((2, 9)).long()},
                    batch_size=torch.Size((2,)),
                ),
                "done": torch.zeros((1,)).bool(),
            },
            batch_size=(),
        )

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _make_spec(self):
        if self.batch_size:
            raise ValueError(f"batch size is not supported, it is {self.batch_size}")

        action_specs = []
        observation_specs = []
        reward_specs = []

        agent_i_action_spec = OneHotDiscreteTensorSpec(
            n=9,
            shape=torch.Size((9,)),
        )
        agent_i_reward_spec = BoundedTensorSpec(
            minimum=-10, maximum=1, dtype=torch.float, shape=torch.Size((1,))
        )
        agent_i_observation_spec = DiscreteTensorSpec(n=3, shape=torch.Size((9,)))

        n_players = 2
        for i in range(n_players):
            action_specs.append(agent_i_action_spec)
            observation_specs.append(agent_i_observation_spec)
            reward_specs.append(agent_i_reward_spec)
        # VmasEnv()
        self.action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": torch.stack(action_specs, dim=0)}, shape=(n_players,)
                )
            }
        )
        self.reward_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"reward": torch.stack(reward_specs, dim=0)}, shape=(n_players,)
                )
            }
        )

        self.observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"observation": torch.stack(observation_specs, dim=0)},
                    shape=(n_players,),
                )
            }
        )
        self.done_spec = DiscreteTensorSpec(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
        )


def test_spec():
    env = TicTacToe()
    check_env_specs(env)


def render(board):
    print(board.numpy())


def interactive_play(actor: Optional[TensorDictModuleBase]):
    env = make_env()

    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        while True:
            state_action = env.reset()
            turn = 0
            if actor is not None:
                computer_player = int(
                    input(f"What is the order of the computer player: ")
                )
            else:
                computer_player = -1

            while torch.sum(state_action["done"]).item() == 0:
                render(state_action[env.board_key][0, :].reshape(3, 3))

                if turn == computer_player:
                    actor(state_action)
                else:
                    action = int(input(f"Next action for player {turn + 1}: "))
                    if action > 8 or action < 0:
                        break

                    action_tensor = torch.zeros((2, 9)).long()
                    action_tensor[turn, action] = 1
                    state_action[env.action_key] = action_tensor
                next_state = env.step(state_action)["next"]
                reward = next_state[("agents", "reward")]
                print("reward:\n" + str(reward.numpy()))
                print()
                state_action = next_state
                next_turn = 1 - turn
                if torch.sum(reward).long().item() != 0:
                    print("Either an invalid move played or one of the players won.")
                    break
                else:
                    turn = next_turn
            answer = input("Do you want to play again (y/t/n): ")
            if answer == "t":
                return True
            if answer == "n":
                return False


@dataclass
class TrainingParams:
    # Training loop
    device = "cpu"
    log_interval = 10

    # model
    eps_init = 0.5
    eps_end = 0.1

    # loss
    gamma = 0.99

    # optimizer
    lr = 1e-4
    wd = 1e-5
    betas = (0.9, 0.999)

    # collector
    total_frames = 1000000
    frame_per_batch = 100
    frame_skip = 1
    buffer_size = 1000

    optim_steps_per_batch = 8


def make_env() -> EnvBase:
    base_env = TicTacToe()
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            # DTypeCastTransform(
            #     dtype_in=torch.long,
            #     dtype_out=torch.float32,
            #     in_keys=[base_env.board_key],
            # ),
            StepCounter(),
        ),
    )
    return env


def make_model(
    env: EnvBase,
    device: str,
    total_frames: int,
    eps_init: float,
    eps_end: float,
) -> Tuple[TensorDictModule, TensorDictModule]:
    num_inputs = env.observation_spec[env.board_key].shape[-1]
    num_actions = env.action_spec.shape[-1]
    num_agents = env.action_spec.shape[-2]

    num_embedding = 10
    embedding = torch.nn.Embedding(3, 10)  # b x n_agent x board x num_embedding
    flatten = torch.nn.Flatten(start_dim=-2)  # b x n_agent x board * num_embedding
    multi_agent_mlp = MultiAgentMLP(
        n_agent_inputs=num_inputs * num_embedding,
        n_agent_outputs=num_actions,
        n_agents=num_agents,
        activation_class=torch.nn.LeakyReLU,
        depth=2,
        num_cells=num_inputs,
        centralised=False,
        share_params=False,
        device=device,
    )

    qvalue_net = torch.nn.Sequential(embedding, flatten, multi_agent_mlp)

    action_value_key = ("agents", "action_value")
    action_value_module = TensorDictModule(
        qvalue_net, in_keys=[env.board_key], out_keys=[action_value_key]
    )

    qvalu_out_keys = [
        ("agents", k) for k in ("action", "action_value", "chosen_action_value")
    ]

    qvalue_module = QValueModule(
        action_value_key=action_value_key,
        out_keys=qvalu_out_keys,
        action_space="one-hot",
    )

    actor = TensorDictSequential(
        action_value_module,
        qvalue_module,
    )

    actor.action_space = env.action_spec

    # noinspection PyTypeChecker
    actor_explore: TensorDictModule = TensorDictSequential(
        actor,
        EGreedyModule(
            spec=env.action_spec,
            annealing_num_steps=total_frames,
            eps_init=eps_init,
            eps_end=eps_end,
            action_key=env.action_key,
        ),
    )
    # Test actors
    actor(env.fake_tensordict())
    actor_explore(env.fake_tensordict())

    return actor, actor_explore


class MultiAgentDQNLoss(DQNLoss):
    @dataclass
    class _AcceptedKeys:
        advantage: NestedKey = ("agents", "advantage")
        value_target: NestedKey = ("agents", "value_target")
        value: NestedKey = ("agents", "chosen_action_value")
        action_value: NestedKey = ("agents", "action_value")
        action: NestedKey = ("agents", "action")
        priority: NestedKey = ("agents", "td_error")
        reward: NestedKey = ("agents", "reward")
        done: NestedKey = ("done",)
        terminated: NestedKey = ("terminated",)

    default_keys = _AcceptedKeys()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        tensordict = tensordict.clone(False)
        tensordict.set(
            ("next",) + self.default_keys.done,
            tensordict.get(("next",) + self.default_keys.done)
            .unsqueeze(-1)
            .expand(tensordict.get_item_shape(("next",) + self.default_keys.reward)),
        )
        tensordict.set(
            ("next",) + self.default_keys.terminated,
            tensordict.get(("next",) + self.default_keys.terminated)
            .unsqueeze(-1)
            .expand(tensordict.get_item_shape(("next",) + self.default_keys.reward)),
        )
        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

        return super().forward(tensordict)


def make_loss_module(
    actor: TensorDictModuleBase, gamma: float
) -> Tuple[TensorDictModuleBase, TargetNetUpdater]:
    loss_module = MultiAgentDQNLoss(actor, delay_value=True)
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    return loss_module, target_updater


def get_collector(
    env: EnvBase,
    actor_explore: TensorDictModuleBase,
    frames_per_batch: int,
    total_frames: int,
) -> DataCollectorBase:
    data_collector = SyncDataCollector(
        env,
        actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
    )
    return data_collector


def get_replay_buffer(buffer_size: int, optim_steps_per_batch: int, batch_size: int):
    replay_buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyMemmapStorage(buffer_size),
        prefetch=optim_steps_per_batch,
    )
    return replay_buffer


def dqn_train(
    training_params: TrainingParams,
    actor: TensorDictModule,
    actor_explore: TensorDictModule,
):
    env = make_env()

    loss_module, target_net_updater = make_loss_module(actor, training_params.gamma)

    collector = get_collector(
        env,
        actor_explore=actor_explore,
        frames_per_batch=training_params.frame_per_batch,
        total_frames=training_params.total_frames,
    )

    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr=training_params.lr,
        weight_decay=training_params.wd,
        betas=training_params.betas,
    )

    exp_name = f"tic_toc_toe_exp_{uuid.uuid1()}"
    # tmpdir = tempfile.TemporaryDirectory()
    # logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir.name)
    logger = TensorboardLogger(exp_name)
    # warnings.warn(f"log dir: {logger.experiment.log_dir}")

    trainer = Trainer(
        collector=collector,
        frame_skip=training_params.frame_skip,
        total_frames=training_params.total_frames,
        loss_module=loss_module,
        optimizer=optimizer,
        optim_steps_per_batch=training_params.optim_steps_per_batch,
        logger=logger,
        log_interval=training_params.log_interval,
    )

    buffer_hook = ReplayBufferTrainer(
        get_replay_buffer(
            training_params.buffer_size,
            training_params.optim_steps_per_batch,
            batch_size=training_params.frame_per_batch,
        ),
        flatten_tensordicts=True,
    )
    buffer_hook.register(trainer)

    weight_updater = UpdateWeights(collector, update_weights_interval=10)
    weight_updater.register(trainer)
    # noinspection PyTypeChecker
    # Change Recorder and add this line:
    # if key == ("next", "reward") or key == ("next", "agents", "reward"):
    recorder = Recorder(
        record_interval=100,  # log every 100 optimization steps
        record_frames=1000,  # maximum number of frames in the record
        frame_skip=1,
        policy_exploration=actor,
        environment=env,
        exploration_type=ExplorationType.MODE,
        log_keys=[("next", "agents", "reward")],
        out_keys={("next", "agents", "reward"): "rewards"},
        log_pbar=True,
    )
    recorder.register(trainer)

    # trainer.register_op("post_optim", target_net_updater.step)

    log_reward = LogReward(log_pbar=True, reward_key=("next", "agents", "reward"))
    log_reward.register(trainer)

    trainer.train()

    return actor


if __name__ == "__main__":
    test_spec()
    # interactive_play(None)
    training_params = TrainingParams()
    actor, actor_explore = make_model(
        env=make_env(),
        device=training_params.device,
        total_frames=training_params.total_frames,
        eps_init=training_params.eps_init,
        eps_end=training_params.eps_end,
    )
    dqn_train(training_params, actor, actor_explore)
    while interactive_play(actor):
        dqn_train(training_params, actor, actor_explore)
