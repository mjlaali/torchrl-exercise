from collections import defaultdict

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorSpec, ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import TransformedEnv, Compose, StepCounter
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss
from torchrl.objectives.value import TD0Estimator
import matplotlib.pyplot as plt

from tqdm import tqdm


def main():
    base_env = GymEnv("CliffWalking-v0")
    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),
        ),
    )
    frames_per_batch = 1000
    total_frames = frames_per_batch * 1000
    sub_batch_size = frames_per_batch // 100
    device = "cpu"
    num_value_improvement = 10

    qvalue_actor, stock_actor, exploration_module = build_actor(
        env.observation_spec["observation"].shape[-1],
        env.action_spec.shape[-1],
        env.action_spec,
    )
    collector = SyncDataCollector(
        env,
        stock_actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        max_frames_per_traj=60,
        device=device,
    )

    value_network = TensorDictSequential(
        qvalue_actor,
        TensorDictModule(module=IdentityModule(), in_keys=["chosen_action_value"], out_keys=["state_value"]),
    )
    target_value_cal = TD0Estimator(
        gamma=0.9,
        value_network=value_network,
    )

    loss_module = DQNLoss(qvalue_actor)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    lr = 1e-3

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    for i, tensordict_data in enumerate(collector):

        for _ in range(num_value_improvement):
            with torch.no_grad():
                target_value_cal(tensordict_data)

            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            for _ in range(frames_per_batch // sub_batch_size):
                batch = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(batch.to(device))
                loss_value = loss_vals["loss"]
                loss_value.backward()

                optim.step()
                optim.zero_grad()

        scheduler.step()
        exploration_module.step(tensordict_data.numel())

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.2f} (init={logs['reward'][0]: 4.0f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 0.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our env horizon).
            # The ``rollout`` method of the env can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(100, qvalue_actor)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.0f} at {i}"
                    f"(init: {logs['eval reward (sum)'][0]: 4.0f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()
    res = env.rollout(max_steps=100, policy=qvalue_actor)
    print(res["action"])

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.savefig('training.png')

    print(list(qvalue_actor.parameters(True))[0])


def build_actor(num_states, num_actions, spec: TensorSpec):
    to_action = torch.nn.Linear(num_states, num_actions, bias=False)
    to_float32 = CastTo(torch.float32)

    qvalue = torch.nn.Sequential(to_float32, to_action)
    qvalue_actor = QValueActor(qvalue, spec=spec)
    exploration_module = EGreedyModule(eps_init=0.2, spec=spec)
    stock_actor = TensorDictSequential(
        qvalue_actor,
        exploration_module,
    )

    return qvalue_actor, stock_actor, exploration_module


class CastTo(torch.nn.Module):
    def __init__(self, dtype):
        super(CastTo, self).__init__()
        self.dtype = dtype

    def forward(self, input):
        return input.to(self.dtype)


class ValueMax(torch.nn.Module):
    def forward(self, input):
        val, _ = torch.max(input, dim=-1, keepdim=True)
        return val


class IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x


if __name__ == '__main__':
    main()
