# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch.nn
import torch.optim
from torchrl.data import Composite
from torchrl.envs import RewardSum, StepCounter, TransformedEnv
from sim_env import SimEnv
from torchrl.modules import MLP, QValueActor
from torchrl.record import VideoRecorder


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(config, seed=None, device="cpu"):
    env = SimEnv(config, seed=seed, device=device)
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_dqn_modules(proof_environment, device):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs
    num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    # Define Q-Value Module
    mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.ReLU,
        out_features=num_outputs,
        num_cells=[120, 84],
        device=device,
    )

    qvalue_module = QValueActor(
        module=mlp,
        spec=Composite(action=action_spec).to(device),
        in_keys=["observation"],
    )
    return qvalue_module


def make_dqn_model(config, seed, device):
    proof_environment = make_env(config, seed=seed, device=device)
    qvalue_module = make_dqn_modules(proof_environment, device=device)
    del proof_environment
    return qvalue_module


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean()
