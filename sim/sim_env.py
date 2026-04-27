import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

from collections import defaultdict

import numpy as np
import torch
import tqdm
from io_api import (
    _build_graph_from_config,
    _build_node_from_config,
    load_config_from_files,
)
from scheduler import Simulator
from tensordict import NonTensorStack, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import multiprocessing, nn
from torchrl.data import (
    Binary,
    Bounded,
    BoundedContinuous,
    Categorical,
    Composite,
    MultiCategorical,
    MultiOneHot,
    OneHot,
    Unbounded,
    UnboundedContinuous,
    UnboundedDiscrete,
)
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, make_composite_from_td, step_mdp

if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("fork")


class SimEnv(EnvBase):
    batch_locked = True

    def __init__(self, config: Dict[str, Any], seed=None, device="cpu"):
        self.config = config

        self.devices_per_op: list = []
        for op_cfg in config["graph"]["ops"]:
            device_dict = op_cfg["runtime_on_device"]
            device_list = list(device_dict.keys())
            self.devices_per_op.append(device_list)

        td_params = self.gen_params()
        super().__init__(device=device)
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)

    def _step(self, tensordict):
        # get dummy observation
        observation = tensordict["observation"]
        # get params from tensordict
        devices_per_op = tensordict["params", "devices_per_op"]
        # get action from tensordict
        # action = tensordict["action"]  # MultiCategorical
        action = tensordict["action"]
        action = self.action_spec.to_categorical(action)  # MultiOneHot
        # Use action to update config with device assignments for each op
        selected_device_idx = action
        action_op_idx = 0
        selected_device_ids = []
        for idx, device_list in enumerate(devices_per_op):
            if len(device_list) >= 2:
                device_id = device_list[selected_device_idx[action_op_idx]]
                selected_device_ids.append(device_id)
                action_op_idx += 1
            elif len(device_list) == 1:
                selected_device_ids.append(
                    device_list[0]
                )  # only one device available, select it
            else:
                selected_device_ids.append(None)  # no device available, set to None
        assert len(selected_device_ids) == len(
            devices_per_op
        )  # Ensure we have a selected device for each op

        for i, device_id in enumerate(selected_device_ids):
            if device_id is not None:
                self.config["graph"]["ops"][i]["device_id"] = device_id

        # Run the simulator with the updated config and get the total runtime
        node = _build_node_from_config(self.config)
        graph = _build_graph_from_config(self.config)

        simulator = Simulator(node=node, graph=graph)
        # TODO: Fix simulator to run without error and return total runtime, then replace the placeholder below
        # result = simulator.run()
        # total_runtime = result.total_runtime
        total_runtime = (
            1.0  # Placeholder while testing the rest of the environment logic
        )
        # Compute reward as negative total runtime (since we want to minimize runtime)
        reward = torch.tensor(-total_runtime)
        done = torch.ones_like(reward, dtype=torch.bool)
        out = TensorDict(
            {
                "observation": observation,  # dummy observation
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)

        observation = torch.zeros(size=(tensordict.shape + (1,)))

        out = TensorDict(
            {
                "observation": observation,  # dummy observation
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out

    def _make_spec(self, td_params):
        self.observation_spec = Composite(
            observation=UnboundedContinuous(
                shape=(*td_params.shape, 1)
            ),  # dummy observation, needed by the RL loop
            params=make_composite_from_td(td_params["params"]),
        )

        self.state_spec = self.observation_spec.clone()

        action_len_list = []
        for device_list in td_params["params", "devices_per_op"]:
            device_len = len(device_list)
            if device_len >= 2:
                action_len_list.append(device_len)
        self.action_spec = MultiOneHot(nvec=tuple(action_len_list))

        self.reward_spec = UnboundedContinuous(shape=(*td_params.shape, 1))

    def _set_seed(self, seed: int | None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        self.rng = rng

    def gen_params(self, batch_size=None) -> TensorDictBase:
        """Returns a ``tensordict`` containing the simulation parameters such as number of devices and number of ops."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "devices_per_op": NonTensorStack(*self.devices_per_op),
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# Training
NUM_CELLS       = 256       # hidden layer width
LR              = 3e-4      # Adam learning rate
MAX_GRAD_NORM   = 1.0       # gradient clipping

# Data collection
FRAMES_PER_BATCH = 64       # steps collected before each update
TOTAL_FRAMES     = 50_000   # total environment interactions

# PPO
CLIP_EPSILON   = 0.2        # PPO clipping parameter ε
GAMMA          = 0.99       # discount factor
LAMBDA         = 0.95       # GAE λ
ENTROPY_COEF   = 1e-4       # entropy bonus coefficient
CRITIC_COEF    = 0.5        # value-function loss coefficient
NUM_EPOCHS     = 10         # gradient steps per collected batch
MINI_BATCH_SZ  = 32         # mini-batch size for PPO updates

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────

config_dict = load_config_from_files(
    "hardware_config.json", "../graph_bert_optimized.json"
)

seed = 0
torch.manual_seed(seed)
device = "cpu"

env = SimEnv(config=config_dict, seed=seed, device=device)

env_specs    = env.specs
action_spec  = env_specs["input_spec", "full_action_spec", "action"]

# Total one-hot action dimension (sum of all nvec components)
num_outputs  = int(sum(action_spec.nvec))
# Observation dimension
input_shape  = int(sum(env.observation_spec["observation"].shape))  # = 1 here

print(f"Observation dim : {input_shape}")
print(f"Action (one-hot) dim : {num_outputs}")
print(f"Action nvec : {action_spec.nvec}")


# ─────────────────────────────────────────────────────────────────────────────
# Actor network
#
# Outputs *logits* for every segment of the MultiOneHot action space.
# ClipPPOLoss expects the distribution to be created from a ProbabilisticActor,
# but here we wire things manually so that the loss can compute log-probs.
# ─────────────────────────────────────────────────────────────────────────────

class MultiOneHotActor(nn.Module):
    """
    Maps a flat observation to logits for each sub-action in the MultiOneHot
    action space, then samples a MultiOneHot action and records log-probs.

    Attributes
    ----------
    nvec : tuple[int]
        Number of categories for each sub-action (same as action_spec.nvec).
    """

    def __init__(self, input_dim: int, hidden_dim: int, nvec: tuple):
        super().__init__()
        self.nvec = nvec

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # One linear head per sub-action
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, n) for n in nvec]
        )

    def forward(self, observation: torch.Tensor):
        """
        Parameters
        ----------
        observation : Tensor[..., input_dim]

        Returns
        -------
        action      : Tensor[..., sum(nvec)]  – one-hot encoded
        log_prob    : Tensor[...]             – sum of log-probs across sub-actions
        """
        h = self.shared(observation)

        one_hot_parts = []
        log_prob = torch.zeros(observation.shape[:-1], device=observation.device)

        for head, n in zip(self.heads, self.nvec):
            logits = head(h)                              # [..., n]
            dist   = torch.distributions.Categorical(logits=logits)
            idx    = dist.sample()                        # [...]
            lp     = dist.log_prob(idx)                  # [...]

            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(-1, idx.unsqueeze(-1), 1.0)

            one_hot_parts.append(one_hot)
            log_prob = log_prob + lp

        action = torch.cat(one_hot_parts, dim=-1)
        return action, log_prob


class MultiOneHotActorEval(nn.Module):
    """
    Same network but re-evaluates log-prob for a given action (used in the
    PPO loss).  Also returns entropy for the entropy bonus.
    """

    def __init__(self, base: MultiOneHotActor):
        super().__init__()
        self.base = base

    @property
    def nvec(self):
        return self.base.nvec

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        """
        Parameters
        ----------
        observation : Tensor[..., input_dim]
        action      : Tensor[..., sum(nvec)]  – stored one-hot action

        Returns
        -------
        log_prob : Tensor[...]
        entropy  : Tensor[...]
        """
        h = self.base.shared(observation)

        log_prob = torch.zeros(observation.shape[:-1], device=observation.device)
        entropy  = torch.zeros_like(log_prob)

        offset = 0
        for head, n in zip(self.base.heads, self.base.nvec):
            logits  = head(h)                             # [..., n]
            dist    = torch.distributions.Categorical(logits=logits)

            # Recover the categorical index from the stored one-hot slice
            one_hot_slice = action[..., offset:offset + n]
            idx = one_hot_slice.argmax(dim=-1)            # [...]

            log_prob = log_prob + dist.log_prob(idx)
            entropy  = entropy  + dist.entropy()
            offset  += n

        return log_prob, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Critic network
# ─────────────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)  # [..., 1]


# ─────────────────────────────────────────────────────────────────────────────
# Instantiate networks
# ─────────────────────────────────────────────────────────────────────────────

actor_net  = MultiOneHotActor(
    input_dim=input_shape, hidden_dim=NUM_CELLS, nvec=tuple(action_spec.nvec)
).to(device)

actor_eval = MultiOneHotActorEval(base=actor_net).to(device)
critic_net = Critic(input_dim=input_shape, hidden_dim=NUM_CELLS).to(device)

optimizer = torch.optim.Adam(
    list(actor_net.parameters()) + list(critic_net.parameters()),
    lr=LR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: collect one batch of trajectories
# ─────────────────────────────────────────────────────────────────────────────

def collect_batch(env: SimEnv, n_steps: int) -> dict:
    """
    Roll out the current actor for ``n_steps`` steps.

    Returns a dict with tensors of shape [n_steps, ...]:
        observations, actions, log_probs_old, rewards, dones, values
    """
    observations = []
    actions      = []
    log_probs    = []
    rewards      = []
    dones        = []
    values       = []

    td = env.reset()

    for _ in range(n_steps):
        obs = td["observation"].float().to(device)          # [*batch, 1]

        with torch.no_grad():
            action, lp = actor_net(obs)                     # sample
            value       = critic_net(obs)                   # [*batch, 1]

        # Store tensors – squeeze batch dim if batch_size=[]
        observations.append(obs.clone())
        actions.append(action.clone())
        log_probs.append(lp.clone())
        values.append(value.clone())

        # Write action into tensordict and step
        td["action"] = action
        td = env.step(td)

        reward = td["next", "reward"].float().to(device)
        done   = td["next", "done"].to(device)

        rewards.append(reward.clone())
        dones.append(done.clone())

        # Move to next state
        td = step_mdp(td)

    # Stack along a new leading dimension: [n_steps, *batch, ...]
    batch = {
        "observations": torch.stack(observations, dim=0),
        "actions"     : torch.stack(actions,      dim=0),
        "log_probs"   : torch.stack(log_probs,    dim=0),
        "rewards"     : torch.stack(rewards,       dim=0),
        "dones"       : torch.stack(dones,         dim=0),
        "values"      : torch.stack(values,        dim=0),
    }
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute GAE returns & advantages
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards: torch.Tensor,   # [T, *batch, 1]
    values:  torch.Tensor,   # [T, *batch, 1]
    dones:   torch.Tensor,   # [T, *batch, 1]
    gamma:   float = GAMMA,
    lam:     float = LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    advantages : Tensor[T, *batch, 1]
    returns    : Tensor[T, *batch, 1]
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae   = torch.zeros_like(rewards[0])

    # Bootstrap value at T is 0 because all episodes end (done=True always here)
    next_value = torch.zeros_like(values[0])

    for t in reversed(range(T)):
        mask       = (~dones[t]).float()
        delta      = rewards[t] + gamma * next_value * mask - values[t]
        last_gae   = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# PPO update step
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(batch: dict, n_epochs: int, mini_batch_size: int) -> dict:
    """
    Performs ``n_epochs`` passes over the collected batch with PPO.

    Returns a dict of scalar losses for logging.
    """
    obs_all      = batch["observations"].reshape(-1, input_shape)   # [N, obs_dim]
    act_all      = batch["actions"].reshape(-1, num_outputs)        # [N, act_dim]
    lp_old_all   = batch["log_probs"].reshape(-1)                   # [N]
    adv_all      = batch["advantages"].reshape(-1)                  # [N]
    ret_all      = batch["returns"].reshape(-1, 1)                  # [N, 1]

    N = obs_all.shape[0]

    # Normalize advantages (improves training stability)
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

    metrics = defaultdict(list)

    for _ in range(n_epochs):
        # Shuffle indices for mini-batches
        perm = torch.randperm(N)

        for start in range(0, N, mini_batch_size):
            idx = perm[start : start + mini_batch_size]

            obs_mb  = obs_all[idx]
            act_mb  = act_all[idx]
            lp_mb   = lp_old_all[idx]
            adv_mb  = adv_all[idx]
            ret_mb  = ret_all[idx]

            # ── Evaluate actor & critic ──────────────────────────────────
            log_prob_new, entropy = actor_eval(obs_mb, act_mb)
            value_new             = critic_net(obs_mb)              # [mb, 1]

            # ── Policy (clipped surrogate) loss ─────────────────────────
            ratio       = torch.exp(log_prob_new - lp_mb)          # [mb]
            surr1       = ratio * adv_mb
            surr2       = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            # ── Value function loss (clipped) ────────────────────────────
            value_loss  = nn.functional.mse_loss(value_new, ret_mb)

            # ── Entropy bonus ────────────────────────────────────────────
            entropy_loss = -entropy.mean()

            # ── Total loss ───────────────────────────────────────────────
            loss = (
                policy_loss
                + CRITIC_COEF  * value_loss
                + ENTROPY_COEF * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor_net.parameters()) + list(critic_net.parameters()),
                MAX_GRAD_NORM,
            )
            optimizer.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(-entropy_loss.item())
            metrics["total_loss"].append(loss.item())

    return {k: np.mean(v) for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    total_frames_collected = 0
    iteration              = 0
    reward_history         = []

    pbar = tqdm.tqdm(total=TOTAL_FRAMES, desc="PPO Training")

    while total_frames_collected < TOTAL_FRAMES:
        # ── 1. Collect data ──────────────────────────────────────────────
        actor_net.eval()
        critic_net.eval()

        with torch.no_grad():
            batch = collect_batch(env, n_steps=FRAMES_PER_BATCH)

        # ── 2. Compute GAE advantages & returns ──────────────────────────
        advantages, returns = compute_gae(
            rewards=batch["rewards"],
            values =batch["values"],
            dones  =batch["dones"],
        )
        batch["advantages"] = advantages
        batch["returns"]    = returns

        # Track reward for monitoring
        mean_reward = batch["rewards"].mean().item()
        reward_history.append(mean_reward)

        # ── 3. PPO update ────────────────────────────────────────────────
        actor_net.train()
        critic_net.train()

        metrics = ppo_update(
            batch          = batch,
            n_epochs       = NUM_EPOCHS,
            mini_batch_size= MINI_BATCH_SZ,
        )

        total_frames_collected += FRAMES_PER_BATCH
        iteration              += 1

        pbar.update(FRAMES_PER_BATCH)
        pbar.set_postfix(
            {
                "iter"        : iteration,
                "reward"      : f"{mean_reward:.4f}",
                "policy_loss" : f"{metrics['policy_loss']:.4f}",
                "value_loss"  : f"{metrics['value_loss']:.4f}",
                "entropy"     : f"{metrics['entropy']:.4f}",
            }
        )

        # Optional: print a more detailed summary every 10 iterations
        if iteration % 10 == 0:
            print(
                f"\n[Iter {iteration:4d} | Frames {total_frames_collected:6d}]"
                f"  reward={mean_reward:.4f}"
                f"  policy={metrics['policy_loss']:.4f}"
                f"  value={metrics['value_loss']:.4f}"
                f"  entropy={metrics['entropy']:.4f}"
            )

    pbar.close()
    print("\nTraining complete.")
    print(f"Final mean reward: {np.mean(reward_history[-10:]):.4f}")
    return reward_history


if __name__ == "__main__":
    reward_history = train()

    # ── Optional: plot learning curve ────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(reward_history, label="Mean reward per batch")
        window = min(20, len(reward_history))
        smoothed = np.convolve(
            reward_history, np.ones(window) / window, mode="valid"
        )
        plt.plot(range(window - 1, len(reward_history)), smoothed,
                 label=f"Smoothed (w={window})", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Mean Reward")
        plt.title("PPO Training – SimEnv")
        plt.legend()
        plt.tight_layout()
        plt.savefig("ppo_training_curve.png", dpi=150)
        plt.show()
        print("Learning curve saved to ppo_training_curve.png")
    except ImportError:
        print("matplotlib not available; skipping plot.")