from typing import Any, Dict, List

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    Binary,
    Categorical,
    Composite,
    OneHot,
    UnboundedContinuous,
    UnboundedDiscrete,
)
from torchrl.envs import EnvBase
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import check_env_specs, make_composite_from_td

from io_api import _build_graph_from_config, _build_node_from_config
from scheduler import Simulator


class SimEnv(EnvBase):
    batch_locked = False

    def __init__(self, config: Dict[str, Any], seed=None, device=None):
        self.config = config

        self.device_ids: list = []
        for dev_cfg in config["hardware_config"]["devices"]:
            self.device_ids.append(dev_cfg["id"])
        self.num_devices = len(self.device_ids)

        self.op_ids: list = []
        for op_cfg in config["graph"]["ops"]:
            self.op_ids.append(op_cfg["id"])
        self.num_ops = len(self.op_ids)

        td_params = self.gen_params(self.num_devices, self.num_ops, device=device)
        super().__init__(device=device)
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)

    def _step(self, tensordict):
        # TODO: make sure out has all the same keys as tensordict, and that the shape is correct
        action = tensordict["action"]
        selected_device_idx = torch.argmax(action, dim=0)
        selected_device_ids = [self.device_ids[i] for i in selected_device_idx]

        for i, op_id in enumerate(self.op_ids):
            device_id = selected_device_ids[i]
            self.config["graph"]["ops"][i]["device_id"] = device_id

        node = _build_node_from_config(self.config)
        graph = _build_graph_from_config(self.config)

        simulator = Simulator(node=node, graph=graph)
        result = simulator.run()
        reward = -result.total_runtime

        out = TensorDict(
            {
                "action": action,
                "params": tensordict["params"],
                "observation": result.total_runtime,
                "reward": reward,
                "done": torch.tensor(True),
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict):
        batch_size = (
            tensordict.batch_size if tensordict is not None else self.batch_size
        )
        if tensordict is None or "params" not in tensordict:
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(
                self.num_devices,
                self.num_ops,
                batch_size=batch_size,
                device=self.device,
            )
        action = torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        out = TensorDict(
            {
                "action": action,
                "params": tensordict["params"],
                "observation": torch.zeros(tensordict.shape, device=self.device),
            },
            tensordict.shape,
        )
        return out

    def _make_spec(self, td_params):
        self.observation_spec = Composite(
            observation=UnboundedContinuous(
                shape=(1,),
                dtype=torch.float32,
            ),
            params=make_composite_from_td(
                td_params["params"], unsqueeze_null_shapes=False
            ),
            shape=(),
        )

        self.state_spec = self.observation_spec.clone()

        self.action_spec = OneHot(
            n=self.num_devices, shape=(self.num_ops, self.num_devices)
        )

        self.reward_spec = UnboundedContinuous(
            shape=(*td_params.shape, 1),
            dtype=torch.float32,
        )

    def _set_seed(self, seed):
        self.rng = torch.random.manual_seed(seed)

    @staticmethod
    def gen_params(
        num_devices: int, num_ops: int, batch_size=None, device=None
    ) -> TensorDictBase:
        """Returns a ``tensordict`` containing the simulation parameters such as number of devices and number of ops."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "num_devices": num_devices,
                        "num_ops": num_ops,
                    },
                    [],
                )
            },
            [],
            device=device,
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td
