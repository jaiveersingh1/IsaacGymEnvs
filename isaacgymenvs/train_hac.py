import isaacgym

import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from isaacgymenvs.tasks import isaacgym_task_map

from HACKerMan.agents.HAC_agent import (
    ExplorationParams,
    HACCoordinator,
    HACParams,
)

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # dump config dict
    experiment_dir = os.path.join("runs", cfg.train.params.config.name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    env: VecTask = isaacgym_task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
    )

    params = HACParams(
        lambda_=0.3,
        gamma=0.95,
        k=3,
        horizon=2,
        lr=0.001,
        action_std=0.1,
        state_std=0.02,
        goal_threshold=0.01,
        batch_size=100,
        num_update_steps=3,
    )

    exploration_params = ExplorationParams(
        exploration_strategy="Normal",
        exploration_frequency=0.2,
        noise_mean=0,
        noise_std=0.02,
    )

    # For cartpole, goal is all zeroes
    goal_state = torch.zeros(env.num_obs)

    initial_state = env.reset()
    coordinator = HACCoordinator(
        env=env,
        initial_state=initial_state,
        goal_state=goal_state,
        params=params,
        exploration_params=exploration_params,
    )

    num_episodes = 100
    for i in range(num_episodes):
        print(f"Episode {i}")
        coordinator.train(100)


if __name__ == "__main__":
    launch_rlg_hydra()
