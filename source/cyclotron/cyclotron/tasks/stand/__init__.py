"""Asimov-1 standing tasks for the RL crash course, Module 1."""

import gymnasium as gym

from . import agents

gym.register(
    id="Asimov1-Stand-Ideal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:Asimov1StandIdealEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1StandIdealPPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-Stand-Ideal-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:Asimov1StandIdealEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1StandIdealPPORunnerCfg",
    },
)
