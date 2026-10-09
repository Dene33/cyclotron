# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 max-jump task: a diagnostic for how high the robot can jump."""

import gymnasium as gym

from . import agents

# Register Gym environments.

gym.register(
    id="Asimov1-MaxJump-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.max_jump_env_cfg:Asimov1MaxJumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1MaxJumpPPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-MaxJump-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.max_jump_env_cfg:Asimov1MaxJumpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1MaxJumpPPORunnerCfg",
    },
)
