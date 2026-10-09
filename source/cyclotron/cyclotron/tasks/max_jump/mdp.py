# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Max-jump MDP terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from cyclotron.tasks.locomotion.mdp import randomize_joint_default_pos  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def episode_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fraction of the episode elapsed, so the policy can time its jumps."""
    return (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1)


def airborne(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 while every body in ``sensor_cfg`` (the feet) is off the ground."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.all(sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0, dim=-1).float()


def airborne_height(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, standing_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Pelvis height above standing, in units of 10 cm, while both feet are off the ground."""
    height = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2] - standing_height
    return airborne(env, sensor_cfg) * torch.clamp(height, min=0.0) / 0.1


def feet_height_asymmetry(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, tolerance: float) -> torch.Tensor:
    """Height difference between the feet beyond ``tolerance``, in units of 10 cm. Keeps the jump two-footed:
    without it the policy hops on one leg and swings the other one up for momentum."""
    feet_z = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    return torch.clamp(torch.abs(feet_z[:, 0] - feet_z[:, 1]) - tolerance, min=0.0) / 0.1


def horizontal_speed_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared horizontal pelvis speed, to keep the jump vertical and in place."""
    return torch.sum(torch.square(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, :2]), dim=-1)


def rising_velocity(env: ManagerBasedRLEnv, standing_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Upward pelvis speed while the pelvis is above standing height: rewards the end of the push-off and the rise,
    but not bobbing up and down in a crouch."""
    data = env.scene[asset_cfg.name].data
    above = data.root_link_pos_w[:, 2] > standing_height
    return above * torch.clamp(data.root_link_lin_vel_w[:, 2], min=0.0)
