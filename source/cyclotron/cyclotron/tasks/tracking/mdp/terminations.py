# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Motion tracking terminations: end the episode when the robot drifts too far from the reference.

Adapted from whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking), Copyright (c) 2024,
The Isaac Lab Project Developers, MIT License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import body_indexes

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Anchor height differs from the reference by more than ``threshold``."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Anchor tilt differs from the reference by more than ``threshold``, compared through projected gravity."""
    asset: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return (motion_gravity_b[:, 2] - robot_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Any of ``body_names`` is further than ``threshold`` above or below its reference height."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ids = body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, ids, -1] - command.robot_body_pos_w[:, ids, -1])
    return torch.any(error > threshold, dim=-1)
