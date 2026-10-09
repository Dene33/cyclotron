# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Motion tracking observations.

Adapted from whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking), Copyright (c) 2024,
The Isaac Lab Project Developers, MIT License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix, flattened per environment."""
    mat = matrix_from_quat(quat)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference anchor position in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference anchor orientation in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    return _rotation_6d(ori)


def _robot_bodies_in_anchor_frame(command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    num_bodies = len(command.cfg.body_names)
    return subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Tracked body positions in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos_b, _ = _robot_bodies_in_anchor_frame(command)
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Tracked body orientations in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, ori_b = _robot_bodies_in_anchor_frame(command)
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)
