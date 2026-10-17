# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Motion tracking rewards.

Adapted from whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking), Copyright (c) 2024,
The Isaac Lab Project Developers, MIT License.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_error_magnitude, yaw_quat

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """Indexes of ``body_names`` among the command's tracked bodies; all of them when ``None``."""
    return [i for i, name in enumerate(command.cfg.body_names) if body_names is None or name in body_names]


def _pose_scale(command: MotionCommand, relax_before_takeoff: bool) -> torch.Tensor | float:
    """Pose tracking weight at the current frame: lowered before each takeoff when ``relax_before_takeoff``."""
    return command.pose_tracking_scale if relax_before_takeoff else 1.0


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, relax_before_takeoff: bool = False
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2) * _pose_scale(command, relax_before_takeoff)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, relax_before_takeoff: bool = False
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2) * _pose_scale(command, relax_before_takeoff)


def motion_foot_contact_match(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Share of sole points whose ground contact matches the reference: heels and toes touch down and lift off at the
    same frames. Times the takeoff and landing, rules out extra hops, and keeps a jump's push on the toes."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return (command.robot_foot_contact == command.foot_contact).float().mean(dim=(1, 2))


def motion_com_takeoff_velocity_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, min_reference_rise: float, axes: str = "z"
) -> torch.Tensor:
    """Track the whole-body centre-of-mass velocity while the reference's centre of mass rises faster than
    ``min_reference_rise``.

    ``axes`` is ``"z"`` for the vertical velocity (jump height) or ``"xy"`` for the horizontal one (jump distance).
    Zero at all other frames. For a jump this is the push-off, which the even tracking rewards weigh too little to get
    the takeoff impulse right; motions without a fast rise are not affected. The centre of mass rather than the pelvis,
    because a body pivoting around the toes moves the pelvis fast without the push a jump needs.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    ids = {"z": [2], "xy": [0, 1]}[axes]
    reference = command.com_lin_vel_w
    error = torch.sum(torch.square(reference[:, ids] - command.robot_com_lin_vel_w[:, ids]), dim=-1)
    return (reference[:, 2] > min_reference_rise) * torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    relax_before_takeoff: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ids = body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_pos_relative_w[:, ids] - command.robot_body_pos_w[:, ids]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2) * _pose_scale(command, relax_before_takeoff)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    relax_before_takeoff: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ids = body_indexes(command, body_names)
    error = quat_error_magnitude(command.body_quat_relative_w[:, ids], command.robot_body_quat_w[:, ids]) ** 2
    return torch.exp(-error.mean(-1) / std**2) * _pose_scale(command, relax_before_takeoff)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ids = body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_lin_vel_w[:, ids] - command.robot_body_lin_vel_w[:, ids]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ids = body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_ang_vel_w[:, ids] - command.robot_body_ang_vel_w[:, ids]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


# Asimov 1's leg joints mirror: the same motion of both legs has q_right = -q_left, so q_left + q_right is a pair's
# asymmetry.
LEG_JOINT_PAIRS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]


def joint_indexes(command: MotionCommand, joint_names: list[str]) -> list[int]:
    """Indexes of the motion's joints that match any of the ``joint_names`` regexes."""
    return [i for i, name in enumerate(command.motion.joint_names) if any(re.fullmatch(p, name) for p in joint_names)]


def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, joint_names: list[str]
) -> torch.Tensor:
    """Tracking of the reference's joint positions for ``joint_names``, e.g. the arms and waist, which the body pose
    rewards barely see: a wrist turned on its own axis moves no tracked body."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ids = joint_indexes(command, joint_names)
    error = torch.sum(torch.square(command.joint_pos[:, ids] - command.robot_joint_pos[:, ids]), dim=-1)
    return torch.exp(-error / std**2)


def motion_leg_symmetry_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """The legs' left/right asymmetry matching the reference's, pair by pair (``LEG_JOINT_PAIRS``), so the legs move
    together as the reference's do. Rewarding symmetry rather than each leg's pose also holds when the robot runs
    early or late: the body pose rewards then pay for moving one leg on the reference's time while the other lags."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    names = command.motion.joint_names
    left = [names.index(f"left_{pair}_joint") for pair in LEG_JOINT_PAIRS]
    right = [names.index(f"right_{pair}_joint") for pair in LEG_JOINT_PAIRS]
    robot, reference = command.robot_joint_pos, command.joint_pos
    error = torch.sum(torch.square((robot[:, left] + robot[:, right]) - (reference[:, left] + reference[:, right])), dim=-1)
    return torch.exp(-error / std**2)


def motion_feet_height_difference(env: ManagerBasedRLEnv, command_name: str, tolerance: float) -> torch.Tensor:
    """How much the robot's feet differ in height beyond the reference's feet (plus ``tolerance``), in units of 10 cm:
    a penalty for lifting or landing one foot before the other."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet = command.motion_foot_ids
    robot_z = command.robot_body_pos_w[:, feet, 2]
    reference_z = command.body_pos_relative_w[:, feet, 2]
    difference = torch.abs((robot_z[:, 0] - robot_z[:, 1]) - (reference_z[:, 0] - reference_z[:, 1]))
    return torch.clamp(difference - tolerance, min=0.0) / 0.1


def motion_anchor_heading_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Tracking of the reference's heading (the anchor's yaw), so the robot jumps where the reference does."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(yaw_quat(command.anchor_quat_w), yaw_quat(command.robot_anchor_quat_w)) ** 2
    return torch.exp(-error / std**2)
