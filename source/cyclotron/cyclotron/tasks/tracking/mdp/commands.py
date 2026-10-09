# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reference motion command for motion tracking.

Adapted from whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking), Copyright (c) 2024,
The Isaac Lab Project Developers, MIT License. Changes: joints and bodies are matched to the motion file by name,
body velocities are link-frame, and Play can always start the motion from its first frame.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """A motion file written by ``./cyclotron.sh --csv-to-npz``: the tracked bodies, the root, and the poses of all
    of ``all_body_names`` (for the centre of mass)."""

    def __init__(
        self, motion_file: str, body_names: list[str], root_body_name: str, all_body_names: list[str], device: str
    ):
        if not motion_file:
            raise ValueError(
                "No motion file set. Pass --motion_file <motion.npz>; make one with ./cyclotron.sh --csv-to-npz."
            )
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        data = np.load(motion_file)
        for key in ("joint_names", "body_names"):
            if key not in data.files:
                raise ValueError(
                    f"{motion_file} has no '{key}'. Convert the motion again with ./cyclotron.sh --csv-to-npz."
                )

        self.fps = float(np.asarray(data["fps"]).item())
        self.joint_names = [str(name) for name in data["joint_names"]]
        file_body_names = [str(name) for name in data["body_names"]]
        missing = [name for name in [*body_names, root_body_name, *all_body_names] if name not in file_body_names]
        if missing:
            raise ValueError(f"{motion_file} has no bodies {missing}.")
        body_ids = [file_body_names.index(name) for name in body_names]
        root_id = file_body_names.index(root_body_name)

        def load(key: str, ids) -> torch.Tensor:
            return torch.tensor(data[key][:, ids], dtype=torch.float32, device=device)

        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = load("body_pos_w", body_ids)
        self.body_quat_w = load("body_quat_w", body_ids)
        self.body_lin_vel_w = load("body_lin_vel_w", body_ids)
        self.body_ang_vel_w = load("body_ang_vel_w", body_ids)
        self.root_pos_w = load("body_pos_w", root_id)
        self.root_quat_w = load("body_quat_w", root_id)
        self.root_lin_vel_w = load("body_lin_vel_w", root_id)
        self.root_ang_vel_w = load("body_ang_vel_w", root_id)
        all_ids = [file_body_names.index(name) for name in all_body_names]
        self.all_body_pos_w = load("body_pos_w", all_ids)
        self.all_body_quat_w = load("body_quat_w", all_ids)
        self.num_frames = self.joint_pos.shape[0]


class MotionCommand(CommandTerm):
    """Plays a reference motion, one frame per environment step, and resets the robot onto it.

    The command is the reference joint positions and velocities at the current frame. Episodes start at a frame
    sampled more often where the robot failed recently (adaptive sampling), or at the first frame when
    ``start_from_first_frame`` is set. When the motion ends, the robot is reset onto a new start frame.
    """

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        if cfg.anchor_body_name not in cfg.body_names:
            raise ValueError(f"anchor_body_name '{cfg.anchor_body_name}' must be one of body_names.")
        self.motion = MotionLoader(
            cfg.motion_file, cfg.body_names, self.robot.body_names[0], self.robot.body_names, self.device
        )

        step_rate = 1.0 / (env.cfg.decimation * env.cfg.sim.dt)
        if abs(self.motion.fps - step_rate) > 1.0e-3:
            raise ValueError(
                f"{cfg.motion_file} is {self.motion.fps:g} fps but the policy runs at {step_rate:g} Hz. Convert it"
                f" again with --output_fps {step_rate:g}."
            )

        self.joint_ids = self.robot.find_joints(self.motion.joint_names, preserve_order=True)[0]
        self.body_ids = self.robot.find_bodies(cfg.body_names, preserve_order=True)[0]
        self.robot_anchor_body_index = self.robot.find_bodies(cfg.anchor_body_name)[0][0]
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)

        # Which sole points of the reference touch the ground at each frame, and from that, how strongly to track the
        # pose: less in the moments before each takeoff, so the robot can push off its own way.
        missing = [name for name in cfg.foot_body_names if name not in cfg.body_names]
        if missing:
            raise ValueError(f"foot_body_names {missing} must be among body_names.")
        self.motion_foot_ids = [cfg.body_names.index(name) for name in cfg.foot_body_names]
        self.robot_foot_ids = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)[0]
        self.sole_points = torch.tensor(cfg.foot_sole_points, dtype=torch.float, device=self.device)
        self.reference_foot_contact = self._sole_contact(
            self.motion.body_pos_w[:, self.motion_foot_ids], self.motion.body_quat_w[:, self.motion_foot_ids]
        )
        self.pose_tracking_scale_by_frame = self._pose_tracking_scale(step_rate)

        # Whole-body centre of mass, with the robot's masses: the takeoff rewards track its velocity, which only a real
        # push changes. A body pivoting around the toes moves the pelvis fast without it.
        self.robot_masses = self.robot.data.default_mass.to(self.device)
        self.reference_com_lin_vel_w = self._reference_com_velocity(step_rate)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # One adaptive-sampling bin per second of motion.
        self.bin_count = int(self.motion.num_frames // step_rate) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        for name in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
            "error_com_lin_vel",
            "foot_contact_match",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _sole_contact(self, foot_pos_w: torch.Tensor, foot_quat_w: torch.Tensor) -> torch.Tensor:
        """Which sole points touch the ground, shaped (..., feet, sole points), from foot poses (..., feet, 3 or 4)."""
        points = self.sole_points.expand(*foot_quat_w.shape[:-1], *self.sole_points.shape)
        quat = foot_quat_w.unsqueeze(-2).expand(*points.shape[:-1], 4)
        heights = foot_pos_w[..., 2:3] + quat_apply(quat, points)[..., 2]
        return heights < self.cfg.contact_height

    def _reference_com_velocity(self, step_rate: float) -> torch.Tensor:
        """Centre-of-mass velocity of the reference at every frame, (frames, 3)."""
        masses = self.robot_masses[0]
        pos, quat = self.motion.all_body_pos_w, self.motion.all_body_quat_w
        com_offsets = self.robot.data.body_com_pos_b[0].expand_as(pos)
        com = (masses[None, :, None] * (pos + quat_apply(quat, com_offsets))).sum(dim=1) / masses.sum()
        return torch.gradient(com, spacing=1.0 / step_rate, dim=0)[0]

    def _pose_tracking_scale(self, step_rate: float) -> torch.Tensor:
        """Per-frame weight of the pose tracking rewards: ``takeoff_pose_scale`` during the ``takeoff_window_s`` before
        each takeoff of the reference (every sole point leaving the ground), 1 elsewhere."""
        airborne = ~self.reference_foot_contact.flatten(1).any(dim=1)
        scale = torch.ones(self.motion.num_frames, device=self.device)
        window = int(round(self.cfg.takeoff_window_s * step_rate))
        for takeoff in (torch.where(airborne[1:] & ~airborne[:-1])[0] + 1).tolist():
            scale[max(takeoff - window, 0) : takeoff] = self.cfg.takeoff_pose_scale
        return scale

    def __str__(self) -> str:
        return (
            f"MotionCommand: {os.path.basename(self.cfg.motion_file)}, {self.motion.num_frames} frames at"
            f" {self.motion.fps:g} fps, {len(self.joint_ids)} joints, {len(self.body_ids)} tracked bodies"
        )

    # Reference motion at the current frame.

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def foot_contact(self) -> torch.Tensor:
        """Reference sole contacts at the current frame, (envs, feet, sole points)."""
        return self.reference_foot_contact[self.time_steps]

    @property
    def com_lin_vel_w(self) -> torch.Tensor:
        """Reference centre-of-mass velocity at the current frame, (envs, 3)."""
        return self.reference_com_lin_vel_w[self.time_steps]

    @property
    def pose_tracking_scale(self) -> torch.Tensor:
        """Weight of the pose tracking rewards at the current frame, (envs,)."""
        return self.pose_tracking_scale_by_frame[self.time_steps]

    # The simulated robot, in the same joint and body order as the reference.

    @property
    def robot_com_lin_vel_w(self) -> torch.Tensor:
        """Robot centre-of-mass velocity, (envs, 3)."""
        momentum = (self.robot_masses.unsqueeze(-1) * self.robot.data.body_com_lin_vel_w).sum(dim=1)
        return momentum / self.robot_masses.sum(dim=1, keepdim=True)

    @property
    def robot_foot_contact(self) -> torch.Tensor:
        """Robot sole contacts, (envs, feet, sole points)."""
        data = self.robot.data
        return self._sole_contact(
            data.body_link_pos_w[:, self.robot_foot_ids], data.body_link_quat_w[:, self.robot_foot_ids]
        )

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.joint_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.joint_ids]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_ids]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_ids]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_ids]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            -1
        )
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            -1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            -1
        )
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        self.metrics["error_com_lin_vel"] = torch.norm(self.com_lin_vel_w - self.robot_com_lin_vel_w, dim=-1)
        self.metrics["foot_contact_match"] = (self.robot_foot_contact == self.foot_contact).float().mean(dim=(1, 2))

    def _sample_start_frames(self, env_ids: Sequence[int]):
        if self.cfg.start_from_first_frame:
            self.time_steps[env_ids] = 0
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.num_frames, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Failures spread backwards in time, so the frames leading up to a failure are practised too.
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.num_frames - 1)
        ).long()

        entropy = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        top_prob, top_bin = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = entropy / math.log(max(self.bin_count, 2))
        self.metrics["sampling_top1_prob"][:] = top_prob
        self.metrics["sampling_top1_bin"][:] = top_bin.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._sample_start_frames(env_ids)
        frames = self.time_steps[env_ids]

        root_pos = self.motion.root_pos_w[frames] + self._env.scene.env_origins[env_ids]
        root_quat = self.motion.root_quat_w[frames]
        root_lin_vel = self.motion.root_lin_vel_w[frames].clone()
        root_ang_vel = self.motion.root_ang_vel_w[frames].clone()

        axes = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor([self.cfg.pose_range.get(key, (0.0, 0.0)) for key in axes], device=self.device)
        noise = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos = root_pos + noise[:, 0:3]
        root_quat = quat_mul(quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5]), root_quat)
        ranges = torch.tensor([self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in axes], device=self.device)
        noise = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += noise[:, :3]
        root_ang_vel += noise[:, 3:]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self.joint_ids] = self.motion.joint_pos[frames] + sample_uniform(
            *self.cfg.joint_position_range, (len(env_ids), len(self.joint_ids)), self.device
        )
        joint_vel[:, self.joint_ids] = self.motion.joint_vel[frames]
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, limits[:, :, 0], limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat([root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids)
        # Without this, the first step after the environment's first reset would compare the robot with no reference
        # at all and end every episode.
        self._move_reference_under_robot()

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.num_frames)[0]
        self._resample_command(env_ids)
        self._move_reference_under_robot()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _move_reference_under_robot(self):
        """Reference body poses moved under the robot: the robot's horizontal anchor position and heading, with the
        reference's anchor height."""
        num_bodies = len(self.cfg.body_names)
        anchor_pos_w = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        anchor_quat_w = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
        delta_pos_w = self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        delta_pos_w[..., 2] = anchor_pos_w[..., 2]
        robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )
                body_cfg = self.cfg.body_visualizer_cfg
                self.current_body_visualizers = [
                    VisualizationMarkers(body_cfg.replace(prim_path=f"/Visuals/Command/current/{name}"))
                    for name in self.cfg.body_names
                ]
                self.goal_body_visualizers = [
                    VisualizationMarkers(body_cfg.replace(prim_path=f"/Visuals/Command/goal/{name}"))
                    for name in self.cfg.body_names
                ]
            visualizers = [self.current_anchor_visualizer, self.goal_anchor_visualizer]
            for visualizer in visualizers + self.current_body_visualizers + self.goal_body_visualizers:
                visualizer.set_visibility(True)
        elif hasattr(self, "current_anchor_visualizer"):
            visualizers = [self.current_anchor_visualizer, self.goal_anchor_visualizer]
            for visualizer in visualizers + self.current_body_visualizers + self.goal_body_visualizers:
                visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the reference motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = ""
    """Motion file from ``./cyclotron.sh --csv-to-npz``. Set it with ``--motion_file``."""

    anchor_body_name: str = MISSING
    """Body whose pose is compared with the reference in the world frame. Other bodies are compared relative to it."""

    body_names: list[str] = MISSING
    """Bodies whose poses and velocities are tracked."""

    pose_range: dict[str, tuple[float, float]] = {}
    """Uniform noise added to the root pose at reset, per axis (x, y, z, roll, pitch, yaw)."""

    velocity_range: dict[str, tuple[float, float]] = {}
    """Uniform noise added to the root velocity at reset, per axis (x, y, z, roll, pitch, yaw)."""

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    """Uniform noise added to the joint positions at reset."""

    start_from_first_frame: bool = False
    """Always start from the first frame instead of sampling start frames. Used for Play."""

    foot_body_names: list[str] = MISSING
    """Feet, for foot contacts. Must be among ``body_names``."""

    foot_sole_points: list[tuple[float, float, float]] = MISSING
    """Points under each foot that touch the ground (e.g. heel and toe), in the foot body's frame."""

    contact_height: float = 0.015
    """A sole point lower than this touches the ground."""

    takeoff_window_s: float = 0.4
    """How long before each takeoff of the reference the pose tracking rewards are scaled down."""

    takeoff_pose_scale: float = 0.2
    """Scale of the pose tracking rewards in that window, so the robot can choose its own push-off."""

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
