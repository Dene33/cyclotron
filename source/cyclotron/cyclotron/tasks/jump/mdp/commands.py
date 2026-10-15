# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Commands for the jump: the tracking reference, played when the robot is told to jump, and the trigger.

The policy never sees the reference; it only shapes the rewards and the critic. Each robot cycles through three
phases, and the reference follows:

- STAND: the reference's first jump frame (``jump_start_s``, where the crouch starts), standing still, under the robot.
- JUMP: the reference from that frame to its end, from where the robot stood when told to jump.
- HOLD: the reference's last frame, standing still, under the robot.

The robot is told to jump through the velocity command the walking policy takes (``twist``): a forward speed while
the jump is commanded, zero otherwise. The command is held for the whole jump, so the policy can tell the crouch
before takeoff from the crouch after landing, and released after landing. Still held when the jump ends, it starts the
next jump.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul, sample_uniform, yaw_quat

from cyclotron.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

STAND, JUMP, HOLD = 0, 1, 2


class JumpCommand(MotionCommand):
    """The tracking reference, played when the robot is told to jump (see the module docstring)."""

    cfg: JumpCommandCfg

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.step_dt = env.cfg.decimation * env.cfg.sim.dt
        num_frames = self.motion.num_frames
        self.start_frame = int(round(cfg.jump_start_s / self.step_dt))
        if not 0 <= self.start_frame < num_frames - 1:
            raise ValueError(f"jump_start_s {cfg.jump_start_s} is outside the motion ({num_frames} frames).")
        airborne = ~self.reference_foot_contact.flatten(1).any(dim=1)
        landings = torch.where(airborne[:-1] & ~airborne[1:])[0] + 1
        if len(landings) == 0:
            raise ValueError(f"{cfg.motion_file} never leaves the ground: there is no jump to learn.")
        self.landing_frame = int(landings[-1])
        self.jump_frames = num_frames - self.start_frame

        # The standing frames: the first jump frame and the last frame, with no motion. They follow the motion's own
        # frames, so every reference quantity is still one index into the motion.
        self.stand_frame, self.hold_frame = num_frames, num_frames + 1
        self._append_still_frames([self.start_frame, num_frames - 1])

        self.phase = torch.full((self.num_envs,), STAND, dtype=torch.long, device=self.device)
        self.phase_steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.trigger_steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.trigger_speed = torch.zeros(self.num_envs, device=self.device)
        self.trigger = torch.zeros(self.num_envs, 3, device=self.device)
        # Where the reference is placed: turned about z by align_quat, then moved by align_pos.
        self.align_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.align_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.align_quat[:, 0] = 1.0

        # Adaptive sampling over the jump's frames only, one bin per second.
        self.bin_count = int(self.jump_frames * self.step_dt) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.metrics["jumping"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["told_to_jump"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"JumpCommand: jump from frame {self.start_frame} (landing at {self.landing_frame}) to"
            f" {self.motion.num_frames - 1}; " + super().__str__()
        )

    def _append_still_frames(self, frames: list[int]):
        """Append copies of ``frames`` with no motion: zero velocities, all sole points as they are."""
        ids = torch.tensor(frames, device=self.device)
        motion = self.motion
        for name in ("joint_pos", "body_pos_w", "body_quat_w", "root_pos_w", "root_quat_w", "all_body_pos_w",
                     "all_body_quat_w"):
            value = getattr(motion, name)
            setattr(motion, name, torch.cat([value, value[ids]]))
        for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w", "root_lin_vel_w", "root_ang_vel_w"):
            value = getattr(motion, name)
            setattr(motion, name, torch.cat([value, torch.zeros_like(value[ids])]))
        self.reference_foot_contact = torch.cat([self.reference_foot_contact, self.reference_foot_contact[ids]])
        self.reference_com_lin_vel_w = torch.cat(
            [self.reference_com_lin_vel_w, torch.zeros_like(self.reference_com_lin_vel_w[ids])]
        )
        self.pose_tracking_scale_by_frame = torch.cat(
            [self.pose_tracking_scale_by_frame, torch.ones(len(frames), device=self.device)]
        )

    # The reference, placed where it is played.

    def _placed_vector(self, vector: torch.Tensor) -> torch.Tensor:
        quat = self.align_quat if vector.dim() == 2 else self.align_quat[:, None, :].expand(*vector.shape[:-1], 4)
        return quat_apply(quat, vector)

    def _placed_position(self, position: torch.Tensor) -> torch.Tensor:
        offset = self.align_pos + self._env.scene.env_origins
        return self._placed_vector(position) + (offset if position.dim() == 2 else offset[:, None, :])

    def _placed_quat(self, quat: torch.Tensor) -> torch.Tensor:
        align = self.align_quat if quat.dim() == 2 else self.align_quat[:, None, :].expand_as(quat)
        return quat_mul(align, quat)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._placed_position(self.motion.body_pos_w[self.time_steps])

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._placed_quat(self.motion.body_quat_w[self.time_steps])

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._placed_vector(self.motion.body_lin_vel_w[self.time_steps])

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._placed_vector(self.motion.body_ang_vel_w[self.time_steps])

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._placed_position(self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index])

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._placed_quat(self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index])

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._placed_vector(self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index])

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._placed_vector(self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index])

    @property
    def com_lin_vel_w(self) -> torch.Tensor:
        return self._placed_vector(self.reference_com_lin_vel_w[self.time_steps])

    def _place_under_robot(self, env_ids: torch.Tensor, frames: torch.Tensor):
        """Place the reference so that its anchor at ``frames`` has the robot anchor's horizontal position and
        heading."""
        motion_pos = self.motion.body_pos_w[frames, self.motion_anchor_body_index]
        motion_quat = self.motion.body_quat_w[frames, self.motion_anchor_body_index]
        turn = yaw_quat(quat_mul(self.robot_anchor_quat_w[env_ids], quat_inv(motion_quat)))
        position = self.robot_anchor_pos_w[env_ids] - self._env.scene.env_origins[env_ids] - quat_apply(turn, motion_pos)
        position[:, 2] = 0.0
        self.align_quat[env_ids] = turn
        self.align_pos[env_ids] = position

    # The phases.

    def _steps(self, time_range: tuple[float, float], count: int) -> torch.Tensor:
        seconds = sample_uniform(time_range[0], time_range[1], (count,), device=self.device)
        return torch.clamp((seconds / self.step_dt).round().long(), min=1)

    def _tell_to_jump(self, env_ids: torch.Tensor, frames: torch.Tensor):
        """Command the jump of ``env_ids``, now at ``frames``: released after landing, or now and then held on into
        the next jump."""
        num_frames = self.motion.num_frames
        release = self.landing_frame + self._steps(self.cfg.release_after_landing_s, len(env_ids))
        release = torch.clamp(release, max=num_frames)
        again = torch.rand(len(env_ids), device=self.device) < self.cfg.jump_again_prob
        release = torch.where(again, torch.full_like(release, num_frames + 1), release)
        self.trigger_steps_left[env_ids] = torch.clamp(release - frames, min=0)
        self.trigger_speed[env_ids] = sample_uniform(*self.cfg.trigger_speed, (len(env_ids),), device=self.device)

    def _start_jump(self, env_ids: torch.Tensor):
        frames = torch.full((len(env_ids),), self.start_frame, dtype=torch.long, device=self.device)
        self.phase[env_ids] = JUMP
        self.time_steps[env_ids] = frames
        self._place_under_robot(env_ids, frames)
        self._tell_to_jump(env_ids, frames)

    def _start_still(self, env_ids: torch.Tensor, phase: int):
        self.phase[env_ids] = phase
        self.time_steps[env_ids] = self.stand_frame if phase == STAND else self.hold_frame
        time_range = self.cfg.stand_time_s if phase == STAND else self.cfg.hold_time_s
        self.phase_steps_left[env_ids] = self._steps(time_range, len(env_ids))

    def _sample_jump_frames(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Frames of the jump to start ``env_ids`` on, sampled more often where robots failed recently."""
        failed = self._env.termination_manager.terminated[env_ids] & (self.phase[env_ids] == JUMP)
        if torch.any(failed):
            offsets = self.time_steps[env_ids][failed] - self.start_frame
            bins = torch.clamp((offsets * self.bin_count) // self.jump_frames, 0, self.bin_count - 1)
            self._current_bin_failed[:] = torch.bincount(bins, minlength=self.bin_count)
        probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        probabilities = torch.nn.functional.pad(
            probabilities.unsqueeze(0).unsqueeze(0), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
        )
        probabilities = torch.nn.functional.conv1d(probabilities, self.kernel.view(1, 1, -1)).view(-1)
        probabilities = probabilities / probabilities.sum()
        bins = torch.multinomial(probabilities, len(env_ids), replacement=True)
        offsets = (bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)) / self.bin_count
        return self.start_frame + (offsets * (self.jump_frames - 1)).long()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        jumping = torch.rand(len(env_ids), device=self.device) >= self.cfg.stand_start_prob
        jump_frames = self._sample_jump_frames(env_ids)
        self._start_still(env_ids, STAND)
        self.phase[env_ids[jumping]] = JUMP
        self.time_steps[env_ids[jumping]] = jump_frames[jumping]
        self.trigger_steps_left[env_ids] = 0
        self._tell_to_jump(env_ids[jumping], jump_frames[jumping])
        self.align_pos[env_ids] = 0.0
        self.align_quat[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        frames = self.time_steps[env_ids]

        # The robot on the reference, as in motion tracking; standing robots now and then in the default pose, where
        # they stand when the walking policy hands over.
        axes = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor([self.cfg.pose_range.get(key, (0.0, 0.0)) for key in axes], device=self.device)
        noise = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos = self.motion.root_pos_w[frames] + self._env.scene.env_origins[env_ids] + noise[:, 0:3]
        root_quat = quat_mul(quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5]), self.motion.root_quat_w[frames])
        ranges = torch.tensor([self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in axes], device=self.device)
        noise = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel = self.motion.root_lin_vel_w[frames] + noise[:, :3]
        root_ang_vel = self.motion.root_ang_vel_w[frames] + noise[:, 3:]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        on_reference = jumping | (torch.rand(len(env_ids), device=self.device) >= self.cfg.default_pose_prob)
        reference_pos = joint_pos[:, self.joint_ids]
        reference_pos[on_reference] = self.motion.joint_pos[frames][on_reference]
        joint_pos[:, self.joint_ids] = reference_pos + sample_uniform(
            *self.cfg.joint_position_range, (len(env_ids), len(self.joint_ids)), self.device
        )
        joint_vel[:, self.joint_ids] = self.motion.joint_vel[frames]
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, limits[:, :, 0], limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat([root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids)
        self._update_trigger()
        self._move_reference_under_robot()

    def _update_trigger(self):
        self.trigger.zero_()
        self.trigger[:, 0] = torch.where(self.trigger_steps_left > 0, self.trigger_speed, 0.0)

    def _update_command(self):
        jumping = self.phase == JUMP
        self.time_steps[jumping] += 1
        self.trigger_steps_left = torch.clamp(self.trigger_steps_left - 1, min=0)
        self.phase_steps_left = torch.clamp(self.phase_steps_left - 1, min=0)

        # A finished jump: the next one while the robot is still told to jump, else hold the landing.
        ended = jumping & (self.time_steps >= self.motion.num_frames)
        again = ended & (self.trigger_steps_left > 0)
        holding_over = (self.phase == HOLD) & (self.phase_steps_left == 0)
        standing_over = (self.phase == STAND) & (self.phase_steps_left == 0)
        self._start_jump(torch.where(again | standing_over)[0])
        self._start_still(torch.where(ended & ~again)[0], HOLD)
        self._start_still(torch.where(holding_over)[0], STAND)

        # Standing still, the reference stays under the robot.
        still = torch.where(self.phase != JUMP)[0]
        self._place_under_robot(still, self.time_steps[still])
        self._update_trigger()
        self._move_reference_under_robot()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["jumping"] = (self.phase == JUMP).float()
        self.metrics["told_to_jump"] = (self.trigger_steps_left > 0).float()


@configclass
class JumpCommandCfg(MotionCommandCfg):
    """Configuration for :class:`JumpCommand`."""

    class_type: type = JumpCommand

    jump_start_s: float = 0.5
    """Where the jump starts in the motion: the end of its standing start, where the crouch begins."""

    stand_time_s: tuple[float, float] = (0.5, 3.0)
    """How long a robot stands before it is told to jump."""

    hold_time_s: tuple[float, float] = (0.5, 1.5)
    """How long a robot holds its landing pose before standing in the starting pose again."""

    release_after_landing_s: tuple[float, float] = (0.2, 0.6)
    """When the jump command is released, after the reference lands."""

    jump_again_prob: float = 0.2
    """How often the command is still held when the jump ends, so the robot jumps again."""

    trigger_speed: tuple[float, float] = (0.25, 0.4)
    """The forward speed the command reads while the robot is told to jump (m/s; the firmware caps it at 0.4)."""

    stand_start_prob: float = 0.6
    """How often an episode starts standing rather than somewhere in the jump."""

    default_pose_prob: float = 0.5
    """How often a robot that starts standing starts in the default pose rather than the reference's."""


class JumpTriggerCommand(CommandTerm):
    """The velocity command (forward speed, sideways speed, yaw rate) the walking policy takes, as the jump's trigger:
    a forward speed while the robot is told to jump, zero otherwise. It follows the :class:`JumpCommand` named
    ``jump_command_name``."""

    cfg: JumpTriggerCommandCfg

    @property
    def command(self) -> torch.Tensor:
        return self._env.command_manager.get_term(self.cfg.jump_command_name).trigger

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass


@configclass
class JumpTriggerCommandCfg(CommandTermCfg):
    """Configuration for :class:`JumpTriggerCommand`."""

    class_type: type = JumpTriggerCommand

    jump_command_name: str = MISSING
    """The :class:`JumpCommand` whose trigger this is."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
