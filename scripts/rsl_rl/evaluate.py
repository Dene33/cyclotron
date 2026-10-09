# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure how reliably a motion tracking policy plays its whole motion.

Many robots start on the reference's first frame and play it to the end under the training task's randomisation.
Reports the share that falls (any termination before the end) and when, how high and far the robots that finish
get, and the peaks of their joint speed, motor torque and foot force.
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a motion tracking policy on many randomised robots.")
parser.add_argument("--task", type=str, default="Asimov1-Tracking-v0", help="Training task; its randomisation is used.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of robots.")
parser.add_argument(
    "--condition",
    choices=["nominal", "randomized", "pushes"],
    default="randomized",
    help="nominal: no randomisation or sensor noise; randomized: the training randomisation without pushes;"
    " pushes: with the training pushes too.",
)
parser.add_argument(
    "--motor_speed_ratio",
    type=float,
    default=None,
    help="Model motor torque falling with speed: to zero at this multiple of each actuator's maximum speed."
    " The real value is not published but is at least 1, the weakest motor.",
)
parser.add_argument("--seed", type=int, default=1, help="Seed for the randomisation.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import os

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils import parse_env_cfg

import cyclotron.tasks  # noqa: F401
from cyclotron.assets.robots.asimov_1 import asimov_1_dc_motor_actuators

installed_version = metadata.version("rsl-rl-lib")
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli), installed_version)
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Only OnPolicyRunner policies can be evaluated, not {agent_cfg.class_name}.")
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = cli_args.resolve_checkpoint(log_root_path, agent_cfg, args_cli)
    if cli_args.apply_motion_file(env_cfg, args_cli, run_dir=os.path.dirname(checkpoint)) is None:
        raise ValueError(f"{args_cli.task} is not a motion tracking task.")

    env_cfg.seed = args_cli.seed
    env_cfg.commands.motion.start_from_first_frame = True
    env_cfg.episode_length_s = 1.0e4  # the motion's end, not a time-out, ends the evaluation
    if args_cli.condition == "nominal":
        for name in list(vars(env_cfg.events)):
            setattr(env_cfg.events, name, None)
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.commands.motion.pose_range = {}
        env_cfg.commands.motion.velocity_range = {}
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    elif args_cli.condition == "randomized":
        env_cfg.events.push_robot = None
    if args_cli.motor_speed_ratio is not None:
        env_cfg.scene.robot = env_cfg.scene.robot.replace(
            actuators=asimov_1_dc_motor_actuators(args_cli.motor_speed_ratio)
        )

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(handle_deprecated_rsl_rl_checkpoint(checkpoint, installed_version))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot, command, contacts = (
        base.scene["robot"],
        base.command_manager.get_term("motion"),
        base.scene["contact_forces"],
    )
    feet = contacts.find_bodies(FEET, preserve_order=True)[0]
    num_envs, device = base.num_envs, base.device
    effort_limit = torch.zeros(num_envs, robot.num_joints, device=device)
    for actuator in robot.actuators.values():
        effort_limit[:, actuator.joint_indices] = actuator.effort_limit
    peak_speed = torch.zeros(num_envs, robot.num_joints, device=device)
    peak_torque = torch.zeros_like(peak_speed)
    speed_loss = torch.zeros_like(peak_speed)  # torque the motor's speed limit took away, beyond the effort limit
    peak_force = torch.zeros(num_envs, device=device)
    failed_at = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    failed_by = [""] * num_envs

    # Peaks at every physics step, not just every policy step, so short impacts are seen.
    update_scene = base.scene.update

    def update_and_record(dt: float):
        update_scene(dt)
        alive = failed_at < 0
        peak_speed[alive] = torch.maximum(peak_speed[alive], robot.data.joint_vel[alive].abs())
        peak_torque[alive] = torch.maximum(peak_torque[alive], robot.data.applied_torque[alive].abs())
        lost = torch.minimum(robot.data.computed_torque.abs(), effort_limit) - robot.data.applied_torque.abs()
        speed_loss[alive] = torch.maximum(speed_loss[alive], lost[alive])
        force = contacts.data.net_forces_w[:, feet].norm(dim=-1).amax(dim=-1)
        peak_force[alive] = torch.maximum(peak_force[alive], force[alive])

    base.scene.update = update_and_record

    obs = env.get_observations()
    start_pos = robot.data.root_link_pos_w.clone()
    max_height = start_pos[:, 2].clone()
    reference = command.motion.root_pos_w
    forward = (reference[-1, :2] - reference[0, :2]) / torch.norm(reference[-1, :2] - reference[0, :2])
    terms = [name for name in base.termination_manager.active_terms if name != "time_out"]
    num_frames = command.motion.num_frames
    with torch.inference_mode():
        for step in range(1, num_frames):  # one more step would start the motion again
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            max_height = torch.maximum(max_height, robot.data.root_link_pos_w[:, 2])
            for i in torch.where(base.termination_manager.terminated & (failed_at < 0))[0].tolist():
                failed_at[i] = step
                failed_by[i] = "+".join(name for name in terms if bool(base.termination_manager.get_term(name)[i]))
    finished = (failed_at < 0).cpu().numpy()
    travel = ((robot.data.root_link_pos_w[:, :2] - start_pos[:, :2]) @ forward).cpu().numpy()
    rise = ((max_height - start_pos[:, 2]) * 100).cpu().numpy()

    fps = command.motion.fps
    print(
        f"\n[INFO] {checkpoint}\n[INFO] {args_cli.condition}"
        f"{f', motor speed ratio {args_cli.motor_speed_ratio:g}' if args_cli.motor_speed_ratio else ''}:"
        f" {num_envs} robots, {num_frames / fps:.2f} s motion"
    )
    print(f"Fell: {(~finished).sum()} of {num_envs} ({(~finished).mean() * 100:.1f}%)")
    if not finished.all():
        times = failed_at.cpu().numpy()[~finished] / fps
        print(f"  at {', '.join(f'{t:.2f}' for t in np.sort(times)[:20])} s{' ...' if len(times) > 20 else ''}")
        kinds = {}
        for i in np.where(~finished)[0]:
            kinds[failed_by[i]] = kinds.get(failed_by[i], 0) + 1
        print("  by " + ", ".join(f"{kind} {n}" for kind, n in sorted(kinds.items(), key=lambda item: -item[1])))
    if not finished.any():
        return

    def spread(values: np.ndarray, digits: int = 1) -> str:
        low, mid, high = np.percentile(values, [5, 50, 95])
        return f"{mid:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"

    print("Robots that finished, median (5th to 95th percentile):")
    print(f"  pelvis rise {spread(rise[finished])} cm, travel {spread(travel[finished], 2)} m")
    names = robot.joint_names
    speed_limit = robot.data.joint_vel_limits[0].cpu().numpy()
    effort = effort_limit[0].cpu().numpy()
    peaks = [("joint speed", peak_speed, speed_limit, "rad/s"), ("torque", peak_torque, effort, "N·m")]
    if args_cli.motor_speed_ratio is not None:
        peaks.append(("torque lost to speed", speed_loss, effort, "N·m"))
    for label, values, limit, unit in peaks:
        values = values.cpu().numpy()[finished]
        top = np.argsort(-np.median(values / limit, axis=0))[:4]
        print(
            f"  peak {label}: "
            + "; ".join(f"{names[j]} {np.median(values[:, j]):.1f} of {limit[j]:.1f} {unit}" for j in top)
        )
    weight = float(robot.data.default_mass[0].sum()) * 9.81
    force = peak_force.cpu().numpy()[finished]
    print(f"  peak foot force {spread(force / weight)} x body weight")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
