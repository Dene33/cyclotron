# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure how reliably a motion tracking policy plays its whole motion.

Many robots start on the reference's first frame and play it to the end under the training task's randomisation. For
the jump (``Asimov1-Jump-v0``), they start standing in the default pose, are told to jump after 1.5 s and hold their
landing for 3 s.
Reports the share that falls (any termination before the end) and when, how high and far the robots that finish
get, their peak foot force, and for every motor its torque, speed and power: peaks against the limits, how long it
spends at them, RMS torque and positive work. ``--save`` also writes these per robot, with physics-rate joint traces
of the first robots, for analysis outside Isaac Sim.
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
parser.add_argument(
    "--save",
    type=str,
    default=None,
    help="Also save the results per robot, and physics-rate joint traces of the first --save_robots robots (.npz).",
)
parser.add_argument("--save_robots", type=int, default=16, help="How many robots' joint traces --save keeps.")
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
# A motor is torque limited when it applies this much less torque (N·m) than the policy's command needs, and speed
# limited at this share of its speed limit or faster.
TORQUE_TOLERANCE = 1.0e-3
SPEED_LIMIT_SHARE = 0.98
# Joint quantities --save records at every physics step.
TRACES = ("joint_pos", "joint_vel", "applied_torque", "computed_torque")


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
    motion_cfg = env_cfg.commands.motion
    jump = hasattr(motion_cfg, "stand_start_prob")  # Asimov1-Jump: the reference plays when the robot is told to jump
    if jump:
        # One jump, as on the robot: standing in the default pose for 1.5 s, then the command held through the jump
        # and released 0.4 s after landing, then holding the landing for 3 s.
        motion_cfg.stand_start_prob = 1.0
        motion_cfg.default_pose_prob = 1.0
        motion_cfg.stand_time_s = (1.5, 1.5)
        motion_cfg.hold_time_s = (3.0, 3.0)
        motion_cfg.release_after_landing_s = (0.4, 0.4)
        motion_cfg.jump_again_prob = 0.0
        motion_cfg.trigger_speed = (0.3, 0.3)
    else:
        motion_cfg.start_from_first_frame = True
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
            actuators=asimov_1_dc_motor_actuators(args_cli.motor_speed_ratio, env_cfg.scene.robot.actuators)
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
    num_envs, num_joints, device = base.num_envs, robot.num_joints, base.device
    effort_limit = torch.zeros(num_envs, num_joints, device=device)
    for actuator in robot.actuators.values():
        effort_limit[:, actuator.joint_indices] = actuator.effort_limit
    speed_limit = robot.data.joint_vel_limits.clone()
    # Per robot and motor, over the motion until the robot falls.
    motors = {
        name: torch.zeros(num_envs, num_joints, device=device)
        for name in (
            "peak_torque",
            "peak_speed",
            "peak_power",
            "torque_limited_time",  # s the motor gave less torque than the policy's command needed
            "speed_limited_time",  # s at SPEED_LIMIT_SHARE of the speed limit or faster
            "torque_squared_time",  # integral of torque squared over time, for the RMS torque
            "positive_work",  # J the motor put in
            "speed_loss",  # torque the motor's speed limit took away, beyond the effort limit
        )
    }
    elapsed = torch.zeros(num_envs, device=device)
    peak_force = torch.zeros(num_envs, device=device)
    failed_at = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    failed_by = [""] * num_envs
    traced = min(args_cli.save_robots, num_envs)
    traces = {key: [] for key in TRACES} if args_cli.save else None

    # At every physics step, not just every policy step, so short impacts and saturations are seen.
    update_scene = base.scene.update

    def update_and_record(dt: float):
        update_scene(dt)
        data = robot.data
        alive = (failed_at < 0).unsqueeze(1)
        torque, speed, power = data.applied_torque.abs(), data.joint_vel.abs(), data.applied_torque * data.joint_vel
        peaks = {
            "peak_torque": torque,
            "peak_speed": speed,
            "peak_power": power.abs(),
            "speed_loss": torch.minimum(data.computed_torque.abs(), effort_limit) - torque,
        }
        for name, value in peaks.items():
            motors[name] = torch.where(alive, torch.maximum(motors[name], value), motors[name])
        motors["torque_limited_time"] += (alive & (data.computed_torque.abs() - torque > TORQUE_TOLERANCE)) * dt
        motors["speed_limited_time"] += (alive & (speed >= SPEED_LIMIT_SHARE * speed_limit)) * dt
        motors["torque_squared_time"] += alive * torque.square() * dt
        motors["positive_work"] += alive * power.clamp(min=0.0) * dt
        elapsed.add_(alive[:, 0] * dt)
        force = contacts.data.net_forces_w[:, feet].norm(dim=-1).amax(dim=-1)
        peak_force[alive[:, 0]] = torch.maximum(peak_force[alive[:, 0]], force[alive[:, 0]])
        if traces is not None:
            for key, values in traces.items():
                values.append(getattr(data, key)[:traced].clone())

    base.scene.update = update_and_record

    obs = env.get_observations()
    start_pos = robot.data.root_link_pos_w.clone()
    max_height = start_pos[:, 2].clone()
    reference = command.motion.root_pos_w
    forward = (reference[-1, :2] - reference[0, :2]) / torch.norm(reference[-1, :2] - reference[0, :2])
    terms = [name for name in base.termination_manager.active_terms if name != "time_out"]
    # One more step would start the motion (or the jump's cycle) again.
    num_steps = command.cycle_steps if jump else command.motion.num_frames - 1
    with torch.inference_mode():
        for step in range(1, num_steps + 1):
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            max_height = torch.maximum(max_height, robot.data.root_link_pos_w[:, 2])
            for i in torch.where(base.termination_manager.terminated & (failed_at < 0))[0].tolist():
                failed_at[i] = step
                failed_by[i] = "+".join(name for name in terms if bool(base.termination_manager.get_term(name)[i]))
    finished = (failed_at < 0).cpu().numpy()
    travel = ((robot.data.root_link_pos_w[:, :2] - start_pos[:, :2]) @ forward).cpu().numpy()
    rise = (max_height - start_pos[:, 2]).cpu().numpy()
    weight = float(robot.data.default_mass[0].sum()) * 9.81
    results = {name: value.cpu().numpy() for name, value in motors.items()}
    duration = elapsed.cpu().numpy()
    results["rms_torque"] = np.sqrt(results.pop("torque_squared_time") / np.maximum(duration, 1.0e-9)[:, None])
    names = robot.joint_names
    effort, speed = effort_limit[0].cpu().numpy(), speed_limit[0].cpu().numpy()
    if args_cli.save:
        info = {
            "checkpoint": checkpoint,
            "motion_file": env_cfg.commands.motion.motion_file,
            "condition": args_cli.condition,
            "motor_speed_ratio": np.nan if args_cli.motor_speed_ratio is None else args_cli.motor_speed_ratio,
            "seed": args_cli.seed,
            "physics_dt": base.physics_dt,
            "decimation": env_cfg.decimation,
            "joint_names": np.array(names),
            "effort_limit": effort,
            "speed_limit": speed,
        }
        per_robot = {
            "failed_at": failed_at.cpu().numpy(),  # policy step of the fall, -1 if the robot finished
            "finished": finished,
            "pelvis_rise": rise,  # m
            "travel": travel,  # m, along the reference's direction of travel
            "peak_foot_force": (peak_force / weight).cpu().numpy(),  # x body weight
            "duration": duration,  # s the motors were measured
            **results,
        }
        save(args_cli.save, info, per_robot, traces)
    fps = command.motion.fps
    order = list(command.joint_ids) + [j for j in range(num_joints) if j not in command.joint_ids]  # policy order
    # Everything is measured. Closing the environment before the report means no early return can leave Isaac Sim
    # hanging when it closes.
    env.close()

    print(
        f"\n[INFO] {checkpoint}\n[INFO] {args_cli.condition}"
        f"{f', motor speed ratio {args_cli.motor_speed_ratio:g}' if args_cli.motor_speed_ratio else ''}:"
        f" {num_envs} robots, {num_steps / fps:.2f} s"
        f" {'of standing, one jump and holding the landing' if jump else 'motion'}"
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
    print(f"  pelvis rise {spread(rise[finished] * 100)} cm, travel {spread(travel[finished], 2)} m")
    force = peak_force.cpu().numpy()[finished]
    print(f"  peak foot force {spread(force / weight)} x body weight")

    # One row per motor, in the policy's joint order.
    median = {name: np.median(value[finished], axis=0) for name, value in results.items()}
    width = max(len(name) for name in names)
    dc = args_cli.motor_speed_ratio is not None
    print(
        f"Motors of the robots that finished, median. Limited: torque below what the command needs, or speed at"
        f" {SPEED_LIMIT_SHARE:.0%} of the limit or more."
    )
    print(
        f"  {'':{width}}  {'torque N·m':^13}  {'limited':>7}  {'RMS':>5}  {'speed rad/s':^13}  {'limited':>7}"
        f"  {'power W':>7}  {'work J':>6}" + ("  lost to speed N·m" if dc else "")
    )
    print(
        f"  {'motor':{width}}  {'peak / limit':^13}  {'':>7}  {'':>5}  {'peak / limit':^13}  {'':>7}  {'peak':>7}"
        f"  {'+':>6}"
    )
    for j in order:
        row = (
            f"  {names[j]:{width}}  {median['peak_torque'][j]:5.1f} / {effort[j]:5.1f}"
            f"  {median['torque_limited_time'][j] * 1000:4.0f} ms  {median['rms_torque'][j]:5.1f}"
            f"  {median['peak_speed'][j]:5.1f} / {speed[j]:5.1f}  {median['speed_limited_time'][j] * 1000:4.0f} ms"
            f"  {median['peak_power'][j]:7.0f}  {median['positive_work'][j]:6.1f}"
        )
        if dc:
            row += f"  {median['speed_loss'][j]:18.1f}"
        print(row)


def save(path: str, info: dict, per_robot: dict, traces: dict[str, list[torch.Tensor]]) -> None:
    """The evaluation's settings, its results per robot, and the first robots' joint traces at every physics step,
    shaped (robots, steps, joints) and named ``trace_<quantity>``.

    A robot's trace is its own motion only up to physics step ``failed_at * decimation``: after a fall the robot is
    reset onto the motion's first frame.
    """
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trace_arrays = {f"trace_{key}": torch.stack(values, dim=1).cpu().numpy() for key, values in traces.items()}
    np.savez(path, **info, **per_robot, **trace_arrays)
    print(f"[INFO] Saved {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
