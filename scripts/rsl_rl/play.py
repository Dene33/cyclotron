# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL evaluation."""


import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--target", type=str, default=None, help="Direct path to the checkpoint file to play (alias for --checkpoint)."
)
parser.add_argument(
    "--strict", action="store_true", help="Stop, instead of warning, if the code changed since the run was trained."
)
parser.add_argument(
    "--record",
    type=str,
    default=None,
    help="Record the first robot's root pose and joint positions to this .npz, for ./cyclotron.sh --render, then exit.",
)
parser.add_argument(
    "--record_steps",
    type=int,
    default=None,
    help="Steps to record. Default: one pass of the reference motion for motion tracking tasks (one cycle of standing,"
    " jumping and holding the landing for the jump), else one episode.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Unknown flags go to Hydra, which would only reject these after Isaac Sim has started.
REMOVED_EXPORT_FLAGS = (
    "--export-only",
    "--export_only",
    "--onnx-output",
    "--onnx_output",
    "--onnx-filename",
    "--onnx_filename",
)
removed = [arg.split("=")[0] for arg in hydra_args if arg.split("=")[0] in REMOVED_EXPORT_FLAGS]
if removed:
    sys.exit(f"[ERROR] {removed[0]} was removed from --play, which no longer writes ONNX. Use ./cyclotron.sh --export.")
if args_cli.target and not args_cli.checkpoint:
    args_cli.checkpoint = args_cli.target
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")


import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import class_to_dict, print_dict
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils.hydra import hydra_task_config

import cyclotron.tasks  # noqa: F401
from cyclotron.code_state import check_out_hint, describe_changes
from cyclotron.policy_io import resolve_policy_io
from cyclotron.policy_loading import load_policy

CODE_CHANGE_CONSEQUENCE = (
    "--play runs the checkpoint in an environment built from the current code, so it may behave differently than in"
    " training."
)


class RolloutRecorder:
    """Records the first robot's root pose and joint positions, for ./cyclotron.sh --render."""

    def __init__(self, env, steps: int | None):
        self.robot = env.scene["robot"]
        self.origin = env.scene.env_origins[0]
        commands = env.command_manager.active_terms
        self.motion = env.command_manager.get_term("motion") if "motion" in commands else None
        if steps is None and self.motion is not None:
            # One pass of the reference motion, or for the jump one cycle: standing, the jump, holding the landing.
            steps = getattr(self.motion, "cycle_steps", self.motion.motion.num_frames - 1)
        elif steps is None:
            steps = int(env.max_episode_length)
        self.steps = steps
        self.fps = 1.0 / env.step_dt
        self.data = {"frame": [], "root_pos": [], "root_quat": [], "joint_pos": []}

    @property
    def done(self) -> bool:
        return len(self.data["frame"]) > self.steps

    def log(self) -> None:
        # For motion tracking tasks, the reference frame the robot is following; otherwise the step.
        if self.motion is not None:
            frame = getattr(self.motion, "shown_frames", self.motion.time_steps)[0].item()
        else:
            frame = len(self.data["frame"])
        self.data["frame"].append(frame)
        # Copies: on a CPU device .cpu().numpy() shares memory with simulation buffers that are updated in place.
        self.data["root_pos"].append((self.robot.data.root_link_pos_w[0] - self.origin).cpu().numpy().copy())
        self.data["root_quat"].append(self.robot.data.root_link_quat_w[0].cpu().numpy().copy())
        self.data["joint_pos"].append(self.robot.data.joint_pos[0].cpu().numpy().copy())

    def save(self, path: str) -> None:
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {key: np.array(values) for key, values in self.data.items()}
        np.savez(path, fps=np.array(self.fps), joint_names=np.array(self.robot.joint_names), **data)
        print(f"[INFO] Recorded {len(self.data['frame'])} frames to {path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    else:
        resume_path = cli_args.resolve_checkpoint(log_root_path, agent_cfg, args_cli)

    log_dir = os.path.dirname(resume_path)

    # Motion tracking tasks: the reference motion, from --motion_file or else the one the policy was trained on.
    cli_args.apply_motion_file(env_cfg, args_cli, run_dir=log_dir)

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Compared where train.py saves params/: after the environment is created, which resolves parts of the config.
    agent_dict = class_to_dict(agent_cfg)
    changes, level = describe_changes(
        log_dir,
        class_to_dict(env_cfg),
        agent_dict,
        CODE_CHANGE_CONSEQUENCE,
        policy_io=resolve_policy_io(env.unwrapped, agent_dict),
    )
    print(changes)
    if level == "error":
        env.close()
        sys.exit(1)
    if level == "warning" and args_cli.strict:
        # Isaac Sim replaces sys.exit with a version that only takes an exit code, so the message is printed first.
        print("[ERROR] Stopped by --strict: the code changed since the run was trained (see above).")
        env.close()
        sys.exit(1)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    try:
        load_policy(runner, resume_path, agent_cfg.class_name, check_out_hint(log_dir))
    except ValueError as error:
        print(f"[ERROR] {error}")
        env.close()
        sys.exit(1)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Older RSL-RL versions reset the policy network itself between episodes.
    if version.parse(installed_version) < version.parse("4.0.0"):
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    recorder = RolloutRecorder(env.unwrapped, args_cli.record_steps) if args_cli.record else None
    obs = env.get_observations()
    timestep = 0
    if recorder is not None:
        recorder.log()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        if recorder is not None:
            recorder.log()
            if recorder.done:
                recorder.save(args_cli.record)
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
