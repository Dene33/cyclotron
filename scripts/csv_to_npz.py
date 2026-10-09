# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a retargeted Asimov-1 motion CSV into a motion file (.npz) for motion tracking (and AMP) training.

The CSV is resampled to the policy rate and replayed kinematically on the Asimov-1 articulation used for training,
which gives the world pose and velocity of every body. The motion is then moved so that it starts above the origin
with its lowest foot contact on the ground, and a short feasibility report is printed: flight time and joint position
and velocity limits.

Adapted from ``scripts/csv_to_npz.py`` in whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking),
Copyright (c) 2024, The Isaac Lab Project Developers, MIT License.

.. code-block:: bash

    ./cyclotron.sh --csv-to-npz --input_file high_jump.csv --input_fps 120 --output_file motions/high_jump.npz
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert a retargeted Asimov-1 motion CSV into a motion file (.npz).")
parser.add_argument("--input_file", type=str, required=True, help="Robot motion CSV, e.g. from soma-retargeter.")
parser.add_argument(
    "--input_fps",
    type=float,
    default=120.0,
    help="Frame rate of the CSV. soma-retargeter keeps the BVH's rate (1 / 'Frame Time'), 120 for SOMA BVH files.",
)
parser.add_argument("--output_file", type=str, required=True, help="Motion file to write (.npz).")
parser.add_argument(
    "--output_fps", type=float, default=50.0, help="Output frame rate. Must match the policy rate (50 Hz for Asimov-1)."
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Keep only these CSV frames, 1-based and both inclusive.",
)
parser.add_argument(
    "--keep_xy",
    action="store_true",
    default=False,
    help="Keep the CSV's horizontal start position instead of moving the first frame above the origin.",
)
parser.add_argument(
    "--keep_z",
    action="store_true",
    default=False,
    help="Keep the CSV's heights instead of moving the motion so its lowest foot contact touches the ground.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# The replay needs no window.
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from cyclotron.assets.robots.asimov_1 import ASIMOV_1_DELAYED_CFG, ASIMOV_1_JOINT_NAMES, ASIMOV_1_URDF_PATH
from cyclotron.motion_csv import RobotMotion, angular_velocity, linear_velocity, load_robot_csv, resample

FEET_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# Both feet this far above the ground counts as a flight phase.
FLIGHT_CLEARANCE = 0.02


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ASIMOV_1_DELAYED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def foot_contact_spheres(urdf_path: str, link_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Centers (in the link frame) and radii of the collision spheres a foot touches the ground with."""
    link = ET.parse(urdf_path).getroot().find(f"link[@name='{link_name}']")
    centers, radii = [], []
    for collision in link.findall("collision") if link is not None else []:
        sphere = collision.find("geometry/sphere")
        if sphere is None:
            continue
        origin = collision.find("origin")
        centers.append([float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0.0] * 3)
        radii.append(float(sphere.get("radius")))
    if not centers:
        raise ValueError(f"{urdf_path} has no collision spheres on {link_name} to find its sole from.")
    return np.array(centers), np.array(radii)


def replay(motion: RobotMotion) -> dict:
    """Pose the robot at every frame without stepping physics and read back the world pose of every body."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / motion.fps, device=args_cli.device))
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    joint_ids = robot.find_joints(motion.joint_names, preserve_order=True)[0]
    origin = scene.env_origins[:1]

    root_pose = torch.tensor(np.concatenate([motion.root_pos, motion.root_quat], axis=1), device=sim.device)
    joint_pos = torch.tensor(motion.joint_pos, dtype=torch.float32, device=sim.device)
    all_joint_pos = robot.data.default_joint_pos.clone()
    body_pos_w, body_quat_w = [], []
    for t in range(motion.num_frames):
        pose = root_pose[t : t + 1].float()
        pose[:, :3] += origin
        all_joint_pos[:, joint_ids] = joint_pos[t : t + 1]
        robot.write_root_link_pose_to_sim(pose)
        robot.write_root_link_velocity_to_sim(torch.zeros(1, 6, device=sim.device))
        robot.write_joint_state_to_sim(all_joint_pos, torch.zeros_like(all_joint_pos))
        sim.forward()
        scene.update(sim.get_physics_dt())
        body_pos_w.append((robot.data.body_link_pos_w[0] - origin[0]).cpu().numpy())
        body_quat_w.append(robot.data.body_link_quat_w[0].cpu().numpy())

    # The root pose read back must be the one written, or the replay did not take effect.
    root_error = np.abs(np.stack(body_pos_w)[:, 0] - motion.root_pos).max()
    if root_error > 1.0e-3:
        raise RuntimeError(f"Replayed root position differs from the CSV by {root_error:.4f} m.")

    replayed = {
        "body_names": list(robot.body_names),
        "body_pos_w": np.stack(body_pos_w).astype(np.float64),
        "body_quat_w": np.stack(body_quat_w).astype(np.float64),
        "joint_pos_limits": robot.data.joint_pos_limits[0, joint_ids].cpu().numpy(),
        "joint_vel_limits": robot.data.joint_vel_limits[0, joint_ids].cpu().numpy(),
    }
    # Release the simulation as Isaac Lab environments do on close; otherwise closing the app hangs.
    del robot, scene
    sim.clear_all_callbacks()
    sim.clear_instance()
    return replayed


def sole_heights(replayed: dict) -> np.ndarray:
    """Height of each foot's lowest contact point at every frame, shaped (frames, feet)."""
    heights = []
    for name in FEET_BODIES:
        body = replayed["body_names"].index(name)
        centers, radii = foot_contact_spheres(ASIMOV_1_URDF_PATH, name)
        rotation = Rotation.from_quat(replayed["body_quat_w"][:, body][:, [1, 2, 3, 0]])
        points = replayed["body_pos_w"][:, body, None, :] + np.stack([rotation.apply(c) for c in centers], axis=1)
        heights.append((points[..., 2] - radii).min(axis=1))
    return np.stack(heights, axis=1)


def report(motion: RobotMotion, joint_vel: np.ndarray, replayed: dict, soles: np.ndarray) -> None:
    """Print what in the motion Asimov-1 may not be able to follow."""
    root_z = motion.root_pos[:, 2]
    print(f"[INFO] Root height: {root_z.min():.3f} to {root_z.max():.3f} m (first frame {root_z[0]:.3f} m)")
    flight = np.all(soles > FLIGHT_CLEARANCE, axis=1)
    if flight.any():
        print(f"[INFO] Both feet off the ground for {flight.sum()} frames ({flight.sum() / motion.fps:.2f} s)")

    lower, upper = replayed["joint_pos_limits"][:, 0], replayed["joint_pos_limits"][:, 1]
    peak_vel = np.abs(joint_vel).max(axis=0)
    for i, name in enumerate(motion.joint_names):
        low, high = motion.joint_pos[:, i].min(), motion.joint_pos[:, i].max()
        if low < lower[i] - 1.0e-3 or high > upper[i] + 1.0e-3:
            print(
                f"[WARN] {name} reaches [{low:+.2f}, {high:+.2f}] rad, outside its limits"
                f" [{lower[i]:+.2f}, {upper[i]:+.2f}]"
            )
        if peak_vel[i] > replayed["joint_vel_limits"][i]:
            print(
                f"[WARN] {name} peaks at {peak_vel[i]:.1f} rad/s, above its {replayed['joint_vel_limits'][i]:.1f}"
                " rad/s limit"
            )


def main():
    motion = load_robot_csv(
        args_cli.input_file,
        fps=args_cli.input_fps,
        joint_names=list(ASIMOV_1_JOINT_NAMES),
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range else None,
    )
    print(f"[INFO] Loaded {args_cli.input_file}: {motion.num_frames} frames at {motion.fps:g} fps")
    if not args_cli.keep_xy:
        motion.root_pos[:, :2] -= motion.root_pos[0, :2]
    motion = resample(motion, args_cli.output_fps)
    print(f"[INFO] Resampled to {motion.num_frames} frames at {motion.fps:g} fps ({motion.duration:.2f} s)")
    replayed = replay(motion)

    soles = sole_heights(replayed)
    lowest = soles.min(axis=1)
    if args_cli.keep_z:
        print(f"[INFO] Lowest foot contact is at {lowest.min():+.3f} m (kept, --keep_z)")
    else:
        if lowest.min() > 0.0:
            # The motion never touches the ground: lower all of it.
            shift = np.full_like(lowest, -lowest.min())
            print(f"[INFO] Lowered the motion {lowest.min():.3f} m so its feet touch the ground")
        else:
            # Lift the frames whose feet sink into the ground. Retargeting can sink them where the robot's joint
            # limits keep it from copying the human, e.g. ankles in a deep crouch.
            shift = np.maximum(-lowest, 0.0)
            sunk = shift > 1.0e-3
            if sunk.any():
                print(
                    f"[INFO] Lifted {sunk.sum()} of {motion.num_frames} frames out of the ground, by up to"
                    f" {shift.max():.3f} m"
                )
        motion.root_pos[:, 2] += shift
        replayed["body_pos_w"][..., 2] += shift[:, None]
        soles += shift[:, None]

    dt = 1.0 / motion.fps
    joint_vel = linear_velocity(motion.joint_pos, dt)
    output_file = os.path.abspath(os.path.expanduser(args_cli.output_file))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(
        output_file,
        fps=np.array([motion.fps]),
        joint_names=np.array(motion.joint_names),
        joint_pos=motion.joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_names=np.array(replayed["body_names"]),
        body_pos_w=replayed["body_pos_w"].astype(np.float32),
        body_quat_w=replayed["body_quat_w"].astype(np.float32),
        body_lin_vel_w=linear_velocity(replayed["body_pos_w"], dt).astype(np.float32),
        body_ang_vel_w=angular_velocity(replayed["body_quat_w"], dt).astype(np.float32),
    )
    report(motion, joint_vel, replayed, soles)
    print(f"[INFO] Saved {output_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
