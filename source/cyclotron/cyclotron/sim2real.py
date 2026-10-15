# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Play a policy on the real Asimov 1 (sim2real), through Menlo's Python SDK.

The robot's Asimov Edge reports and takes the 25 motors in the firmware's order, in the robot model's (URDF's) signs,
except the ankles: each is a parallel linkage, reported as its two motors (A, B). menlo-sdk takes trajectory targets in
the same frame it reports and sends the firmware each ankle's pitch and roll. The neck is locked in the 23-joint model
the policies are trained on. This module converts between the two and steps the policy, a motion tracking one
(``TrackingController``) or one with the walking policy's interface (``VelocityController``); it imports neither Isaac
Lab nor menlo-sdk, so it can be tested anywhere. ``scripts/sim2real.py`` connects it to a robot.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cyclotron.sim2sim import PolicyRun, load_reference, tracking_observation, velocity_observation

# menlo-sdk's ``menlo.asimov.robots.ASIMOV_1_BIPED_JOINTS``: the firmware's motor table, in its order.
FIRMWARE_JOINTS = (
    "L_Hip_Pitch",
    "L_Hip_Roll",
    "L_Hip_Yaw",
    "L_Knee",
    "L_Ankle_A",
    "L_Ankle_B",
    "R_Hip_Pitch",
    "R_Hip_Roll",
    "R_Hip_Yaw",
    "R_Knee",
    "R_Ankle_A",
    "R_Ankle_B",
    "L_Shoulder_Pitch",
    "L_Shoulder_Roll",
    "L_Shoulder_Yaw",
    "L_Elbow",
    "L_Wrist_Yaw",
    "R_Shoulder_Pitch",
    "R_Shoulder_Roll",
    "R_Shoulder_Yaw",
    "R_Elbow",
    "R_Wrist_Yaw",
    "Waist_Yaw",
    "Neck_Yaw",
    "Neck_Pitch",
)
NECK_JOINTS = ("Neck_Yaw", "Neck_Pitch")

# The robot model's joint behind every other motor. An ankle's motor A stands for its pitch and motor B for its roll.
MODEL_JOINTS = {
    "L_Hip_Pitch": "left_hip_pitch_joint",
    "L_Hip_Roll": "left_hip_roll_joint",
    "L_Hip_Yaw": "left_hip_yaw_joint",
    "L_Knee": "left_knee_joint",
    "L_Ankle_A": "left_ankle_pitch_joint",
    "L_Ankle_B": "left_ankle_roll_joint",
    "R_Hip_Pitch": "right_hip_pitch_joint",
    "R_Hip_Roll": "right_hip_roll_joint",
    "R_Hip_Yaw": "right_hip_yaw_joint",
    "R_Knee": "right_knee_joint",
    "R_Ankle_A": "right_ankle_pitch_joint",
    "R_Ankle_B": "right_ankle_roll_joint",
    "L_Shoulder_Pitch": "left_shoulder_pitch_joint",
    "L_Shoulder_Roll": "left_shoulder_roll_joint",
    "L_Shoulder_Yaw": "left_shoulder_yaw_joint",
    "L_Elbow": "left_elbow_joint",
    "L_Wrist_Yaw": "left_wrist_yaw_joint",
    "R_Shoulder_Pitch": "right_shoulder_pitch_joint",
    "R_Shoulder_Roll": "right_shoulder_roll_joint",
    "R_Shoulder_Yaw": "right_shoulder_yaw_joint",
    "R_Elbow": "right_elbow_joint",
    "R_Wrist_Yaw": "right_wrist_yaw_joint",
    "Waist_Yaw": "waist_yaw_joint",
}
ANKLE_MOTORS = (("L_Ankle_A", "L_Ankle_B"), ("R_Ankle_A", "R_Ankle_B"))

# The ankle linkage, from menlo-sdk's ``menlo.asimov.robots`` (the firmware's ``apply_ankle_coupling``):
# A = K_PITCH * pitch - K_ROLL * roll and B = -K_PITCH * pitch - K_ROLL * roll.
ANKLE_K_PITCH = 2.02
ANKLE_K_ROLL = 0.8

# The neck uses the elbow and wrist actuator (EC-A4310), so it is held with their gains.
NECK_STIFFNESS = 40.0
NECK_DAMPING = 2.0

# The firmware drives each ankle motor with its own PD gains. Menlo's ankle model (cyclotron's ASIMOV_1_ACTUATORS since
# the 2026-09-24 walking policy) gives the ankle joint the motors' gains times K^2 / 2: the 110 / 5 the firmware uses
# are 224.42 / 10.201 at the pitch (K = 2.02) and 35.2 / 1.6 at the roll (K = 0.8). So a trajectory's ankle entries
# carry motor gains: the joint's divided by K^2 / 2.
ANKLE_GAIN_FACTORS = (ANKLE_K_PITCH**2 / 2, ANKLE_K_ROLL**2 / 2)
# The gains Asimov Edge accepts in a trajectory.
EDGE_KP_RANGE = (0.0, 500.0)
EDGE_KD_RANGE = (0.0, 5.0)


def ankle_from_motors(motor_a, motor_b):
    """Ankle (pitch, roll) from its motors (A, B). Linear, so it converts velocities too."""
    return (motor_a - motor_b) / (2 * ANKLE_K_PITCH), -(motor_a + motor_b) / (2 * ANKLE_K_ROLL)


def motors_from_ankle(pitch, roll):
    """Ankle motors (A, B) from its (pitch, roll)."""
    return ANKLE_K_PITCH * pitch - ANKLE_K_ROLL * roll, -ANKLE_K_PITCH * pitch - ANKLE_K_ROLL * roll


class JointMap:
    """Converts between the policy's joints (robot model order and signs) and the firmware's motors."""

    def __init__(self, joint_names: Sequence[str]):
        self.joint_names = list(joint_names)
        model_to_motor = {joint: motor for motor, joint in MODEL_JOINTS.items()}
        missing = [joint for joint in self.joint_names if joint not in model_to_motor]
        if missing:
            raise ValueError(f"No motor for joints {missing}.")
        self.motor_index = np.array([FIRMWARE_JOINTS.index(model_to_motor[joint]) for joint in self.joint_names])
        self.ankles = [
            (FIRMWARE_JOINTS.index(a), FIRMWARE_JOINTS.index(b), self.joint_names.index(MODEL_JOINTS[a]),
             self.joint_names.index(MODEL_JOINTS[b]))
            for a, b in ANKLE_MOTORS
        ]
        self.neck_index = np.array([FIRMWARE_JOINTS.index(motor) for motor in NECK_JOINTS])

    def to_joints(self, motors: Sequence[float]) -> np.ndarray:
        """Joint positions or velocities, in the policy's order, from the 25 motors' as Asimov Edge reports them."""
        motors = np.asarray(motors, dtype=np.float64)
        joints = motors[self.motor_index].copy()
        for a, b, pitch, roll in self.ankles:
            joints[pitch], joints[roll] = ankle_from_motors(motors[a], motors[b])
        return joints

    def to_motors(self, joints: Sequence[float], neck: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
        """The 25 motors' values, as menlo-sdk takes trajectory targets, from the policy's joints and the neck's."""
        joints = np.asarray(joints, dtype=np.float64)
        motors = np.zeros(len(FIRMWARE_JOINTS))
        motors[self.motor_index] = joints
        for a, b, pitch, roll in self.ankles:
            motors[a], motors[b] = motors_from_ankle(joints[pitch], joints[roll])
        motors[self.neck_index] = neck
        return motors

    def gains(self, stiffness: Sequence[float], damping: Sequence[float]) -> tuple[list[float], list[float]]:
        """Per-motor PD gains for a trajectory, from the policy's per-joint gains. An ankle's motor A gets its pitch
        gains and motor B its roll gains, as motor gains (``ANKLE_GAIN_FACTORS``); they are equal for a policy
        trained with Menlo's ankle model."""
        kp = np.full(len(FIRMWARE_JOINTS), np.nan)
        kd = np.full(len(FIRMWARE_JOINTS), np.nan)
        kp[self.motor_index], kd[self.motor_index] = stiffness, damping
        for a, b, pitch, roll in self.ankles:
            for motor, joint, factor in ((a, pitch, ANKLE_GAIN_FACTORS[0]), (b, roll, ANKLE_GAIN_FACTORS[1])):
                kp[motor], kd[motor] = stiffness[joint] / factor, damping[joint] / factor
        kp[self.neck_index], kd[self.neck_index] = NECK_STIFFNESS, NECK_DAMPING
        return kp.tolist(), kd.tolist()

    def joint_gains(self, kp: Sequence[float], kd: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """The policy's per-joint gains from a trajectory's per-motor gains: the inverse of :meth:`gains`."""
        kp, kd = np.asarray(kp, dtype=np.float64), np.asarray(kd, dtype=np.float64)
        stiffness, damping = kp[self.motor_index].copy(), kd[self.motor_index].copy()
        for a, b, pitch, roll in self.ankles:
            for motor, joint, factor in ((a, pitch, ANKLE_GAIN_FACTORS[0]), (b, roll, ANKLE_GAIN_FACTORS[1])):
                stiffness[joint], damping[joint] = kp[motor] * factor, kd[motor] * factor
        return stiffness, damping


def load_joint_limits(urdf_path: str | Path, joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper position limits of ``joint_names`` from the robot's URDF."""
    limits = {
        joint.get("name"): (float(joint.find("limit").get("lower")), float(joint.find("limit").get("upper")))
        for joint in ET.parse(urdf_path).getroot().iter("joint")
        if joint.find("limit") is not None
    }
    missing = [name for name in joint_names if name not in limits]
    if missing:
        raise ValueError(f"No limits for joints {missing} in {urdf_path}.")
    lower, upper = zip(*(limits[name] for name in joint_names))
    return np.array(lower), np.array(upper)


def heading(quat: np.ndarray) -> float:
    """Yaw of a body with orientation ``quat`` (w, x, y, z): where its x axis points, seen from above."""
    x_axis = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).apply([1.0, 0.0, 0.0])
    return float(np.arctan2(x_axis[1], x_axis[0]))


def turn_to_heading(quats: np.ndarray, yaw: float) -> np.ndarray:
    """``quats`` (N, 4; w, x, y, z) turned about the world's z axis so that the first faces ``yaw``."""
    turn = Rotation.from_euler("z", yaw - heading(quats[0]))
    turned = (turn * Rotation.from_quat(quats[:, [1, 2, 3, 0]])).as_quat()
    return turned[:, [3, 0, 1, 2]]


@dataclass
class Step:
    """One policy step: what the policy saw and did, and the joint targets after clamping to the joint limits."""

    observation: np.ndarray
    action: np.ndarray
    target: np.ndarray
    clamped: np.ndarray


class TrackingController:
    """Steps a motion tracking policy on what the robot measures, one policy step at a time.

    The reference is turned to the robot's heading when the controller starts: the IMU's yaw zero is wherever it was
    when the firmware started, while the policy was trained starting on the reference. The controller can first track
    the reference's first frame, standing still, for a few steps (``settle_steps``), so the policy balances the robot
    into its pose before the motion starts. After the reference's last frame, it keeps tracking that frame.
    """

    def __init__(
        self,
        run: PolicyRun,
        policy: Callable[[np.ndarray], np.ndarray],
        joint_limits: tuple[np.ndarray, np.ndarray],
    ):
        self.run = run
        self.policy = policy
        self.lower, self.upper = joint_limits
        reference = load_reference(run)
        self.reference_q, self.reference_dq = reference["joint_pos"], reference["joint_vel"]
        self.reference_quat = reference["pelvis_quat"]
        self.frames = len(self.reference_q)
        self.dt = run.physics_dt * run.decimation
        self.start(self.reference_quat[0])

    def start(self, robot_quat: np.ndarray, settle_steps: int = 0) -> None:
        """Start the reference from its first frame, facing the way the robot does, after ``settle_steps`` steps
        on that frame standing still."""
        self.aligned_quat = turn_to_heading(self.reference_quat, heading(robot_quat))
        self.frame = 0
        self.settle_steps = settle_steps
        self.action = np.zeros(len(self.run.joint_names), dtype=np.float32)

    @property
    def done(self) -> bool:
        """Whether the reference has been played once (as ``sim2sim.play`` does: one step per frame but the last)."""
        return self.frame >= self.frames - 1

    def first_pose(self) -> np.ndarray:
        """The reference's first joint positions, within the joint limits: where the robot should start."""
        return np.clip(self.reference_q[0], self.lower, self.upper)

    def step(self, quat: np.ndarray, ang_vel: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray) -> Step:
        """The next joint targets, from the pelvis orientation (w, x, y, z) and angular velocity (pelvis frame), and
        the joint positions and velocities, in the policy's order."""
        frame = min(self.frame, self.frames - 1)
        settling = self.settle_steps > 0
        observation = tracking_observation(
            self.reference_q[frame],
            np.zeros_like(self.reference_dq[frame]) if settling else self.reference_dq[frame],
            self.aligned_quat[frame],
            np.asarray(quat, dtype=np.float64),
            np.asarray(ang_vel, dtype=np.float64),
            np.asarray(joint_pos) - self.run.default_joint_pos,
            np.asarray(joint_vel),
            self.action,
        )
        self.action = np.asarray(self.policy(observation), dtype=np.float32)
        target = self.run.joint_targets(self.action)
        clamped_target = np.clip(target, self.lower, self.upper)
        if settling:
            self.settle_steps -= 1
        else:
            self.frame += 1
        return Step(observation, self.action, clamped_target, clamped_target != target)


class VelocityController:
    """Steps a policy with the walking policy's interface (``Asimov1-Velocity``, ``Asimov1-Jump``) on what the robot
    measures and a velocity command, one policy step at a time. The observation is built as the run's env.yaml
    records; the targets are the run's (scaled and clipped as in training), within the joint limits."""

    def __init__(
        self,
        run: PolicyRun,
        policy: Callable[[np.ndarray], np.ndarray],
        joint_limits: tuple[np.ndarray, np.ndarray],
    ):
        if run.interface != "velocity":
            raise ValueError(f"Expected a policy with the walking policy's interface, got a {run.interface} policy.")
        self.run = run
        self.policy = policy
        self.lower, self.upper = joint_limits
        self.dt = run.physics_dt * run.decimation
        self.start()

    def start(self) -> None:
        self.action = np.zeros(len(self.run.joint_names), dtype=np.float32)

    def step(
        self,
        gravity: np.ndarray,
        ang_vel: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        command: np.ndarray,
    ) -> Step:
        """The next joint targets, from gravity's direction and the angular velocity (pelvis frame), the joint
        positions and velocities (the policy's order) and the velocity command (forward, sideways, yaw rate)."""
        observation = velocity_observation(self.run, ang_vel, gravity, command, joint_pos, joint_vel, self.action)
        self.action = np.asarray(self.policy(observation), dtype=np.float32)
        target = self.run.joint_targets(self.action)
        clamped_target = np.clip(target, self.lower, self.upper)
        return Step(observation, self.action, clamped_target, clamped_target != target)
