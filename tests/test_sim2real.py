from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cyclotron.sim2real import (
    FIRMWARE_JOINTS,
    MODEL_JOINTS,
    NECK_DAMPING,
    NECK_STIFFNESS,
    JointMap,
    TrackingController,
    VelocityController,
    ankle_from_motors,
    heading,
    load_joint_limits,
    motors_from_ankle,
    turn_to_heading,
)
from cyclotron.sim2sim import PolicyRun, anchor_orientation_6d

URDF = Path(__file__).resolve().parents[1] / "third_party" / "asimov-1" / "sim-model" / "urdf" / "asimov_1.urdf"
JOINTS = list(MODEL_JOINTS.values())


def quat_wxyz(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat()[[3, 0, 1, 2]]


def test_ankle_conversion_round_trips():
    pitch, roll = np.array([0.3, -0.35, 0.0]), np.array([0.05, -0.1, 0.1])
    np.testing.assert_allclose(ankle_from_motors(*motors_from_ankle(pitch, roll)), (pitch, roll), atol=1e-12)


def test_joint_map_round_trips_and_keeps_the_neck():
    joint_map = JointMap(JOINTS)
    joints = np.random.default_rng(0).uniform(-0.3, 0.3, len(JOINTS))
    motors = joint_map.to_motors(joints, neck=(0.1, -0.2))

    np.testing.assert_allclose(joint_map.to_joints(motors), joints, atol=1e-12)
    np.testing.assert_allclose(motors[[FIRMWARE_JOINTS.index("Neck_Yaw"), FIRMWARE_JOINTS.index("Neck_Pitch")]],
                               [0.1, -0.2])
    knee = JOINTS.index("right_knee_joint")
    assert motors[FIRMWARE_JOINTS.index("R_Knee")] == joints[knee]


def test_joint_map_follows_the_policys_joint_order():
    joints = ["right_ankle_roll_joint", "left_knee_joint", "right_ankle_pitch_joint"] + [
        name for name in JOINTS if name not in ("right_ankle_roll_joint", "left_knee_joint", "right_ankle_pitch_joint")
    ]
    joint_map = JointMap(joints)
    values = np.zeros(len(joints))
    values[:3] = (0.05, 0.4, 0.2)  # right ankle roll, left knee, right ankle pitch
    motors = joint_map.to_motors(values)

    assert motors[FIRMWARE_JOINTS.index("L_Knee")] == 0.4
    a, b = motors[FIRMWARE_JOINTS.index("R_Ankle_A")], motors[FIRMWARE_JOINTS.index("R_Ankle_B")]
    np.testing.assert_allclose(ankle_from_motors(a, b), (0.2, 0.05))


def test_joint_map_rejects_unknown_joints():
    with pytest.raises(ValueError, match="neck_pitch_joint"):
        JointMap(["neck_pitch_joint"])


def test_ankle_gains_are_motor_gains():
    # Menlo's ankle model: the joint gains are the motors' times K^2 / 2, so the firmware's 110 / 5 per motor are
    # 224.42 / 10.201 at the pitch and 35.2 / 1.6 at the roll.
    joint_map = JointMap(JOINTS)
    stiffness, damping = np.full(len(JOINTS), 150.0), np.full(len(JOINTS), 5.0)
    for joint, kp, kd in (("ankle_pitch", 224.42, 10.201), ("ankle_roll", 35.2, 1.6)):
        for side in ("left", "right"):
            stiffness[JOINTS.index(f"{side}_{joint}_joint")], damping[JOINTS.index(f"{side}_{joint}_joint")] = kp, kd
    kp, kd = joint_map.gains(stiffness, damping)

    for motor in ("L_Ankle_A", "L_Ankle_B", "R_Ankle_A", "R_Ankle_B"):
        assert kp[FIRMWARE_JOINTS.index(motor)] == pytest.approx(110.0, rel=1e-4)  # Menlo rounded 224.422
        assert kd[FIRMWARE_JOINTS.index(motor)] == pytest.approx(5.0)
    assert kp[FIRMWARE_JOINTS.index("L_Knee")] == 150.0
    back = joint_map.joint_gains(kp, kd)
    np.testing.assert_allclose(back[0], stiffness)
    np.testing.assert_allclose(back[1], damping)


def test_gains_cover_every_motor():
    joint_map = JointMap(JOINTS)
    kp, kd = joint_map.gains(np.arange(len(JOINTS), dtype=float), np.ones(len(JOINTS)))

    assert len(kp) == len(kd) == len(FIRMWARE_JOINTS)
    assert kp[FIRMWARE_JOINTS.index("Waist_Yaw")] == JOINTS.index("waist_yaw_joint")
    assert kp[FIRMWARE_JOINTS.index("Neck_Yaw")] == NECK_STIFFNESS and kd[FIRMWARE_JOINTS.index("Neck_Pitch")] == NECK_DAMPING
    assert not np.isnan(kp).any() and not np.isnan(kd).any()


def test_sdk_sends_the_ankle_pitch_and_roll_and_accepts_the_limits():
    robots = pytest.importorskip("menlo.asimov.robots")
    assert tuple(robots.ASIMOV_1_BIPED_JOINTS) == FIRMWARE_JOINTS
    joint_map = JointMap(JOINTS)
    joints = np.zeros(len(JOINTS))
    joints[JOINTS.index("left_ankle_pitch_joint")], joints[JOINTS.index("left_ankle_roll_joint")] = -0.35, 0.1
    motors = tuple(joint_map.to_motors(joints))

    robots.check_ankle_limits(motors)  # at the limits: accepted
    wire = robots.trajectory_wire_positions(motors)
    np.testing.assert_allclose(wire[FIRMWARE_JOINTS.index("L_Ankle_A")], -0.35)
    np.testing.assert_allclose(wire[FIRMWARE_JOINTS.index("L_Ankle_B")], 0.1)
    joints[JOINTS.index("left_ankle_pitch_joint")] = -0.4
    with pytest.raises(ValueError):
        robots.check_ankle_limits(tuple(joint_map.to_motors(joints)))


def test_joint_limits_come_from_the_urdf():
    lower, upper = load_joint_limits(URDF, ["left_knee_joint", "right_ankle_roll_joint"])
    np.testing.assert_allclose(lower, [0.0, -0.1])
    np.testing.assert_allclose(upper, [1.5, 0.1])


def test_turn_to_heading_keeps_tilt_and_relative_turns():
    quats = np.array([quat_wxyz(Rotation.from_euler("zyx", [0.3 + 0.1 * i, 0.2, -0.1])) for i in range(4)])
    turned = turn_to_heading(quats, yaw=-1.0)

    assert heading(turned[0]) == pytest.approx(-1.0)
    for original, new in zip(quats, turned):
        # Tilt (projected gravity) is unchanged, and so is each frame's orientation relative to the first.
        np.testing.assert_allclose(Rotation.from_quat(new[[1, 2, 3, 0]]).inv().apply([0, 0, -1]),
                                   Rotation.from_quat(original[[1, 2, 3, 0]]).inv().apply([0, 0, -1]), atol=1e-12)
        np.testing.assert_allclose(anchor_orientation_6d(turned[0], new), anchor_orientation_6d(quats[0], original),
                                   atol=1e-12)


@pytest.fixture
def run(tmp_path) -> PolicyRun:
    frames = 5
    rng = np.random.default_rng(1)
    pelvis_quat = np.array([quat_wxyz(Rotation.from_euler("z", 0.5 + 0.01 * i)) for i in range(frames)])
    np.savez(
        tmp_path / "motion.npz",
        fps=np.array(50.0),
        joint_names=np.array(JOINTS),
        joint_pos=rng.uniform(-0.1, 0.1, (frames, len(JOINTS))),
        joint_vel=rng.uniform(-1.0, 1.0, (frames, len(JOINTS))),
        body_pos_w=np.zeros((frames, 1, 3)),
        body_quat_w=pelvis_quat[:, None, :],
    )
    n = len(JOINTS)
    return PolicyRun(
        joint_names=JOINTS,
        default_joint_pos=np.full(n, 0.05),
        stiffness=np.full(n, 100.0),
        damping=np.full(n, 5.0),
        effort_limit=np.full(n, 40.0),
        armature=np.full(n, np.nan),
        actuator_groups=["joint"] * n,
        action_scale=np.full(n, 0.25),
        action_clip=np.tile([-np.inf, np.inf], (n, 1)),
        physics_dt=0.005,
        decimation=4,
        root_height=0.639,
        interface="tracking",
        observations=[],
        motion_file=str(tmp_path / "motion.npz"),
    )


def test_tracking_controller_builds_the_training_observation(run):
    controller = TrackingController(run, lambda obs: np.full(len(JOINTS), 1.0), (np.full(23, -1.0), np.full(23, 0.2)))
    n = len(JOINTS)
    robot_quat = quat_wxyz(Rotation.from_euler("z", -2.0))  # faces elsewhere: the reference is turned to it
    controller.start(robot_quat)
    first = controller.step(robot_quat, np.array([0.1, 0.2, 0.3]), np.full(n, 0.15), np.full(n, 0.5))

    obs = first.observation
    assert obs.shape == (4 * n + 6 + 3 + n,)
    np.testing.assert_allclose(obs[:n], controller.reference_q[0], atol=1e-6)
    np.testing.assert_allclose(obs[n : 2 * n], controller.reference_dq[0], atol=1e-6)
    np.testing.assert_allclose(obs[2 * n : 2 * n + 6], [1, 0, 0, 1, 0, 0], atol=1e-6)  # aligned at the start
    np.testing.assert_allclose(obs[2 * n + 6 : 2 * n + 9], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(obs[2 * n + 9 : 3 * n + 9], 0.1, atol=1e-6)  # relative to the default pose
    np.testing.assert_allclose(obs[3 * n + 9 : 4 * n + 9], 0.5, atol=1e-6)
    np.testing.assert_allclose(obs[4 * n + 9 :], 0.0)  # no action yet
    # 0.05 + 0.25 * 1.0 = 0.3 is past the upper limit of 0.2.
    np.testing.assert_allclose(first.target, 0.2)
    assert first.clamped.all()

    second = controller.step(robot_quat, np.zeros(3), np.zeros(n), np.zeros(n))
    np.testing.assert_allclose(second.observation[4 * n + 9 :], 1.0)  # the last action


def test_velocity_controller_steps_the_recipe_and_clips_the_targets(run):
    from cyclotron.sim2sim import ObservationTerm

    run.interface = "velocity"
    run.observations = [
        ObservationTerm("base_ang_vel", "base_ang_vel", 0.25),
        ObservationTerm("projected_gravity", "projected_gravity"),
        ObservationTerm("command", "command"),
        ObservationTerm("joint_vel_slot", "joint_vel", 0.1, [1, 0]),
        ObservationTerm("actions", "actions"),
    ]
    run.action_clip[0] = (-0.1, 0.1)
    seen = []
    controller = VelocityController(
        run, lambda obs: seen.append(obs) or np.ones(len(JOINTS)), (np.full(23, -1.0), np.full(23, 0.2))
    )
    n = len(JOINTS)
    joint_vel = np.arange(n, dtype=float)
    first = controller.step(np.array([0, 0, -1.0]), np.array([4.0, 0, 0]), np.zeros(n), joint_vel, np.array([0.3, 0, 0]))

    np.testing.assert_allclose(seen[0][:11], [1, 0, 0, 0, 0, -1, 0.3, 0, 0, 0.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(seen[0][11:], 0.0)  # no action yet
    # 0.05 + 0.25 is clipped by the run to 0.1 for the first joint, by the joint limits to 0.2 for the others.
    assert first.target[0] == pytest.approx(0.1) and np.allclose(first.target[1:], 0.2)
    second = controller.step(np.array([0, 0, -1.0]), np.zeros(3), np.zeros(n), joint_vel, np.zeros(3))
    np.testing.assert_allclose(second.observation[11:], 1.0)  # the last action
    with pytest.raises(ValueError):
        run.interface = "tracking"
        VelocityController(run, lambda obs: obs, (np.full(23, -1.0), np.full(23, 1.0)))


def test_tracking_controller_plays_once_then_holds_the_last_frame(run):
    controller = TrackingController(run, lambda obs: np.zeros(len(JOINTS)), (np.full(23, -1.0), np.full(23, 1.0)))
    controller.start(quat_wxyz(Rotation.identity()))
    n = len(JOINTS)
    for _ in range(controller.frames - 1):
        assert not controller.done
        controller.step(np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(n), np.zeros(n))
    assert controller.done
    held = controller.step(np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(n), np.zeros(n))
    np.testing.assert_allclose(held.observation[:n], controller.reference_q[-1], atol=1e-6)


def test_tracking_controller_settles_on_the_first_frame_standing_still(run):
    controller = TrackingController(run, lambda obs: np.zeros(len(JOINTS)), (np.full(23, -1.0), np.full(23, 1.0)))
    controller.start(quat_wxyz(Rotation.identity()), settle_steps=2)
    n = len(JOINTS)
    for _ in range(2):
        settling = controller.step(np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(n), np.zeros(n))
        np.testing.assert_allclose(settling.observation[:n], controller.reference_q[0], atol=1e-6)
        np.testing.assert_allclose(settling.observation[n : 2 * n], 0.0)  # standing still
    assert controller.frame == 0
    playing = controller.step(np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(n), np.zeros(n))
    np.testing.assert_allclose(playing.observation[n : 2 * n], controller.reference_dq[0], atol=1e-6)
    assert controller.frame == 1
