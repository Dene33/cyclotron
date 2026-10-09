from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cyclotron.motion_csv import SOMA_ROOT_COLUMNS, angular_velocity, linear_velocity, load_robot_csv, resample

JOINTS = ["hip_joint", "knee_joint", "ankle_joint"]


def _write_soma_csv(path, rows, joint_columns):
    lines = [",".join(SOMA_ROOT_COLUMNS + joint_columns)]
    lines += [",".join(str(v) for v in [i, *row]) for i, row in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n")


def test_soma_csv_converts_units_and_reorders_joints(tmp_path):
    # Joint columns in a different order, plus one the robot does not drive.
    columns = ["knee_joint", "neck_joint", "ankle_joint", "hip_joint"]
    euler = [10.0, -5.0, 90.0]
    rows = [[100.0, 200.0, 62.0, *euler, 30.0, 99.0, -45.0, 90.0]] * 2
    csv = tmp_path / "motion.csv"
    _write_soma_csv(csv, rows, columns)

    motion = load_robot_csv(str(csv), fps=120.0, joint_names=JOINTS)

    np.testing.assert_allclose(motion.root_pos[0], [1.0, 2.0, 0.62])
    np.testing.assert_allclose(motion.joint_pos[0], np.deg2rad([90.0, 30.0, -45.0]))
    assert motion.joint_names == JOINTS
    # soma-retargeter writes Rotation.from_quat(q).as_euler("xyz", degrees=True); reading must invert it.
    quat_xyzw = motion.root_quat[0][[1, 2, 3, 0]]
    np.testing.assert_allclose(Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True), euler, atol=1e-9)


def test_soma_csv_without_a_robot_joint_is_rejected(tmp_path):
    csv = tmp_path / "motion.csv"
    _write_soma_csv(csv, [[0.0] * 8] * 2, ["hip_joint"])
    with pytest.raises(ValueError, match="knee_joint"):
        load_robot_csv(str(csv), fps=30.0, joint_names=["hip_joint", "knee_joint"])


def test_headerless_csv_uses_meters_xyzw_and_radians(tmp_path):
    quat_xyzw = Rotation.from_euler("z", 30, degrees=True).as_quat()
    row = [0.5, -0.25, 0.7, *quat_xyzw, 0.1, 0.2, 0.3]
    csv = tmp_path / "motion.csv"
    csv.write_text("\n".join(",".join(str(v) for v in row) for _ in range(3)) + "\n")

    motion = load_robot_csv(str(csv), fps=30.0, joint_names=JOINTS)

    np.testing.assert_allclose(motion.root_pos[0], [0.5, -0.25, 0.7])
    np.testing.assert_allclose(motion.root_quat[0], quat_xyzw[[3, 0, 1, 2]])
    np.testing.assert_allclose(motion.joint_pos[0], [0.1, 0.2, 0.3])


def test_frame_range_is_one_based_and_inclusive(tmp_path):
    rows = [[float(i), 0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(10)]
    csv = tmp_path / "motion.csv"
    _write_soma_csv(csv, rows, JOINTS)

    motion = load_robot_csv(str(csv), fps=30.0, joint_names=JOINTS, frame_range=(3, 5))

    np.testing.assert_allclose(motion.root_pos[:, 0], [0.02, 0.03, 0.04])
    with pytest.raises(ValueError, match="outside"):
        load_robot_csv(str(csv), fps=30.0, joint_names=JOINTS, frame_range=(5, 11))


def test_resample_interpolates_positions_and_rotations(tmp_path):
    # One second at 120 fps: moving along x at 1 m/s while turning about z at 1 rad/s.
    times = np.arange(121) / 120.0
    rows = [[100.0 * t, 0.0, 60.0, 0.0, 0.0, np.rad2deg(t), np.rad2deg(t), 0.0, 0.0] for t in times]
    csv = tmp_path / "motion.csv"
    _write_soma_csv(csv, rows, JOINTS)

    motion = resample(load_robot_csv(str(csv), fps=120.0, joint_names=JOINTS), fps=50.0)

    assert motion.num_frames == 51
    assert motion.fps == 50.0
    out_times = np.arange(51) / 50.0
    np.testing.assert_allclose(motion.root_pos[:, 0], out_times, atol=1e-9)
    np.testing.assert_allclose(motion.joint_pos[:, 0], out_times, atol=1e-9)
    yaw = Rotation.from_quat(motion.root_quat[:, [1, 2, 3, 0]]).as_euler("xyz")[:, 2]
    np.testing.assert_allclose(yaw, out_times, atol=1e-9)


def test_velocities_of_constant_motion():
    dt = 0.02
    t = np.arange(20) * dt
    np.testing.assert_allclose(linear_velocity(np.stack([3.0 * t] * 3, axis=1), dt), 3.0)

    # Two bodies turning at different constant rates, shaped (frames, bodies, 4).
    quats = np.stack(
        [
            Rotation.from_euler("z", 2.0 * t).as_quat()[:, [3, 0, 1, 2]],
            Rotation.from_euler("x", -1.0 * t).as_quat()[:, [3, 0, 1, 2]],
        ],
        axis=1,
    )
    omega = angular_velocity(quats, dt)
    assert omega.shape == (20, 2, 3)
    np.testing.assert_allclose(omega[:, 0], [[0.0, 0.0, 2.0]] * 20, atol=1e-9)
    np.testing.assert_allclose(omega[:, 1], [[-1.0, 0.0, 0.0]] * 20, atol=1e-9)
