"""Read retargeted robot motion CSVs and resample them for ``./cyclotron.sh --csv-to-npz``.

Plain NumPy and SciPy with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Root columns written by soma-retargeter: position in cm, then extrinsic xyz Euler angles in degrees.
SOMA_ROOT_COLUMNS = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
]


@dataclass
class RobotMotion:
    """A robot motion sampled at a fixed rate. Quaternions are (w, x, y, z), lengths in meters, angles in radians."""

    fps: float
    root_pos: np.ndarray  # (frames, 3), world frame
    root_quat: np.ndarray  # (frames, 4), world frame
    joint_pos: np.ndarray  # (frames, joints), in joint_names order
    joint_names: list[str]

    @property
    def num_frames(self) -> int:
        return self.root_pos.shape[0]

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps


def _wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return quat_xyzw[..., [3, 0, 1, 2]]


def _xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_wxyz[..., [1, 2, 3, 0]]


def _has_header(first_line: str) -> bool:
    try:
        float(first_line.split(",")[0])
    except ValueError:
        return True
    return False


def load_robot_csv(
    path: str, fps: float, joint_names: list[str], frame_range: tuple[int, int] | None = None
) -> RobotMotion:
    """Load a robot motion CSV, with joints returned in ``joint_names`` order.

    Two layouts are accepted:

    - soma-retargeter, recognised by its header row: root position in cm, root rotation as extrinsic xyz Euler
      angles in degrees, then joint angles in degrees, one column per joint named in the header. Columns for joints
      not in ``joint_names`` (such as undriven ones) are dropped.
    - Headerless, as in the Unitree retargeted LAFAN data used by whole_body_tracking: root position in m, root
      quaternion (x, y, z, w), then joint angles in radians in ``joint_names`` order.

    Args:
        path: The CSV file.
        fps: Frame rate of the CSV. It is not stored in the file.
        joint_names: Joints to return, in this order.
        frame_range: Frames to keep, as (start, end), 1-based and both inclusive, counted over data rows.
    """
    with open(path, encoding="utf-8") as f:
        first_line = f.readline().strip()
    header = _has_header(first_line)
    data = np.loadtxt(path, delimiter=",", skiprows=1 if header else 0, ndmin=2)
    if frame_range is not None:
        start, end = frame_range
        if not 1 <= start <= end <= data.shape[0]:
            raise ValueError(f"--frame_range {start} {end} is outside the CSV's frames 1 to {data.shape[0]}.")
        data = data[start - 1 : end]
    if data.shape[0] < 2:
        raise ValueError(f"{path} needs at least 2 frames, found {data.shape[0]}.")

    if header:
        columns = [name.strip() for name in first_line.split(",")]
        if columns[: len(SOMA_ROOT_COLUMNS)] != SOMA_ROOT_COLUMNS:
            raise ValueError(
                f"{path} has a header but not soma-retargeter's root columns {SOMA_ROOT_COLUMNS}; found {columns[:7]}."
            )
        file_joints = columns[len(SOMA_ROOT_COLUMNS) :]
        missing = [name for name in joint_names if name not in file_joints]
        if missing:
            raise ValueError(f"{path} has no column for joints {missing}. Was it retargeted to the right robot?")
        root_pos = data[:, 1:4] * 0.01
        root_quat = _wxyz(Rotation.from_euler("xyz", data[:, 4:7], degrees=True).as_quat())
        joint_columns = [len(SOMA_ROOT_COLUMNS) + file_joints.index(name) for name in joint_names]
        joint_pos = np.deg2rad(data[:, joint_columns])
    else:
        expected = 7 + len(joint_names)
        if data.shape[1] != expected:
            raise ValueError(
                f"{path} has {data.shape[1]} columns; a headerless CSV needs {expected}: root position (3), root"
                f" quaternion x, y, z, w (4) and {len(joint_names)} joint angles."
            )
        root_pos = data[:, 0:3]
        root_quat = _wxyz(data[:, 3:7] / np.linalg.norm(data[:, 3:7], axis=1, keepdims=True))
        joint_pos = data[:, 7:]

    return RobotMotion(
        fps=float(fps),
        root_pos=root_pos.astype(np.float64),
        root_quat=root_quat.astype(np.float64),
        joint_pos=joint_pos.astype(np.float64),
        joint_names=list(joint_names),
    )


def resample(motion: RobotMotion, fps: float) -> RobotMotion:
    """Resample to ``fps``, interpolating positions linearly and rotations spherically. The last frame is kept when
    the duration is a whole number of output frames."""
    src_times = np.arange(motion.num_frames) / motion.fps
    times = np.arange(0.0, motion.duration + 1.0e-9, 1.0 / fps)
    times = np.minimum(times, src_times[-1])

    def lerp(values: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(times, src_times, values[:, i]) for i in range(values.shape[1])], axis=1)

    slerp = Slerp(src_times, Rotation.from_quat(_xyzw(motion.root_quat)))
    return RobotMotion(
        fps=float(fps),
        root_pos=lerp(motion.root_pos),
        root_quat=_wxyz(slerp(times).as_quat()),
        joint_pos=lerp(motion.joint_pos),
        joint_names=list(motion.joint_names),
    )


def linear_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """Time derivative along the first axis: central differences inside, one-sided at the ends."""
    return np.gradient(positions, dt, axis=0)


def angular_velocity(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity of a sequence of rotations, shaped (frames, ..., 4) -> (frames, ..., 3).

    Central differences inside, one-sided at the ends, matching :func:`linear_velocity`.
    """
    shape = quats_wxyz.shape[:-1]
    flat = quats_wxyz.reshape(shape[0], -1, 4)
    omega = np.zeros(flat.shape[:-1] + (3,))
    for i in range(flat.shape[1]):
        rot = Rotation.from_quat(_xyzw(flat[:, i]))
        omega[1:-1, i] = (rot[2:] * rot[:-2].inv()).as_rotvec() / (2.0 * dt)
        omega[0, i] = (rot[1] * rot[0].inv()).as_rotvec() / dt
        omega[-1, i] = (rot[-1] * rot[-2].inv()).as_rotvec() / dt
    return omega.reshape(shape + (3,))
