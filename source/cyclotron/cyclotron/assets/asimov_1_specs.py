# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 actuator limits, without Isaac Lab, so tools outside Isaac Sim can use them.

Per actuator group of ``ASIMOV_1_ACTUATORS``, from https://docs.menlo.ai/asimov/1/overview/system-tour/mechanical.
The speed at which an actuator can no longer produce torque (no-load speed) is not published; it is at least the
maximum speed.
"""

ASIMOV_1_PEAK_TORQUE = {
    "hip_pitch": 120.0,
    "hip_roll": 90.0,
    "hip_yaw": 60.0,
    "knee": 75.0,
    "ankle_pitch": 145.4,
    "ankle_roll": 57.6,
    "waist": 120.0,
    "shoulder_pitch": 90.0,
    "shoulder_roll": 75.0,
    "shoulder_yaw": 60.0,
    "elbow_wrist": 36.0,
}
"""Peak torque (N·m), available for short events."""

ASIMOV_1_MAX_SPEED = {
    "hip_pitch": 12.57,
    "hip_roll": 3.98,
    "hip_yaw": 5.45,
    "knee": 12.25,
    "ankle_pitch": 9.32,
    "ankle_roll": 9.32,
    "waist": 12.57,
    "shoulder_pitch": 3.98,
    "shoulder_roll": 12.25,
    "shoulder_yaw": 5.45,
    "elbow_wrist": 9.32,
}
"""Maximum speed (rad/s), the same as the URDF's velocity limits."""
