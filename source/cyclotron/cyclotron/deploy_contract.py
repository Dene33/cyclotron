# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Compare the deployment contracts of two exported policies.

``./cyclotron.sh --export`` writes a contract into ``policy.onnx``'s metadata (docs/export.md): what a robot runtime
checks before driving a robot with the file. Two policies with the same contract run in the same runtime, with the
same settings, e.g. a jump policy where the walking policy runs.
"""

from __future__ import annotations

import json
import math

# The keys a runtime checks; the rest (commit, robot model, code checks) are for traceability.
CONTRACT_KEYS = (
    "deploy_metadata_version",
    "obs_dim",
    "action_dim",
    "observation_names",
    "joint_names",
    "action_scale",
    "action_offset",
    "raw_action_clip",
    "action_clip",
    "joint_stiffness",
    "joint_damping",
    "sim_dt",
    "decimation",
    "policy_rate_hz",
)


def read_contract(onnx_path: str) -> dict[str, object]:
    """The deploy metadata of an exported ``policy.onnx``, values decoded from JSON where they are JSON."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    metadata = {}
    for prop in model.metadata_props:
        try:
            metadata[prop.key] = json.loads(prop.value)
        except ValueError:
            metadata[prop.key] = prop.value
    return metadata


def _same(a: object, b: object, rel_tol: float) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, rel_tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)
    return a == b


def contract_differences(a: dict[str, object], b: dict[str, object], rel_tol: float = 1e-5) -> list[str]:
    """One line per contract key on which ``a`` and ``b`` differ, including keys only one of them has."""
    differences = []
    for key in CONTRACT_KEYS:
        if key not in a or key not in b:
            if key in a or key in b:
                differences.append(f"{key}: only in {'the first' if key in a else 'the second'}")
            continue
        if not _same(a[key], b[key], rel_tol):
            differences.append(f"{key}: {a[key]} != {b[key]}")
    return differences
