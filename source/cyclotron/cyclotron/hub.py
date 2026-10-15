"""Prepare a training run for ``./cyclotron.sh --export`` and ``--share``.

Plain Python with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
"""

from __future__ import annotations

import hashlib
import ntpath
import os
import re

# Training task for each experiment name, so a run can be exported without passing --task.
EXPERIMENT_TASKS = {
    "asimov1_velocity": "Asimov1-Velocity-v0",
    "asimov_velocity_amp": "Asimov1-Velocity-AMP-v0",
    "asimov1_tracking": "Asimov1-Tracking-v0",
    "asimov1_jump": "Asimov1-Jump-v0",
}


def checkpoints(run_dir: str) -> list[str]:
    """The run's ``model_<iteration>.pt`` checkpoint files, oldest iteration first."""
    names = [name for name in os.listdir(run_dir) if re.fullmatch(r"model_\d+\.pt", name)]
    return sorted(names, key=lambda name: int(name[len("model_") : -len(".pt")]))


def read_experiment_name(agent_yaml_path: str) -> str | None:
    """Return ``experiment_name`` from a run's ``params/agent.yaml``, without loading Isaac Lab's yaml tags."""
    with open(agent_yaml_path) as f:
        match = re.search(r"^experiment_name: *(\S+)", f.read(), re.MULTILINE)
    return match.group(1).strip("'\"") if match else None


def infer_task(run_dir: str) -> str:
    """Return the training task of a run directory, from the experiment name in its ``params/agent.yaml``."""
    agent_yaml_path = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(agent_yaml_path):
        raise ValueError(f"Cannot infer the task: {agent_yaml_path} not found. Pass it with --task.")
    experiment_name = read_experiment_name(agent_yaml_path)
    if experiment_name not in EXPERIMENT_TASKS:
        raise ValueError(f"Cannot infer the task for experiment '{experiment_name}'. Pass it with --task.")
    return EXPERIMENT_TASKS[experiment_name]


# Keys holding files on the training machine, e.g. asset_path, cache_dir, motion_files. USD prim paths (prim_path,
# filter_prim_paths_expr, ...) also start with "/" but name scene objects, not files, so they are kept.
_LOCAL_PATH_KEY = re.compile(r"\w*(_path|_dir|_file|_files)")
_ABSOLUTE_PATH = re.compile(r"(/|~|[A-Za-z]:[\\/])")
_KEY_LINE = re.compile(r"( *)(- )?([\w.-]+): ?(.*?)(\s*)$")
_ITEM_LINE = re.compile(r"( *)- (.*?)(\s*)$")


def _strip_value(value: str, removed: list[str]) -> str:
    quote = value[0] if value[:1] in ("'", '"') and value[-1:] == value[:1] else ""
    path = value[1:-1] if quote else value
    if not _ABSOLUTE_PATH.match(path):
        return value
    removed.append(path)
    return quote + ntpath.basename(path.rstrip("/\\")) + quote


def strip_local_paths(text: str) -> tuple[str, list[str]]:
    """Replace training-machine file paths in a run's yaml with their file names, so the yaml can be shared.

    Works line by line rather than loading and dumping the yaml, so every other line, including Isaac Lab's
    Python tags, stays byte for byte the same. Returns the new text and the paths that were replaced.
    """
    lines, removed = [], []
    list_indent = None  # indent of a key like motion_files whose path list is being read
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        item = _ITEM_LINE.fullmatch(body)
        if list_indent is not None and item and len(item.group(1)) >= list_indent and ": " not in item.group(2):
            lines.append(item.group(1) + "- " + _strip_value(item.group(2), removed) + item.group(3) + newline)
            continue
        list_indent = None
        key = _KEY_LINE.fullmatch(body)
        if key and _LOCAL_PATH_KEY.fullmatch(key.group(3)) and "prim" not in key.group(3):
            indent, dash, name, value, trailing = key.groups()
            if not value:
                list_indent = len(indent) + len(dash or "")
            else:
                body = f"{indent}{dash or ''}{name}: {_strip_value(value, removed)}{trailing}"
        lines.append(body + newline)
    return "".join(lines), removed


TRACKING_CARD = """\
---
library_name: asimov
pipeline_tag: robotics
license: bsd-3-clause
tags:
- motion-tracking
---

# {title}

{summary}

A motion tracking policy: it makes Asimov 1 follow one reference motion, `{motion}`, frame by frame. It reads the
motion at run time, so the motion file is part of the policy.

## Files

- `policy.onnx`: the policy, from `obs` (1, 124) to `actions` (1, 23). It includes its observation normalisation.
- `{motion}`: the reference motion at {fps:g} Hz, {duration:.2f} s long. `joint_pos` and `joint_vel` are in the
  policy's joint order (`joint_names`); `body_pos_w` and `body_quat_w` (w, x, y, z) give every link's pose, the first
  being the pelvis.
- `env.yaml` and `agent.yaml`: the training task and PPO settings, including the joint order, default pose, PD gains
  and effort limits.
{extra_files}
## Running it

Start with the robot in the motion's first pose. Then every {step_ms:g} ms, for frame `t` = 0, 1, 2, ... of the motion:

1. Build the observation, 124 values in this order:
   - the motion's joint positions (23) and joint velocities (23) at frame `t`;
   - the motion's pelvis orientation in the robot's pelvis frame, as the first two columns of the rotation matrix,
     row by row (6);
   - the pelvis angular velocity in the pelvis frame, from the IMU (3);
   - the joint positions minus the default pose (23), and the joint velocities (23);
   - the previous action (23), zeros at the start.
2. Send joint position targets of default pose + {action_scale:g} x action to the PD controllers.

At the motion's end, hold the last pose or hand over to another policy.

## Status

{checkpoint}Trained in simulation (Isaac Lab) only. Check it with `./cyclotron.sh --evaluate` and `--sim2sim` before
running it on a robot. `./cyclotron.sh --view` plays velocity policies, not this one.
{motion_source}
Training and evaluation code: [{code_name}]({code_url}).
"""


def link_text(url: str) -> str:
    """Short text for a link to ``url``: without the scheme and, on GitHub, the host."""
    return re.sub(r"^https?://(github\.com/)?", "", url).rstrip("/")


def tracking_model_card(
    title: str,
    summary: str,
    motion: str,
    duration: float,
    fps: float,
    action_scale: float,
    code_url: str,
    checkpoint: str | None = None,
    motion_source: str | None = None,
    extra_files: str = "",
) -> str:
    """Model card (README.md) for a shared motion tracking policy. ``extra_files`` lists further uploaded files, one
    markdown list item per line."""
    return TRACKING_CARD.format(
        title=title,
        summary=summary,
        motion=motion,
        duration=duration,
        fps=fps,
        step_ms=1000.0 / fps,
        action_scale=action_scale,
        code_url=code_url,
        code_name=link_text(code_url),
        checkpoint=f"Checkpoint `{checkpoint}`. " if checkpoint else "",
        motion_source=f"\n## Reference motion\n\n{motion_source}\n" if motion_source else "",
        extra_files=extra_files,
    )


JUMP_CARD = """\
---
library_name: asimov
pipeline_tag: robotics
license: bsd-3-clause
tags:
- jump
---

# {title}

{summary}

A forward jump on command for Asimov 1, with the walking policy's interface: it runs where the walking policy runs.

## Interface

`policy.onnx` has the deployment contract of the walking policy it was trained against ({walking}): the same entries
in its ONNX metadata for the observations, joint order, action scales, offsets and clips, PD gains and 50 Hz rate. A
runtime that runs that walking policy runs this one with the same settings:

- 78 observations: the pelvis angular velocity x 0.25, gravity's direction, the velocity command (forward speed,
  sideways speed, yaw rate), the joint positions relative to the default pose and the joint velocities x 0.1 in the
  motor-slot order, and the previous action.
- 23 joint position targets: the default pose plus the action times its scale (0.08 for the ankle rolls, 0.25 for
  the other joints), with the ankle rolls clipped to 0.09 rad either way.

The velocity command is the trigger:

- Zero: the robot stands and balances.
- A forward speed of {speed:g} m/s (it was trained with {speed_low:g} to {speed_high:g}), held for the whole jump,
  about {hold:.1f} s, then released: one forward jump. The robot lands about {landing:.1f} s after the command starts.
  Held on after that, it jumps again.

It doesn't walk: hand over to the walking policy once it stands.

## Files

- `policy.onnx`: the policy, from `obs` (1, 78) to `actions` (1, 23), with its observation normalisation and the
  deployment contract in its metadata.
- `env.yaml` and `agent.yaml`: the training task and PPO settings, including the observation recipe, joint order,
  default pose, PD gains and effort limits.
{extra_files}
## How it was trained

`Asimov1-Jump-v0` in the training code, with PPO in Isaac Lab: the motion tracking rewards of `Asimov1-Tracking-v0`
on `{motion}`, but the policy never sees the reference motion. It only shapes the rewards and what the critic sees, so
the policy doesn't need it to run. The ankle targets are clamped in training as Asimov 1's firmware clamps them
(0.35 rad pitch, 0.1 rad roll).
{motion_source}
## Status

{checkpoint}Trained in simulation only. Before a robot, play it in MuJoCo (`./cyclotron.sh --sim2sim`), then on the
robot with `./cyclotron.sh --sim2real`, hanging from its gantry hook.

Training and evaluation code: [{code_name}]({code_url}).
"""


def jump_model_card(
    title: str,
    summary: str,
    walking: str,
    motion: str,
    speed: float,
    speed_range: tuple[float, float],
    hold_s: float,
    landing_s: float,
    code_url: str,
    checkpoint: str | None = None,
    motion_source: str | None = None,
    extra_files: str = "",
) -> str:
    """Model card (README.md) for a shared jump policy (``Asimov1-Jump``). ``speed`` is the forward speed to tell it
    to jump with, ``speed_range`` the speeds it was trained with, ``hold_s`` how long the command is held and
    ``landing_s`` when it lands, from the command's start."""
    return JUMP_CARD.format(
        title=title,
        summary=summary,
        walking=walking,
        motion=motion,
        speed=speed,
        speed_low=speed_range[0],
        speed_high=speed_range[1],
        hold=hold_s,
        landing=landing_s,
        code_url=code_url,
        code_name=link_text(code_url),
        checkpoint=f"Checkpoint `{checkpoint}`. " if checkpoint else "",
        motion_source=f"\nReference motion: {motion_source}\n" if motion_source else "",
        extra_files=extra_files,
    )


def rehash_uploaded_run_configs(code_state: str, recorded: dict[str, str], uploaded: dict[str, str]) -> str:
    """Point a run's ``code_state.yaml`` text at the copies of ``env.yaml`` and ``agent.yaml`` being uploaded, which
    have local paths stripped: each sha256 training recorded (``recorded``) becomes that of the uploaded text."""
    for name, digest in recorded.items():
        if name in uploaded:
            code_state = code_state.replace(digest, hashlib.sha256(uploaded[name].encode()).hexdigest())
    return code_state
