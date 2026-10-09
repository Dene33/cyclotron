"""Prepare a training run for sharing on the Hugging Face Hub with ``./cyclotron.sh --share``.

Plain Python with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
"""

from __future__ import annotations

import ntpath
import re

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
) -> str:
    """Model card (README.md) for a shared motion tracking policy."""
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
    )
