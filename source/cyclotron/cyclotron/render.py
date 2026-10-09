"""Render robot motions with the URDF's own meshes, posed from root poses and joint angles, for ``./cyclotron.sh
--render``.

Plain NumPy, SciPy, trimesh and matplotlib with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
Meshes are simplified by vertex clustering so a frame with several robots draws in a few seconds.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def _origin(element) -> tuple[np.ndarray, Rotation]:
    if element is None:
        return np.zeros(3), Rotation.identity()
    xyz = np.array([float(v) for v in element.get("xyz", "0 0 0").split()])
    rpy = [float(v) for v in element.get("rpy", "0 0 0").split()]
    return xyz, Rotation.from_euler("xyz", rpy)


def simplify(vertices: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: merge the vertices that fall in the same ``cell``-sized cube and drop collapsed faces."""
    keys = np.floor(vertices / cell).astype(np.int64)
    _, cluster = np.unique(keys, axis=0, return_inverse=True)
    cluster = cluster.reshape(-1)
    merged = np.zeros((cluster.max() + 1, 3))
    np.add.at(merged, cluster, vertices)
    merged /= np.bincount(cluster)[:, None]
    faces = cluster[faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)[1]
    return merged, faces[np.sort(first)]


class RobotModel:
    """A URDF's kinematic tree and visual meshes."""

    def __init__(self, urdf_path: str, cell: float = 0.008):
        root = ET.parse(urdf_path).getroot()
        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        self.children: dict[str, list[dict]] = {}
        child_links = set()
        for joint in root.findall("joint"):
            axis = joint.find("axis")
            entry = {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "child": joint.find("child").get("link"),
                "origin": _origin(joint.find("origin")),
                "axis": np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else np.zeros(3),
            }
            self.children.setdefault(joint.find("parent").get("link"), []).append(entry)
            child_links.add(entry["child"])
        self.root_link = next(link.get("name") for link in root.findall("link") if link.get("name") not in child_links)

        # Visual meshes in their link's frame.
        self.meshes: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for link in root.findall("link"):
            for visual in link.findall("visual"):
                mesh_el = visual.find("geometry/mesh")
                if mesh_el is None:
                    continue
                mesh = trimesh.load(os.path.normpath(os.path.join(urdf_dir, mesh_el.get("filename"))), force="mesh")
                scale = np.array([float(v) for v in mesh_el.get("scale", "1 1 1").split()])
                vertices, faces = simplify(np.asarray(mesh.vertices) * scale, np.asarray(mesh.faces), cell)
                xyz, rot = _origin(visual.find("origin"))
                self.meshes.setdefault(link.get("name"), []).append((rot.apply(vertices) + xyz, faces))

    @property
    def num_triangles(self) -> int:
        return sum(len(faces) for parts in self.meshes.values() for _, faces in parts)

    def forward_kinematics(
        self, root_pos: np.ndarray, root_rot: Rotation, joint_positions: dict[str, float]
    ) -> dict[str, tuple[np.ndarray, Rotation]]:
        """World pose of every link. Joints missing from ``joint_positions`` are at 0."""
        poses = {self.root_link: (np.asarray(root_pos, dtype=float), root_rot)}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            parent_pos, parent_rot = poses[parent]
            for joint in self.children.get(parent, []):
                xyz, rot = joint["origin"]
                pos, child_rot = parent_pos + parent_rot.apply(xyz), parent_rot * rot
                if joint["type"] in ("revolute", "continuous") and np.any(joint["axis"]):
                    angle = joint_positions.get(joint["name"], 0.0)
                    child_rot = child_rot * Rotation.from_rotvec(joint["axis"] / np.linalg.norm(joint["axis"]) * angle)
                elif joint["type"] == "prismatic":
                    pos = pos + child_rot.apply(joint["axis"]) * joint_positions.get(joint["name"], 0.0)
                poses[joint["child"]] = (pos, child_rot)
                stack.append(joint["child"])
        return poses

    def posed_triangles(
        self, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, joint_positions: dict[str, float]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """World triangles (n, 3, 3) of the posed robot, and a ground shadow polygon per link."""
        root_rot = Rotation.from_quat(np.asarray(root_quat_wxyz)[[1, 2, 3, 0]])
        poses = self.forward_kinematics(root_pos, root_rot, joint_positions)
        triangles, shadows = [], []
        for link, parts in self.meshes.items():
            pos, rot = poses[link]
            points = []
            for vertices, faces in parts:
                world = rot.apply(vertices) + pos
                triangles.append(world[faces])
                points.append(world)
            flat = np.concatenate(points)[:, :2]
            try:
                hull = ConvexHull(flat)
            except Exception:  # degenerate hull, e.g. a tiny or flat part
                continue
            shadows.append(np.column_stack([flat[hull.vertices], np.full(len(hull.vertices), 0.001)]))
        return np.concatenate(triangles), shadows


@dataclass
class Actor:
    """One robot in the scene: its motion, sampled at the video's frames, and how it is drawn."""

    label: str
    root_pos: np.ndarray  # (frames, 3)
    root_quat: np.ndarray  # (frames, 4), w, x, y, z
    joint_pos: np.ndarray  # (frames, joints)
    joint_names: list[str]
    color: tuple[float, float, float]
    label_color: str


@dataclass
class View:
    """Orthographic camera: a cube of ``2 * half`` around ``center``, seen from ``elev`` and ``azim`` degrees and
    magnified ``zoom`` times."""

    center: np.ndarray
    half: float
    elev: float
    azim: float
    zoom: float = 1.0


WIDTH, HEIGHT = 1280, 720


def _axes(fig, view: View):
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    ax.view_init(elev=view.elev, azim=view.azim)
    for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), view.center):
        set_lim(c - view.half, c + view.half)
    ax.set_box_aspect((1, 1, 1), zoom=view.zoom)
    ax.set_axis_off()
    return ax


def project(view: View, points: np.ndarray) -> np.ndarray:
    """Pixel coordinates (from the bottom left of the frame) of world ``points`` (N x 3)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d

    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax = _axes(fig, view)
    fig.canvas.draw()
    x, y, _ = proj3d.proj_transform(points[:, 0], points[:, 1], points[:, 2], ax.get_proj())
    pixels = ax.transData.transform(np.column_stack([x, y]))
    plt.close(fig)
    return pixels


def fit_view(view: View, points: np.ndarray, margin: tuple[float, float, float, float] = (80, 80, 50, 60)) -> View:
    """Move and zoom ``view``, keeping its angle, so ``points`` fill the frame up to ``margin`` pixels from its left,
    right, bottom and top edges."""
    lo = np.array([margin[0], margin[2]], dtype=float)
    hi = np.array([WIDTH - margin[1], HEIGHT - margin[3]], dtype=float)
    frame_center = np.array([WIDTH, HEIGHT]) / 2
    for _ in range(3):  # matplotlib zooms about the frame centre, so a pass or two more settles any rounding
        pixels = project(view, points)
        extent = pixels.max(axis=0) - pixels.min(axis=0)
        scale = np.min((hi - lo) / np.maximum(extent, 1e-6))
        # Where the content centre must be now so that the extra zoom about the frame centre lands it mid-margins.
        target = frame_center + ((lo + hi) / 2 - frame_center) / scale
        shift = target - (pixels.max(axis=0) + pixels.min(axis=0)) / 2
        # Pixels per metre along the world axes; moving the camera centre by d moves the content by -jacobian @ d.
        probe = project(view, view.center[None] + np.vstack([np.zeros(3), np.eye(3)]))
        jacobian = (probe[1:] - probe[0]).T
        view = View(view.center - np.linalg.pinv(jacobian) @ shift, view.half, view.elev, view.azim, view.zoom * scale)
    return view


_LIGHT = np.array([0.4, -0.3, 0.85]) / np.linalg.norm([0.4, -0.3, 0.85])


def _shade(triangles: np.ndarray, color) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    lambert = np.abs(normals @ _LIGHT)  # two-sided: mesh winding is not consistent
    return np.clip(np.asarray(color)[None] * (0.45 + 0.6 * lambert[:, None]), 0.0, 1.0)


def render_frame(
    model: RobotModel,
    actors: list[Actor],
    index: int,
    view: View,
    path: str,
    time_s: float,
    title: str = "",
    caption: str = "",
    label_size: float = 13.0,
    label_height: float = 1.72,
) -> None:
    """Draw frame ``index`` of every actor to a 1280x720 PNG at ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    triangles, colors, shadows = [], [], []
    for actor in actors:
        joints = dict(zip(actor.joint_names, actor.joint_pos[index]))
        tris, shadow = model.posed_triangles(actor.root_pos[index], actor.root_quat[index], joints)
        triangles.append(tris)
        colors.append(_shade(tris, actor.color))
        shadows += shadow

    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax = _axes(fig, view)

    # Ground grid every 25 cm, then shadows, then the robots on top. 3D axes are square, so drawing is not clipped to
    # them: a zoomed-in view uses the whole frame.
    lo, hi = view.center[:2] - view.half - 0.5, view.center[:2] + view.half + 0.5
    grid = [[[x, lo[1], 0], [x, hi[1], 0]] for x in np.arange(np.floor(lo[0] / 0.25) * 0.25, hi[0], 0.25)]
    grid += [[[lo[0], y, 0], [hi[0], y, 0]] for y in np.arange(np.floor(lo[1] / 0.25) * 0.25, hi[1], 0.25)]
    ax.add_collection3d(Line3DCollection(grid, colors=(0.82, 0.82, 0.82), linewidths=0.6, zorder=0, clip_on=False))
    ax.add_collection3d(
        Poly3DCollection(shadows, facecolors=(0, 0, 0, 0.13), edgecolors="none", zorder=1, clip_on=False)
    )
    triangles, colors = np.concatenate(triangles), np.concatenate(colors)
    ax.add_collection3d(
        Poly3DCollection(triangles, facecolors=colors, edgecolors=colors, linewidths=0.15, zorder=2, clip_on=False)
    )

    for actor in actors:
        pos = actor.root_pos[index]
        ax.text(
            pos[0],
            pos[1],
            label_height,
            actor.label,
            ha="center",
            fontsize=label_size,
            zorder=3,
            color=actor.label_color,
            weight="bold",
        )
    fig.text(0.02, 0.95, f"t = {time_s:.2f} s", fontsize=14, family="monospace")
    if title:
        fig.text(0.98, 0.95, title, fontsize=12, ha="right", color="#555555")
    if caption:
        fig.text(0.5, 0.03, caption, fontsize=11, ha="center", color="#555555")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
