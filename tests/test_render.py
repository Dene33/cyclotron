from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from cyclotron.render import HEIGHT, WIDTH, RobotModel, View, fit_view, project, simplify

URDF = Path(__file__).resolve().parents[1] / "third_party" / "asimov-1" / "sim-model" / "urdf" / "asimov_1.urdf"


def test_simplify_merges_vertices_and_drops_collapsed_faces():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.05)
    vertices, faces = simplify(np.asarray(mesh.vertices), np.asarray(mesh.faces), cell=0.02)

    assert 0 < len(faces) < len(mesh.faces)
    assert np.all((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2]))
    assert faces.max() < len(vertices)
    # Merged vertices stay on the sphere's scale.
    assert np.abs(np.linalg.norm(vertices, axis=1) - 0.05).max() < 0.02


def test_fit_view_fills_the_frame_up_to_the_margins():
    # Two robots' worth of points, far from the centre of a camera framed for a much larger scene.
    points = np.array([[x, y, z] for x in (1.0, 1.6) for y in (-0.5, -0.2) for z in (0.0, 1.7)])
    view = View(center=np.array([0.0, 0.0, 0.9]), half=4.0, elev=25.0, azim=45.0)
    margin = (80, 80, 50, 60)

    fitted = fit_view(view, points, margin)
    pixels = project(fitted, points)

    assert fitted.zoom > 1.0 and (fitted.elev, fitted.azim) == (view.elev, view.azim)
    lo, hi = pixels.min(axis=0), pixels.max(axis=0)
    np.testing.assert_array_less([margin[0] - 1, margin[2] - 1], lo)
    np.testing.assert_array_less(hi, [WIDTH - margin[1] + 1, HEIGHT - margin[3] + 1])
    # The tighter direction touches its margins on both sides.
    assert np.isclose(lo[0], margin[0], atol=2) or np.isclose(lo[1], margin[2], atol=2)
    assert np.isclose(hi[0], WIDTH - margin[1], atol=2) or np.isclose(hi[1], HEIGHT - margin[3], atol=2)


@pytest.mark.skipif(not URDF.is_file(), reason="Asimov-1 model not checked out")
def test_forward_kinematics_matches_the_default_standing_pose():
    model = RobotModel(str(URDF), cell=0.02)
    # Asimov-1's default standing pose, as in cyclotron.assets.robots.asimov_1.
    joints = {"left_hip_pitch_joint": -0.15, "left_knee_joint": 0.45, "left_ankle_pitch_joint": -0.30}
    poses = model.forward_kinematics(np.array([0.0, 0.0, 0.639]), Rotation.identity(), joints)

    ankle_pos, ankle_rot = poses["left_ankle_roll_link"]
    np.testing.assert_allclose(ankle_pos[2], 0.059, atol=2e-3)  # as Isaac Sim reports for this pose
    sole_tilt = np.degrees(np.arccos(ankle_rot.apply([0.0, 0.0, 1.0])[2]))
    assert sole_tilt < 1.0

    triangles, shadows = model.posed_triangles(np.array([0.0, 0.0, 0.639]), np.array([1.0, 0.0, 0.0, 0.0]), joints)
    assert triangles.shape[1:] == (3, 3) and len(triangles) == model.num_triangles
    assert triangles[..., 2].min() > -0.01  # nothing below the ground
    assert len(shadows) > 0
