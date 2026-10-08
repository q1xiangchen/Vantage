"""Camera-pose helpers (numpy only).

Grounds the planner's (pose, magnitude, reference_view) choice into a 6-DoF
target pose for view synthesis. Poses are 4x4 c2w (OpenCV), relative to the
reference view.
"""
from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np


def _rot(axis, deg: float) -> np.ndarray:
    """3x3 rotation about `axis` by `deg` degrees (Rodrigues)."""
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    a = a / n
    th = math.radians(deg)
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]], dtype=np.float64)
    R = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)
    return R.astype(np.float32)


def relative_pose(yaw: float = 0.0, pitch: float = 0.0,
                  dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
    """4x4 c2w (OpenCV) target pose relative to the reference view.

    Degrees about camera axes: +yaw faces left, +pitch looks up.
    Translation in scene-spread units: +dx right, +dy down, +dz forward.
    """
    R = _rot([0.0, -1.0, 0.0], yaw) @ _rot([1.0, 0.0, 0.0], pitch)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R
    m[:3, 3] = [dx, dy, dz]
    return m


# --- Poses: magnitude -> relative_pose kwargs ---------------------------------
MAGNITUDES = ('small', 'medium', 'large')
_ROTATE = {
    'pan_left': {    # rotate in place to face left
        'small':  dict(yaw=30.0),
        'medium': dict(yaw=60.0),
        'large':  dict(yaw=90.0),
    },
    'pan_right': {    # rotate in place to face right
        'small':  dict(yaw=-30.0),
        'medium': dict(yaw=-60.0),
        'large':  dict(yaw=-90.0),
    },
    'tilt_up': {    # rotate in place to look up
        'small':  dict(pitch=30.0),
        'medium': dict(pitch=60.0),
        'large':  dict(pitch=90.0),
    },
    'tilt_down': {    # rotate in place to look down
        'small':  dict(pitch=-30.0),
        'medium': dict(pitch=-60.0),
        'large':  dict(pitch=-90.0),
    },
    'turn_around': {    # face the opposite side
        'default':  dict(yaw=180.0),
    },
}

_TRANSLATE = {
    'move_forward': {   # move along the view axis, into the scene
        'small':  dict(dz=0.40),
        'medium': dict(dz=0.80),
        'large':  dict(dz=1.40),
    },
    'move_backward': {  # move along the view axis, out of the scene
        'small':  dict(dz=-0.40),
        'medium': dict(dz=-0.80),
        'large':  dict(dz=-1.40),
    },
    'move_left': {   # slide left, facing unchanged
        'small':  dict(dx=-0.40),
        'medium': dict(dx=-0.80),
        'large':  dict(dx=-1.40),
    },
    'move_right': {   # slide right, facing unchanged
        'small':  dict(dx=0.40),
        'medium': dict(dx=0.80),
        'large':  dict(dx=1.40),
    },
    'pedestal_up': {    # raise straight up (+dy is down)
        'small':  dict(dy=-0.30),
        'medium': dict(dy=-0.55),
        'large':  dict(dy=-0.90),
    },
    'pedestal_down': {    # lower straight down
        'small':  dict(dy=0.30),
        'medium': dict(dy=0.55),
        'large':  dict(dy=0.90),
    },
}

_ROTATE_TRANSLATE = {
    # Orbits: rotation angle about the scene centroid.
    'orbit_left': {     # orbit left, keeping the scene in view
        'small':  dict(yaw=-30.0),
        'medium': dict(yaw=-60.0),
        'large':  dict(yaw=-90.0),
    },
    'orbit_right': {    # orbit right, keeping the scene in view
        'small':  dict(yaw=30.0),
        'medium': dict(yaw=60.0),
        'large':  dict(yaw=90.0),
    },
    'orbit_up': {       # orbit over the scene, looking down
        'small':  dict(pitch=-30.0),
        'medium': dict(pitch=-60.0),
        'large':  dict(pitch=-90.0),
    },
    'orbit_down': {     # orbit under the scene, looking up
        'small':  dict(pitch=30.0),
        'medium': dict(pitch=60.0),
        'large':  dict(pitch=90.0),
    },
    # Elevation views: pitch increment added to the reference camera's pitch.
    'birds_eye_view': {      # rise to a top-down view
        'small':  dict(pitch=-40.0),
        'medium': dict(pitch=-55.0),
        'large':  dict(pitch=-70.0),
    },
    'worms_eye_view': {      # drop to a bottom-up view
        'small':  dict(pitch=40.0),
        'medium': dict(pitch=55.0),
        'large':  dict(pitch=70.0),
    },
}

_POSE_FAMILY_TABLE = {
    **_ROTATE,
    **_TRANSLATE,
    **_ROTATE_TRANSLATE,
}

POSE_FAMILIES = tuple(_POSE_FAMILY_TABLE.keys())

# Synthesized as a rotation about the scene centroid (pivot_mode='centroid').
ORBIT_FAMILIES = ('orbit_left', 'orbit_right', 'orbit_up', 'orbit_down')

# Synthesized as a vertical rise/drop with the heading kept (pivot_mode='elevation').
ELEVATION_FAMILIES = ('birds_eye_view', 'worms_eye_view')


def pose_family_kwargs(family: str, magnitude: str) -> dict:
    """`relative_pose` kwargs for (pose, magnitude). A pose with a single
    'default' entry (e.g. `turn_around`) ignores `magnitude`."""
    try:
        buckets = _POSE_FAMILY_TABLE[family]
    except KeyError:
        raise ValueError(
            f'unknown pose={family!r}; expected one of {POSE_FAMILIES}')
    if 'default' in buckets:
        return dict(buckets['default'])
    try:
        return dict(buckets[magnitude])
    except KeyError:
        raise ValueError(
            f'unknown magnitude={magnitude!r}; expected one of {MAGNITUDES}')


# --- Grounded viewpoint spec --------------------------------------------------
_DOF_KEYS = ('yaw', 'pitch', 'dx', 'dy', 'dz')


@dataclass
class ViewpointSpec:
    """The planned view: a 6-DoF move from `reference_view`.

    Attributes:
        pose_family (str): The pose (see POSE_FAMILIES).
        magnitude (str): small / medium / large.
        target_pose (list): 4x4 c2w (OpenCV) relative to the reference view.
        reference_view (int): 1-indexed input view the move starts from.
        reasoning (str): The planner's justification.
        dof (dict): The grounded yaw / pitch / dx / dy / dz.
    """
    pose_family: str
    magnitude: str
    target_pose: list  # 4x4 nested list (numpy-free so it pickles across ray)
    reference_view: int = 1
    reasoning: str = ''
    dof: dict = field(default_factory=dict)


def ground_viewpoint(
    pose_family: str, magnitude: str, reference_view: int = 1,
    reasoning: str = '', num_views: Optional[int] = None,
) -> ViewpointSpec:
    """Ground (pose, magnitude, reference_view) into a `ViewpointSpec`.
    Raises ValueError on invalid input; `num_views` bounds `reference_view`.

    With a single input view, an invalid `reference_view` falls back to 1
    instead of failing (InternVL3-8B answers it with compass bearings on
    OmniSpatial, even after retries)."""
    if pose_family not in POSE_FAMILIES:
        raise ValueError(
            f'pose="{pose_family}" is invalid. Expected one of {POSE_FAMILIES}.')
    if magnitude not in MAGNITUDES:
        raise ValueError(
            f'magnitude="{magnitude}" is invalid. Expected one of {MAGNITUDES}.')
    single_view = num_views == 1
    if not isinstance(reference_view, int) or reference_view < 1:
        if not single_view:
            raise ValueError(
                f'reference_view="{reference_view}" is invalid. Expected a 1-indexed '
                'integer (1 = the first input view).')
        reference_view = 1
    if num_views is not None and reference_view > num_views:
        if not single_view:
            raise ValueError(
                f'reference_view={reference_view} is out of range for {num_views} '
                'input view(s).')
        reference_view = 1
    kwargs = pose_family_kwargs(pose_family, magnitude)
    dof = {k: float(kwargs.get(k, 0.0)) for k in _DOF_KEYS}
    pose = relative_pose(**dof)
    return ViewpointSpec(
        pose_family=pose_family,
        magnitude=magnitude,
        target_pose=pose.tolist(),
        reference_view=reference_view,
        reasoning=reasoning,
        dof=dof,
    )
