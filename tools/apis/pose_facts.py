"""Describe relative camera motion between input views in natural language."""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np


# Below these thresholds a component is not mentioned (it is within estimation noise).
STAY_REL = 0.05        # |translation| / viewing distance below this -> "same position"
TURN_MIN_DEG = 5.0     # heading change below this -> "about the same heading"
TILT_MIN_DEG = 10.0    # tilt change below this -> not mentioned
COMP_MIN_FRAC = 0.25   # translation component below this fraction of |t| -> not mentioned
COMP_SLIGHT_FRAC = 0.5  # mentioned component below this fraction -> "slightly ..."
# "circled around the content in front" needs a real sideways move AND a turn back
# toward the side the camera came from (opposite signs).
ORBIT_MIN_REL = 0.10
ORBIT_MIN_TURN_DEG = 10.0
ORBIT_MIN_LATERAL_FRAC = 0.4
# Pairs listed: every i < j while the view count is small, else each view vs Image 1.
ALL_PAIRS_MAX_VIEWS = 4


def gravity_frame_np(R: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a gravity-aligned camera basis with right, down, and forward axes."""
    up = up / np.linalg.norm(up)
    fwd = R[:, 2]
    fwd_h = fwd - np.dot(fwd, up) * up
    if np.linalg.norm(fwd_h) < 1e-4:          # looking straight up/down
        fwd_h = R[:, 1] - np.dot(R[:, 1], up) * up
    fwd_g = fwd_h / np.linalg.norm(fwd_h)
    down_g = -up
    right_g = np.cross(down_g, fwd_g)
    right_g /= np.linalg.norm(right_g)
    fwd_g = np.cross(right_g, down_g)
    return np.stack([right_g, down_g, fwd_g], axis=1)


def _heading_pitch(f: np.ndarray):
    """(yaw_right_deg, pitch_up_deg) of a direction given in a [right, down, fwd] frame."""
    yaw = math.degrees(math.atan2(f[0], f[2]))
    pitch = math.degrees(math.atan2(-f[1], math.hypot(f[0], f[2])))
    return yaw, pitch


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def relative_motion(c2w_i: np.ndarray, c2w_j: np.ndarray, up: np.ndarray, gravity_align: bool = True) -> Dict:
    """Express one camera’s motion relative to another in the chosen reference frame."""
    c2w_i = np.asarray(c2w_i, dtype=np.float64)
    c2w_j = np.asarray(c2w_j, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    G = gravity_frame_np(c2w_i[:3, :3], up) if gravity_align else c2w_i[:3, :3]
    d = G.T @ (c2w_j[:3, 3] - c2w_i[:3, 3])          # [right, down, fwd]
    yaw_i, pitch_i = _heading_pitch(G.T @ c2w_i[:3, 2])
    yaw_j, pitch_j = _heading_pitch(G.T @ c2w_j[:3, 2])
    return dict(
        right=float(d[0]), up=float(-d[1]), forward=float(d[2]),
        dist=float(np.linalg.norm(d)),
        yaw_right_deg=_wrap180(yaw_j - yaw_i),
        pitch_up_deg=float(pitch_j - pitch_i),
    )


def _round5(a: float) -> int:
    return int(5 * round(abs(a) / 5.0))


def describe_motion(m: Dict, view_dist: Optional[float], style: str = 'v1') -> Dict:
    """Verbalize the translation and rotation between two views."""
    dist = m['dist']
    rel = dist / view_dist if view_dist and view_dist > 1e-9 else None
    labels: Dict = {'rel_dist': rel}

    # --- translation -----------------------------------------------------------
    if dist < 1e-9 or (rel is not None and rel < STAY_REL):
        move = 'stayed at about the same position'
        labels['lateral'] = 'none'
    else:
        comps = [
            ('to the right' if m['right'] > 0 else 'to the left', abs(m['right'])),
            ('forward' if m['forward'] > 0 else 'backward', abs(m['forward'])),
            ('up' if m['up'] > 0 else 'down', abs(m['up'])),
        ]
        comps.sort(key=lambda c: -c[1])
        words = []
        for k, (name, mag) in enumerate(comps):
            frac = mag / dist
            if k == 0:
                words.append(name)
            elif frac >= COMP_MIN_FRAC:
                words.append(('slightly ' if frac < COMP_SLIGHT_FRAC else '') + name)
        move = 'moved ' + ' and '.join(words)
        if rel is not None:
            move += f' (by about {rel:.1f} times its distance to the scene)'
        lat_frac = abs(m['right']) / dist
        labels['lateral'] = (('right' if m['right'] > 0 else 'left')
                             if lat_frac >= COMP_MIN_FRAC else 'none')

    # --- rotation ----------------------------------------------------------------
    yaw, pitch = m['yaw_right_deg'], m['pitch_up_deg']
    if abs(yaw) < TURN_MIN_DEG:
        turn = 'kept about the same heading'
        labels['turn'] = 'none'
    else:
        side = 'right' if yaw > 0 else 'left'
        turn = f'turned about {_round5(yaw)} degrees to the {side}'
        labels['turn'] = side
    if abs(pitch) >= TILT_MIN_DEG:
        turn += f' and tilted about {_round5(pitch)} degrees {"up" if pitch > 0 else "down"}'

    sentence = f'the camera {move}, and {turn}.'

    # --- orbit summary -------------------------------------------------------------
    # A sideways move with a turn back toward where the camera came from is the
    # signature of circling around the content in front of the cameras.
    labels['orbit'] = 'none'
    if (rel is not None and rel >= ORBIT_MIN_REL and abs(yaw) >= ORBIT_MIN_TURN_DEG
            and dist > 0 and abs(m['right']) / dist >= ORBIT_MIN_LATERAL_FRAC
            and (m['right'] > 0) != (yaw > 0)):
        side = 'right' if m['right'] > 0 else 'left'
        labels['orbit'] = side
        # Seen from above (looking down the vertical), circling to the camera's LEFT
        # around content ahead of it is CLOCKWISE (check_verbalizer.py verifies this
        # against the signed swept angle of the camera centres about the pivot).
        labels['orbit_sense'] = 'clockwise' if side == 'left' else 'counter-clockwise'
        if style == 'v2':
            # orbit-first: the circling direction leads, the turn is explained as the
            # consequence of keeping the content in view (v1 listed the turn on par
            # with the move, and a 4B reader latched onto "turned left").
            turn_side = 'right' if yaw > 0 else 'left'
            sentence = (f'the camera circled to the {side} around the content in front '
                        f'of it ({labels["orbit_sense"]} as seen from above): it {move}, '
                        f'while turning about {_round5(yaw)} degrees to the {turn_side} '
                        'to keep that content in view.')
        else:
            sentence += (f' Together, this means the camera circled to the {side} '
                         'around the content in front of it.')
    labels['sentence'] = sentence
    return labels


def pair_list(n: int) -> List[tuple]:
    if n <= ALL_PAIRS_MAX_VIEWS:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    return [(0, j) for j in range(1, n)]


POSE_FACTS_HEADER = (
    'Relative camera poses between the input images, computed by a 3D reconstruction '
    'of the input images (not by visual guessing). Directions are given as seen by '
    'the camera of the earlier image: "left/right/forward" are relative to where that '
    'camera faces, and "up/down" is true vertical.'
)


def verbalize_pose_facts(
    c2ws: Sequence[np.ndarray],
    up: np.ndarray,
    view_dists: Optional[Sequence[Optional[float]]] = None,
    style: str = 'v1',
    gravity_align: bool = True,
) -> Dict:
    """Return natural-language relations and geometric measurements for input-view pairs."""
    n = len(c2ws)
    if n < 2:
        return {'text': '', 'pairs': []}
    view_dists = list(view_dists) if view_dists is not None else [None] * n
    lines, pairs = [], []
    for i, j in pair_list(n):
        m = relative_motion(c2ws[i], c2ws[j], up, gravity_align=gravity_align)
        lab = describe_motion(m, view_dists[i], style=style)
        lines.append(f'- Image {j + 1} relative to Image {i + 1}: {lab["sentence"]}')
        pairs.append({'i': i + 1, 'j': j + 1, **m, **lab})
    header = POSE_FACTS_HEADER if gravity_align else POSE_FACTS_HEADER.replace('"up/down" is true vertical.', '"up/down" follows that camera’s image axes, without gravity alignment.')
    return {'text': header + '\n' + '\n'.join(lines), 'pairs': pairs, 'gravity_aligned': gravity_align}
