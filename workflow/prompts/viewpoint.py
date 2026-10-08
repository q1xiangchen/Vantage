"""Pose-family vocabulary for the s2 pose planner's viewpoint choice.

The s2 pose planner plans the view to seek: it chooses a
coarse, named *pose* plus a *magnitude* bucket — never raw camera numbers (a
frozen VLM has no sense of the scene's metric / spread scale). The grounding from
(family, magnitude) -> explicit 6-DoF camera delta lives in `tools.apis.pose_utils`
(`ground_viewpoint`, the single source of truth for the numbers). The delta is
RELATIVE to the FIRST input view; identity = the first input view.

This module only holds the human-readable descriptions shown in the planner prompt;
keep them coarse and viewpoint-centric, and keep the keys in sync with pose_utils.
"""
from tools.apis.pose_utils import POSE_FAMILIES


POSE_FAMILY_DESCRIPTIONS = {
    # orbit: translate AND turn back, so the near region stays roughly centered
    # while you change angle (parallax reveals occluded sides / what is behind it).
    'orbit_left':     "to uncover the BACKGROUND the centered near region hides — orbit left so the region rotates and NEW background swings in at the RIGHT edge (the goal is this new RIGHT-side background; do NOT expect the scene's left content) -> FG: stays centered, rotates to show its LEFT (right side self-occludes). BG: backdrop sweeps LEFT. ENTERS: new background at RIGHT edge. LEAVES: the region's RIGHT side + background at LEFT edge",
    'orbit_right':    "to uncover the BACKGROUND the centered near region hides — orbit right so the region rotates and NEW background swings in at the LEFT edge (the goal is this new LEFT-side background; do NOT expect the scene's right content) -> FG: stays centered, rotates to show its RIGHT (left side self-occludes). BG: backdrop sweeps RIGHT. ENTERS: new background at LEFT edge. LEAVES: the region's LEFT side + background at RIGHT edge",
    'orbit_up':       "you need the TOP of the centered near region and what it hides above -> FG: stays centered, tilts to show its TOP (front face foreshortens). BG: backdrop drifts UP, ground opens below. ENTERS: ground around the region at BOTTOM edge. LEAVES: upright front faces + horizon off the top",
    'orbit_down':     "you need the UNDERSIDE of the centered near region and what it hides below -> FG: stays centered, tilts to show its UNDERSIDE (front face foreshortens). BG: backdrop drifts DOWN, ground drops away. ENTERS: area above/behind the region (ceiling/sky) at TOP edge. LEAVES: object TOPS + ground off the bottom",
    # translate the camera, facing unchanged: a sideways/forward scan with parallax
    # (the near region is NOT recentered, so it drifts across the frame).
    'move_forward':   "a target ahead is too small or far to read — close in to inspect it -> FG: enlarges fast and looms, edge objects grow past the frame and exit. BG: enlarges only slightly.",
    'move_backward':  "you need more surrounding context than the frame shows -> FG: shrinks fast toward center. BG: shrinks only slightly. ENTERS: more context at EVERY edge. LEAVES: nothing (frame only gains margin)",
    'move_left':      "a near object hides background on its LEFT — strafe left so it drifts right and uncovers that side -> FG: drifts strongly RIGHT. BG: drifts slightly right. ENTERS: new content at LEFT edge. LEAVES: RIGHT edge (mostly near objects)",
    'move_right':     "a near object hides background on its RIGHT — strafe right so it drifts left and uncovers that side -> FG: drifts strongly LEFT. BG: drifts slightly left. ENTERS: new content at RIGHT edge. LEAVES: LEFT edge (mostly near objects)",
    # vantage: raise/lower the camera (magnitude = height) with the HEADING locked
    # to the reference view (image top = reference forward, left/right preserved);
    # the tilt is auto-aimed at the scene centre and the scene stays centred
    'birds_eye_view': "you need the overall ground layout / spatial arrangement from above, KEEPING the reference view's left/right — image TOP = reference forward -> FG: seen from above, TOP and footprint visible, height foreshortened. BG: ground layout spreads across the frame, far things toward TOP. ENTERS: full ground footprint and the gaps between objects. LEAVES: upright front faces + horizon off the top",
    'worms_eye_view': "you need a low upward view — undersides and exaggerated relative heights, KEEPING the reference view's left/right -> FG: towers overhead, underside visible, height exaggerated. BG: sits low, far things toward BOTTOM. ENTERS: area overhead (sky/ceiling) at TOP edge. LEAVES: ground plane + object TOPS off the bottom",
    # pedestal: raise/lower the camera straight up/down, facing unchanged
    'pedestal_up':    "a near object below occludes content behind it — raise straight up to de-occlude vertically -> FG: slides DOWN strongly. BG: slides DOWN slightly. ENTERS: new area behind the near object. LEAVES: bottom edge of the reference view",
    'pedestal_down':  "a near object above occludes content underneath it — lower straight down to de-occlude vertically -> FG: slides UP strongly. BG: slides UP slightly. ENTERS: new area under the near object. LEAVES: top edge of the reference view",
    # pan/tilt: rotate in place, camera does NOT move -> no parallax, no new depth
    # info; FG and BG move together, only off-frame neighbours come into view.
    'pan_left':       "the target is just off-frame LEFT (outside the view, not occluded) -> ENTERS: off-frame content at LEFT edge. LEAVES: RIGHT-edge content",
    'pan_right':      "the target is just off-frame RIGHT (outside the view, not occluded) -> ENTERS: off-frame content at RIGHT edge. LEAVES: LEFT-edge content",
    'tilt_up':        "the target is just off-frame ABOVE (outside the view, not occluded) -> ENTERS: upper area/ceiling at TOP edge. LEAVES: bottom-edge content",
    'tilt_down':      "the target is just off-frame BELOW (outside the view, not occluded) -> ENTERS: lower area/floor at BOTTOM edge. LEAVES: top-edge content",
    # turn_around: face the opposite side (magnitude ignored)
    'turn_around':    "the target is BEHIND the camera -> FG & BG both leave entirely: whatever was BEHIND fills the frame. LEAVES: the entire reference view (a large change)",
}

_missing = set(POSE_FAMILIES) - set(POSE_FAMILY_DESCRIPTIONS)
assert not _missing, (
    'every active pose_utils.POSE_FAMILIES family needs a POSE_FAMILY_DESCRIPTIONS '
    f'entry; missing: {_missing}'
)
