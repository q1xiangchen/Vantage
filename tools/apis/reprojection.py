"""Pose-dependent view synthesis: render the reconstructed cloud at a target pose.

The backend's `reconstruct(images)` returns the pose-independent scene state:

    pts (N,3) / cols (N,3)       coloured point cloud
    R0 (3,3) / p0 (3,)           anchor (first input view) rotation / centre
    G (3,3)                      gravity frame the pose delta acts in
    up (3,)                      world up
    spread                       camera-baseline scale (translation units)
    view_dist                    anchor stand-off (standoff guard)
    scene_scale                  horizontal scene radius
    fov_content                  anchor content FoV (deg)
    splat_r                      splat radius
    K (3,3) / W / H              synthesis camera
    diag                         scene diagnostics

`target_pose` is a 4x4 c2w (OpenCV) delta relative to the anchor view.
"""
import math

import numpy as np
import torch
from PIL import Image

# Diagonal FoV of the synthesis camera.
SYNTH_FOV_DEG = 120.0

# --- Elevation (birds/worms_eye) -----------------------------------------------
# The camera rises/drops on the anchor's vertical line, heading kept; the FoV is
# narrowed for scenes viewed from outside.
# Below this coverage at full FoV the pose is `infeasible`.
ELEVATION_COV_FLOOR = 0.55
ELEVATION_CONTENT_R = 1.2                  # content cylinder radius, x scene_scale
# FoV ladder (shrink / ref / widen): the first rung reaching this coverage wins.
ELEVATION_COV_BACKSTOP = 0.5
ELEVATION_FOV_SHRINK = 0.8
ELEVATION_FOV_WIDEN = 1.2
# If full FoV would still gain this much coverage, use full FoV.
ELEVATION_COV_GAP = 0.2
# Minimum FoV.
ELEVATION_FOV_MIN = 32.0
# Aim at the scene centre instead of the anchor's central ray when the ray lands
# farther than this x scene_scale from it.
ELEVATION_AIM_OFF_MAX = 1.0
# The anchor is inside the scene (full FoV) below this x the median horizontal radius.
ELEVATION_NEAR_RATIO = 1.0
# Max look-down/up angle; the pitch increment is scaled by the headroom to it.
ELEVATION_ALPHA_MAX = 80.0

# --- Splat ------------------------------------------------------------------------
# Splat at SUPERSAMPLE x resolution, then downscale (antialiasing).
SUPERSAMPLE = 2.0
# Points within (1+SPLAT_ZTOL) of the nearest depth are averaged.
SPLAT_ZTOL = 0.01

# --- Guards -------------------------------------------------------------------------
# Keep the camera above the floor (1st-percentile height + FLOOR_LIFT x scene_scale).
FLOOR_LIFT = 0.05
# Back the camera off when the framed content is closer than STANDOFF_MIN x the
# anchor's stand-off; skipped when under STANDOFF_MIN_FRAC of the cloud is framed.
STANDOFF_MIN = 0.7
STANDOFF_MIN_FRAC = 0.02


def _so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> rotation vector."""
    cos = ((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2).clamp(-1, 1)
    ang = torch.arccos(cos)
    axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = torch.sin(ang)
    if float(s.abs()) < 1e-6:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    return axis * (ang / (2 * s))


def _so3_exp(v: torch.Tensor) -> torch.Tensor:
    """Rotation vector -> rotation matrix."""
    ang = v.norm()
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    if float(ang) < 1e-9:
        return eye
    k = v / ang
    K = torch.zeros(3, 3, device=v.device, dtype=v.dtype)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]
    return eye + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)


def closed_form_inverse_se3(mats: torch.Tensor) -> torch.Tensor:
    """Batch SE(3) inverse (c2w <-> w2c)."""
    R = mats[..., :3, :3]
    t = mats[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(mats)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ t
    out[..., 3, 3] = 1.0
    return out


def gravity_frame(R0: torch.Tensor, up_t: torch.Tensor) -> torch.Tensor:
    """Gravity frame G with columns [right, down, forward]: down is true vertical,
    forward is the anchor heading on the ground plane."""
    fwd = R0[:, 2]
    fwd_h = fwd - torch.dot(fwd, up_t) * up_t        # anchor heading along the ground
    if fwd_h.norm() < 1e-4:                          # looking straight up/down
        fwd_h = R0[:, 1] - torch.dot(R0[:, 1], up_t) * up_t
    fwd_g = torch.nn.functional.normalize(fwd_h, dim=0)
    down_g = -up_t
    right_g = torch.nn.functional.normalize(torch.linalg.cross(down_g, fwd_g), dim=0)
    fwd_g = torch.linalg.cross(right_g, down_g)      # re-orthonormalize
    return torch.stack([right_g, down_g, fwd_g], dim=1)


class ReprojectionSynthesizer:
    """Pose-dependent synthesis. Subclasses provide `reconstruct()` and `device`."""

    @staticmethod
    def _intrinsics_from_fov(fov_deg: float, W: int, H: int, device) -> torch.Tensor:
        """Centred pinhole K with `fov_deg` over the frame diagonal."""
        diag = math.hypot(W, H)
        f = (diag / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        K = torch.eye(3, device=device)
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = W / 2.0
        K[1, 2] = H / 2.0
        return K

    @staticmethod
    def _depth_edge_mask(depth, rtol: float, k: int = 3):
        """True where the local (k x k) relative depth jump exceeds `rtol`."""
        import torch.nn.functional as F
        d = depth.unsqueeze(1)                              # (S,1,H,W)
        pad = k // 2
        dp = F.pad(d, (pad, pad, pad, pad), mode="replicate")
        dmax = F.max_pool2d(dp, k, stride=1)
        dmin = -F.max_pool2d(-dp, k, stride=1)
        jump = (dmax - dmin) / d.clamp_min(1e-6)
        return (jump > rtol).squeeze(1)                     # (S,H,W)

    def _splat(self, pts_world, cols, viewmat, K, W, H, radius: int):
        """Z-buffer splat of the cloud. Returns (H,W,3) in [0,1], white background."""
        dev = pts_world.device
        R, t = viewmat[:3, :3], viewmat[:3, 3]
        p_cam = (R @ pts_world.T + t[:, None]).T               # (N,3)
        z = p_cam[:, 2]
        front = z > 1e-4
        p_cam, cols, z = p_cam[front], cols[front], z[front]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = (fx * p_cam[:, 0] / z + cx)
        v = (fy * p_cam[:, 1] / z + cy)

        r = int(radius)
        offs = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                if dx * dx + dy * dy <= r * r]
        zbuf = torch.full((H * W,), float("inf"), device=dev)
        blocks = []                                            # (flat, z, colour) per offset
        # pass 1: per-pixel nearest depth across all splat offsets
        for dx, dy in offs:
            ui = (u + dx).round().long()
            vi = (v + dy).round().long()
            ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            flat = (vi * W + ui)[ok]
            zc, cc = z[ok], cols[ok]
            zbuf.scatter_reduce_(0, flat, zc, reduce="amin", include_self=True)
            blocks.append((flat, zc, cc))
        # pass 2: average every point within the depth tolerance of the winner
        out = torch.ones(H * W, 3, device=dev)                 # white background
        acc = torch.zeros(H * W, 3, device=dev)
        cnt = torch.zeros(H * W, device=dev)
        for flat, zc, cc in blocks:
            win = zc <= zbuf[flat] * (1 + SPLAT_ZTOL) + 1e-6
            fw = flat[win]
            acc.index_add_(0, fw, cc[win])
            cnt.index_add_(0, fw, torch.ones_like(fw, dtype=cnt.dtype))
        has = cnt > 0
        out[has] = acc[has] / cnt[has, None]
        return out.reshape(H, W, 3)

    @staticmethod
    def _floor_height(recon: dict) -> torch.Tensor:
        """Lowest allowed camera height (floor + FLOOR_LIFT x scene_scale)."""
        h_floor = torch.quantile(torch.matmul(recon["pts"], recon["up"]), 0.01)
        return h_floor + FLOOR_LIFT * recon["scene_scale"]

    def _floor_guard(self, c2w_tgt: torch.Tensor, recon: dict) -> torch.Tensor:
        """Lift the camera straight up if it is below the floor."""
        up = recon["up"]
        h_min = self._floor_height(recon)
        h_cam = torch.dot(c2w_tgt[:3, 3], up)
        if float(h_cam) < float(h_min):
            c2w_tgt = c2w_tgt.clone()
            c2w_tgt[:3, 3] = c2w_tgt[:3, 3] + (h_min - h_cam) * up
        return c2w_tgt

    def _scene_standoff(self, c2w_tgt: torch.Tensor, recon: dict):
        """Back the camera off along its view axis if the framed content is too
        close (e.g. facing a wall). Returns (c2w, backoff distance)."""
        R, C = c2w_tgt[:3, :3], c2w_tgt[:3, 3]
        pts = recon["pts"]
        pc = (pts - C) @ R                      # rows = R^T (pts - C): points in cam frame
        z = pc[:, 2]
        front = z > 1e-4
        K, W, H = recon["K"], recon["W"], recon["H"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        zc = z.clamp_min(1e-6)
        u = fx * pc[:, 0] / zc + cx
        v = fy * pc[:, 1] / zc + cy
        framed = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        n_framed = int(framed.sum())
        if n_framed < max(1, int(STANDOFF_MIN_FRAC * pts.shape[0])):
            return c2w_tgt, 0.0                 # nothing meaningful in view; leave the pose
        # judge by the central half of the frame
        centre = (framed & (u >= W / 4) & (u < 3 * W / 4)
                  & (v >= H / 4) & (v < 3 * H / 4))
        if int(centre.sum()) < 200:
            return c2w_tgt, 0.0                 # centre looks into empty space
        z_near = z[centre].float().median()
        deficit = (STANDOFF_MIN * recon["view_dist"] - z_near).clamp_min(0.0)
        back = float(deficit)
        if back <= 0.0:
            return c2w_tgt, 0.0
        out = c2w_tgt.clone()
        out[:3, 3] = C - R[:, 2] * back         # dolly back along camera forward (OpenCV +z)
        return out, back

    @torch.inference_mode()
    def synthesize_view(self, recon: dict, target_pose, pivot_mode: str = "ahead"):
        """Render one view from `reconstruct(...)` output at `target_pose`.
        Returns (image, diag).

        pivot_mode:
          - 'ahead': rotate in place, then translate in frame G.
          - 'centroid': orbit about the scene centre at the anchor's height.
          - 'elevation': rise/drop on the anchor's vertical line, heading kept,
            aimed at the scene; the FoV adapts to the scene.
        Then the standoff and floor guards apply.
        """
        dev = self.device
        delta = torch.as_tensor(np.asarray(target_pose, np.float32)).reshape(4, 4).to(dev)
        G, R0, p0 = recon["G"], recon["R0"], recon["p0"]
        R_d = delta[:3, :3]
        c2w_tgt = torch.eye(4, device=dev)
        up = recon["up"]
        elevation = None
        if pivot_mode == "elevation":
            sc = recon["scene_scale"]
            f0 = R0[:, 2]
            down_g = -up
            fh = f0 - torch.dot(f0, up) * up          # anchor heading, horizontal
            if float(fh.norm()) < 1e-3:               # anchor looks (near) straight up/down
                cu = -R0[:, 1]                        # fall back to camera-up as a proxy
                fh = cu - torch.dot(cu, up) * up
            fh = fh / fh.norm().clamp_min(1e-6)
            right = torch.linalg.cross(down_g, fh)    # horizontal right (yaw stays put)
            right = right / right.norm().clamp_min(1e-6)
            # target on the anchor central ray, at the scene's forward distance
            d = torch.dot(recon["pts"].mean(0) - p0, f0).clamp_min(0.25 * sc)
            aim = p0 + f0 * d

            def _elevation_pose(cam: torch.Tensor) -> torch.Tensor:
                """Heading-locked look-at from `cam` toward `aim` (pitch only)."""
                vec = aim - cam
                a = torch.dot(vec, fh).clamp_min(1e-3 * sc)  # keep the aim ahead
                c = torch.dot(vec, down_g)            # vertical offset to the target
                fwd = fh * a + down_g * c             # pitched within the heading plane
                fwd = fwd / fwd.norm().clamp_min(1e-6)
                down = torch.linalg.cross(fwd, right)  # OpenCV y = z x x
                down = down / down.norm().clamp_min(1e-6)
                m = torch.eye(4, device=dev)
                m[:3, :3] = torch.stack([right, down, fwd], dim=1)
                m[:3, 3] = cam
                return m

            h_min = self._floor_height(recon)
            # the pose pitch is an increment on the anchor's own pitch (birds < 0)
            pitch_deg = math.degrees(float(_so3_log(R_d)[0]))
            theta = abs(pitch_deg)
            cen = recon["pts"].median(0).values     # junk-robust scene centre
            off_v = (aim - cen) - torch.dot(aim - cen, up) * up
            aim_off = float(off_v.norm()) / float(sc)
            aim_mode = "ray"
            if aim_off > ELEVATION_AIM_OFF_MAX:
                aim = cen                           # rebind: closure sees it
                aim_mode = "centroid"
                d_h = float(torch.dot(cen - p0, fh).clamp_min(0.75 * sc))
            else:
                d_h = float(torch.dot(aim - p0, fh).clamp_min(1e-3 * sc))
            dz = float(torch.dot(p0 - aim, up))     # anchor height above aim
            # total angle: increment scaled by the headroom to ELEVATION_ALPHA_MAX
            alpha0 = math.degrees(math.atan2(dz, d_h))
            if pitch_deg < 0:
                room = min(max((ELEVATION_ALPHA_MAX - alpha0)
                               / ELEVATION_ALPHA_MAX, 0.0), 1.0)
                alpha = alpha0 + theta * room
            else:
                room = min(max((alpha0 + ELEVATION_ALPHA_MAX)
                               / ELEVATION_ALPHA_MAX, 0.0), 1.0)
                alpha = alpha0 - theta * room
            if aim_mode == "centroid":
                # behind the centre along the heading
                h = math.tan(math.radians(alpha)) * d_h
                cam = aim - fh * d_h + h * up
            else:
                # on the anchor's vertical line
                cam = p0 + (math.tan(math.radians(alpha)) * d_h - dz) * up
            # floor guard before aiming
            h_cam = torch.dot(cam, up)
            if float(h_cam) < float(h_min):
                cam = cam + (h_min - h_cam) * up
            c2w_tgt = _elevation_pose(cam)
            # ---- FoV: coverage of the content near the aim ----
            rel = recon["pts"] - aim
            horiz = rel - torch.outer(rel @ up, up)
            pts_c = recon["pts"][horiz.norm(dim=-1)
                                 <= ELEVATION_CONTENT_R * float(sc)]
            Rt, Ct = c2w_tgt[:3, :3], c2w_tgt[:3, 3]
            pc = (pts_c - Ct) @ Rt
            zc_ = pc[:, 2].clamp_min(1e-6)
            xn, yn = pc[:, 0] / zc_, pc[:, 1] / zc_
            front = pc[:, 2] > 1e-4
            W_, H_ = recon["W"], recon["H"]
            half_diag = math.hypot(W_, H_) / 2.0

            def cov(fov_deg: float) -> float:
                if len(pts_c) == 0:
                    return 0.0
                f = half_diag / math.tan(math.radians(fov_deg) / 2)
                ok = (front & (xn.abs() * f <= W_ / 2)
                      & (yn.abs() * f <= H_ / 2))
                return float(ok.float().mean())

            cov_max = cov(SYNTH_FOV_DEG)
            infeasible = cov_max < ELEVATION_COV_FLOOR
            # anchor inside the scene -> full FoV; outside -> FoV ladder
            dvec = p0 - cen
            hd = float((dvec - torch.dot(dvec, up) * up).norm())
            rel_c = recon["pts"] - recon["pts"].mean(0)
            hr = (rel_c - torch.outer(rel_c @ up, up)).norm(dim=-1)
            q50 = float(torch.quantile(hr, 0.5).clamp_min(1e-3))
            d_cen = hd / q50
            if infeasible or d_cen < ELEVATION_NEAR_RATIO:
                fov_sel = SYNTH_FOV_DEG
            else:
                fov_ref = float(recon["fov_content"])
                for f in (fov_ref * ELEVATION_FOV_SHRINK, fov_ref,
                          fov_ref * ELEVATION_FOV_WIDEN):
                    fov_sel = min(max(f, ELEVATION_FOV_MIN),
                                  SYNTH_FOV_DEG)
                    if cov(fov_sel) >= ELEVATION_COV_BACKSTOP:
                        break
                if cov_max - cov(fov_sel) > ELEVATION_COV_GAP:
                    fov_sel = SYNTH_FOV_DEG
            K_sel = self._intrinsics_from_fov(fov_sel, W_, H_, dev)
            # splat radius scales with sqrt of the magnification
            ratio = float(K_sel[0, 0]) / float(recon["K"][0, 0])
            splat_r = max(1, min(4, round(recon["splat_r"] * math.sqrt(ratio))))
            recon = {**recon, "K": K_sel, "splat_r": splat_r}
            elevation = dict(
                theta=theta, alpha=alpha, ref_pitch=alpha0,
                fov=fov_sel, coverage=cov(fov_sel), cov_max=cov_max,
                infeasible=infeasible, aim_centred=(aim_mode == "centroid"),
                height=float(torch.dot(c2w_tgt[:3, 3] - p0, up)),
            )
        else:
            # clamp the pitch at straight up/down so the view never flips over
            cos_d = (R_d[0, 0] + R_d[1, 1] + R_d[2, 2] - 1) / 2
            if float(cos_d) < -1 + 1e-4:
                # ~180 deg (turn_around): _so3_log is singular, apply directly
                R_world = G @ R_d @ G.T
            else:
                rv = _so3_log(R_d)
                eu0 = torch.asin(torch.dot(R0[:, 2], up).clamp(-1, 1))
                rx = torch.clamp(eu0 + rv[0], -math.pi / 2, math.pi / 2) - eu0
                R_world = G @ _so3_exp(torch.stack([rx, rv[1], rv[2]])) @ G.T
            c2w_tgt[:3, :3] = R_world @ R0
            if pivot_mode == "centroid":
                # pivot: cloud centroid at the anchor's height
                c_cloud = recon["pts"].mean(0)
                C = c_cloud + torch.dot(p0 - c_cloud, up) * up
                c2w_tgt[:3, 3] = C + R_world @ (p0 - C)
            else:
                # 'ahead': translate in frame G
                c2w_tgt[:3, 3] = p0 + G @ (delta[:3, 3] * recon["spread"])
        # guards: standoff, then floor (last, so it has the final say on height)
        c2w_tgt, standoff = self._scene_standoff(c2w_tgt, recon)
        c2w_tgt = self._floor_guard(c2w_tgt, recon)
        diag = {"pivot_mode": pivot_mode, "standoff": standoff}
        if elevation is not None:
            diag.update({f"elevation_{k}": v for k, v in elevation.items()})
        viewmat = closed_form_inverse_se3(c2w_tgt[None])[0]  # (4,4) world->cam
        # supersample, then downscale to the native frame
        K_r = recon["K"].clone()
        W_r = int(round(recon["W"] * SUPERSAMPLE))
        H_r = int(round(recon["H"] * SUPERSAMPLE))
        K_r[0, 0] = K_r[0, 0] * SUPERSAMPLE
        K_r[1, 1] = K_r[1, 1] * SUPERSAMPLE
        K_r[0, 2] = W_r / 2.0
        K_r[1, 2] = H_r / 2.0
        radius = int(round(int(recon["splat_r"]) * SUPERSAMPLE))
        img = self._splat(recon["pts"], recon["cols"], viewmat, K_r, W_r, H_r,
                          radius=radius)
        arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        out = Image.fromarray(arr).resize((recon["W"], recon["H"]), Image.Resampling.LANCZOS)
        return out, diag
