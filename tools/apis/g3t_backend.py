"""Reconstruct a scene with G3T and reuse its geometry to synthesize views."""
import math
import os
import sys
from typing import List

import torch
from PIL import Image

from tools.apis.reprojection import (
    ReprojectionSynthesizer,
    SYNTH_FOV_DEG,
    gravity_frame,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REPO = os.path.join(_REPO_ROOT, "tools/third_party/g3t")
# Empty => download from Hugging Face.
DEFAULT_CKPT = os.environ.get("VANTAGE_G3T_CKPT", "")


class G3TViewSynthesizer(ReprojectionSynthesizer):
    """Loads G3T once; each call reconstructs the scene and reprojects it."""

    IMG_LOAD_RESOLUTION = 1024   # square-pad/resize size before downscale (G3T default)
    INFERENCE_RESOLUTION = 518   # G3T runs square at 518 (run_inference.py default)
    CONF_DROP_QUANTILE = 0.01    # drop the least-confident points
    DEPTH_EDGE_RTOL = 0.03       # relative depth jump that counts as a depth edge
    MAX_POINTS = 1_500_000       # cap fused cloud size for GPU memory

    def __init__(self, repo: str = DEFAULT_REPO, ckpt: str = DEFAULT_CKPT,
                 device: str = "cuda") -> None:
        self.device = device

        if repo not in sys.path:
            sys.path.insert(0, repo)
        from vggt.models.g3t import G3T

        if ckpt:
            if not os.path.exists(ckpt):
                raise FileNotFoundError(
                    f"G3T checkpoint not found at {ckpt}. Unset VANTAGE_G3T_CKPT to "
                    "auto-download from https://huggingface.co/thatbrguy/g3t, or "
                    "point it at a valid .pt."
                )
            model = G3T(enable_point=True, enable_depth=True,
                        enable_gravity_camera_heads=True)
            state = torch.load(ckpt, map_location="cpu")
            state = state.get("model", state) if isinstance(state, dict) else state
            model.load_state_dict(state, strict=False)
        else:
            model = G3T.from_pretrained("thatbrguy/g3t")
        self.model = model.to(device).eval()

    # ------------------------------------------------------------------ utils
    def _preprocess(self, images: List[Image.Image]):
        """Resize and pad input views for G3T and return their valid-pixel masks."""
        import torch.nn.functional as F
        from torchvision import transforms as TF
        to_tensor = TF.ToTensor()
        L, R = self.IMG_LOAD_RESOLUTION, self.INFERENCE_RESOLUTION
        out, masks = [], []
        for im in images:
            if im.mode == "RGBA":                            # blend onto white
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im)
            im = im.convert("RGB")
            w, h = im.size
            m = max(w, h)
            left, top = (m - w) // 2, (m - h) // 2
            sq = Image.new("RGB", (m, m), (0, 0, 0))         # black square pad
            sq.paste(im, (left, top))
            sq = sq.resize((L, L), Image.Resampling.BICUBIC)
            out.append(to_tensor(sq))
            # valid region: where the image was pasted
            s = R / m
            mask = torch.zeros(R, R, dtype=torch.bool)
            x0, x1 = round(left * s), round((left + w) * s)
            y0, y1 = round(top * s), round((top + h) * s)
            mask[y0:y1, x0:x1] = True
            masks.append(mask)
        imgs = torch.stack(out, 0)                           # (S,3,L,L)
        imgs = F.interpolate(imgs, size=(R, R), mode="bilinear", align_corners=False)
        return imgs, torch.stack(masks, 0)

    # [pose-facts-all b1] keys of the G3T prediction that _build_recon reads.
    _FWD_KEYS = ("local_pose_enc", "global_pose_enc", "world_points",
                 "world_points_conf", "depth")
    FWD_CACHE_MAX = 256

    @torch.inference_mode()
    def _forward(self, images: List[Image.Image]) -> dict:
        """Run and time G3T inference on the input views."""
        import time
        dev = self.device
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        imgs, valid = self._preprocess(images)             # (S,3,H,W) in [0,1], (S,H,W) bool
        imgs, valid = imgs.to(dev), valid.to(dev)
        with torch.amp.autocast("cuda", enabled=str(dev).startswith("cuda"), dtype=torch.bfloat16):
            pred = self.model(images=imgs[None])            # batches to (1,S,...)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return dict(imgs=imgs, valid=valid, pred=pred,
                    forward_s=time.perf_counter() - t0)

    def _cache_put(self, key: str, images: List[Image.Image], fwd: dict) -> None:
        """Store a G3T forward result in the bounded CPU cache."""
        import collections
        if not hasattr(self, "_fwd_cache"):
            self._fwd_cache = collections.OrderedDict()
        pred = {k: fwd["pred"][k].detach().to("cpu") for k in self._FWD_KEYS
                if k in fwd["pred"]}
        self._fwd_cache[key] = dict(
            sizes=[im.size for im in images], imgs=fwd["imgs"].to("cpu"),
            valid=fwd["valid"].to("cpu"), pred=pred, forward_s=fwd["forward_s"])
        self._fwd_cache.move_to_end(key)
        while len(self._fwd_cache) > self.FWD_CACHE_MAX:
            self._fwd_cache.popitem(last=False)

    def _cache_pop(self, key: str, images: List[Image.Image]):
        """Retrieve and remove the cached forward result for the same view dimensions."""
        ent = getattr(self, "_fwd_cache", {}).pop(key, None) if key else None
        if ent is None or ent["sizes"] != [im.size for im in images]:
            return None
        dev = self.device
        return dict(imgs=ent["imgs"].to(dev), valid=ent["valid"].to(dev),
                    pred={k: v.to(dev) for k, v in ent["pred"].items()},
                    forward_s=0.0)

    @torch.inference_mode()
    def reconstruct(self, images: List[Image.Image], anchor_index: int = 0,
                    fwd: dict = None) -> dict:
        """Build the scene using raw G3T camera heads; optionally reuse a forward."""
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import make_4x4, closed_form_inverse_se3
        if not 0 <= anchor_index < len(images):
            raise ValueError(f'Invalid anchor_index={anchor_index}')
        dev = self.device
        performed_forward = fwd is None
        fwd = self._forward(images) if fwd is None else fwd
        imgs, valid, pred = fwd['imgs'], fwd['valid'], fwd['pred']
        S, _, H, W = imgs.shape
        # ---- camera poses: w2c = (g->c) @ (w->g) ----
        g2c, intr = pose_encoding_to_extri_intri(
            pred["local_pose_enc"], (H, W), pose_encoding_type="noT_quaR_FoV")
        w2g, _ = pose_encoding_to_extri_intri(
            pred["global_pose_enc"], (H, W), pose_encoding_type="absT_quaRy_noFoV")
        w2c = (make_4x4(g2c).float() @ make_4x4(w2g).float())[0]   # (S,4,4) world->cam
        c2w = closed_form_inverse_se3(w2c)                         # (S,4,4) cam->world
        cam_centers = c2w[:, :3, 3]                                # (S,3)

        # ---- fused cloud: world points coloured by their pixels ----
        pts = pred["world_points"][0].float().reshape(-1, 3)       # (S*H*W,3)
        cols = imgs.permute(0, 2, 3, 1).reshape(-1, 3).clamp(0, 1)  # (S*H*W,3)
        cfd = pred["world_points_conf"][0].float().reshape(-1)     # (S*H*W,)

        # drop depth-edge pixels (smeared floaters)
        if "depth" in pred:
            depth = pred["depth"][0].float().squeeze(-1)           # (S,H,W)
            edge = self._depth_edge_mask(depth, self.DEPTH_EDGE_RTOL).reshape(-1)
            cfd = cfd.clone()
            cfd[edge] = 0.0

        # per-point source view id
        vid = torch.arange(S, device=dev, dtype=torch.uint8).repeat_interleave(H * W)
        keep = torch.isfinite(pts).all(-1)
        keep &= valid.reshape(-1)                                  # drop black square-padding pixels
        thr = torch.quantile(cfd[keep].float(), self.CONF_DROP_QUANTILE)
        keep &= cfd >= thr
        keep &= cfd > 1e-5                                          # always drop zeroed edges
        pts, cols, vid = pts[keep], cols[keep], vid[keep]
        if len(pts) > self.MAX_POINTS:
            sel = torch.randperm(len(pts), device=dev)[: self.MAX_POINTS]
            pts, cols, vid = pts[sel], cols[sel], vid[sel]

        # ---- anchor camera + gravity frame G (the pose delta acts in G) ----
        # up = world +/-Y, sign taken from the cameras' mean "down".
        R0 = c2w[anchor_index, :3, :3]
        p0 = c2w[anchor_index, :3, 3]
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev)
        cam_down = c2w[:, :3, 1].mean(0)               # mean camera "down" in world
        grav_down = y_axis if torch.dot(cam_down, y_axis) >= 0 else -y_axis
        up_t = -grav_down                              # exact canonical vertical
        G = gravity_frame(R0, up_t)                    # gravity-frame world axes

        # Synthesis camera: the anchor's native size at the fixed SYNTH_FOV_DEG.
        # fov_content (the anchor's own FoV) is used only by the elevation mode.
        Wr, Hr = images[anchor_index].size
        m0 = max(Wr, Hr)
        s0 = self.INFERENCE_RESOLUTION / m0        # native px -> 518-frame px
        f0_px = float(intr[0, anchor_index, 0, 0] + intr[0, anchor_index, 1, 1]) / 2
        content_diag = math.hypot(Wr * s0, Hr * s0)
        fov_content = math.degrees(2 * math.atan(content_diag / (2 * f0_px)))
        K = self._intrinsics_from_fov(getattr(self, "render_fov_deg", SYNTH_FOV_DEG), Wr, Hr, dev)

        # Splat radius from the projected point spacing (clamped 1..4).
        rd = 2.0 * float(K[0, 0]) * math.tan(math.radians(fov_content) / 2)
        splat_r = min(4, max(1, math.ceil(rd / self.INFERENCE_RESOLUTION)))

        spread = (cam_centers - p0).norm(dim=-1).mean().clamp_min(1e-3)
        if S == 1:
            spread = pts.std(0).norm().clamp_min(1e-3)
        # view_dist: the anchor's stand-off, the median distance to the content in
        # the central half of its frame. Falls back to the anchor's full frame,
        # then to the median over views.
        wp_grid = pred["world_points"][0].float()          # (S,H,W,3)
        vds = {}
        for v in range(S):
            mv = vid == v
            if int(mv.sum()) > 100:
                vds[v] = float((pts[mv] - c2w[v, :3, 3]).norm(dim=-1).median())
        view_dist_med = (float(torch.tensor(list(vds.values())).median())
                         if vds else float(spread))
        q4 = H // 4
        ctr = wp_grid[anchor_index, q4:-q4, q4:-q4].reshape(-1, 3)
        cok = (valid[anchor_index, q4:-q4, q4:-q4].reshape(-1)
               & torch.isfinite(ctr).all(-1))
        if int(cok.sum()) > 100:
            view_dist = float((ctr[cok] - p0).norm(dim=-1).median())
            view_dist_source = 'anchor_central_half_median'
        else:
            view_dist = vds.get(anchor_index, view_dist_med)
            view_dist_source = ('anchor_filtered_full_frame_median' if anchor_index in vds
                                else 'median_over_valid_views' if vds else 'camera_spread')
        view_dist_unclamped = float(view_dist)
        view_dist = torch.tensor(view_dist, device=dev).clamp_min(1e-3)
        # scene_scale: horizontal footprint radius (90th percentile from the centroid).
        rel = pts - pts.mean(0)
        hp = rel - torch.matmul(rel, up_t)[:, None] * up_t     # horizontal offsets
        scene_scale = torch.quantile(hp.norm(dim=-1), 0.90).clamp_min(1e-3)
        diag = dict(
            g3t_forward_s=float(fwd.get("forward_s", 0.0)),
            g3t_forward_calls=int(performed_forward), anchor_index=anchor_index,
            n_views=S, n_points=int(pts.shape[0]), fov_content=fov_content,
            fov_fixed=getattr(self, "render_fov_deg", SYNTH_FOV_DEG), backend="g3t",
            splat_r=splat_r, scene_scale=float(scene_scale),
            view_dist=float(view_dist), view_dist_source=view_dist_source,
            view_dist_unclamped=view_dist_unclamped, view_dist_minimum=1e-3,
        )
        return dict(cams=c2w, vid=vid, pts=pts, cols=cols, R0=R0, p0=p0, G=G, up=up_t,
                    spread=spread, scene_scale=scene_scale, view_dist=view_dist,
                    fov_content=fov_content, splat_r=splat_r,
                    K=K, W=Wr, H=Hr, diag=diag)

    @torch.inference_mode()
    def pose_facts(self, images: List[Image.Image], style: str = "v1",
                   forward_key: str = None) -> dict:
        """Estimate camera relations and retain the forward for synthesis."""
        from tools.apis.pose_facts import verbalize_pose_facts
        if len(images) < 2:
            return {'text': '', 'pairs': [], 'n_views': len(images)}
        fwd = self._forward(images)
        recon = self.reconstruct(images, 0, fwd=fwd)
        cams = recon["cams"].float().cpu().numpy()
        up = recon["up"].float().cpu().numpy()
        pts, vid = recon["pts"], recon["vid"]
        dists = []
        for v in range(len(images)):
            mv = vid == v
            if int(mv.sum()) > 100:
                c = recon["cams"][v, :3, 3]
                dists.append(float((pts[mv] - c).norm(dim=-1).median()))
            else:
                dists.append(None)
        out = verbalize_pose_facts(list(cams), up, dists, style=style,
                                   gravity_align=getattr(self, "pose_gravity_align", True))
        out['view_dists'] = dists
        out['backend'] = recon['diag'].get('backend', 'g3t')
        out['n_views'] = len(images)
        out['conf_q50'] = recon["diag"].get("conf_q50")
        out['g3t_forward_s'] = float(fwd["forward_s"])
        out['g3t_forward_calls'] = 1
        if forward_key:
            self._cache_put(forward_key, images, fwd)
        return out

    def synthesize(self, images: List[Image.Image], target_pose, pivot_mode: str = "ahead",
                   anchor_index: int = 0, forward_key: str = None):
        """Render a view at a target pose relative to the selected reference camera."""
        if target_pose is None:
            raise ValueError(
                "G3TViewSynthesizer.synthesize requires an explicit target_pose "
                "(the model decides the 6-DoF); there is no default motion.")
        fwd = self._cache_pop(forward_key, images) if forward_key else None
        recon = self.reconstruct(images, anchor_index, fwd=fwd)
        if forward_key:
            recon["diag"]["fwd_cache_hit"] = fwd is not None
        image, pose_diag = self.synthesize_view(recon, target_pose, pivot_mode)
        return image, {**recon["diag"], **pose_diag}
