"""
Stitch magnifier tiles using feature matching and several blend modes.

See README.md for setup, examples, and full option list.

Pipeline: load tiles (API or folder) -> crop interior -> estimate neighbor
strides (AKAZE/SIFT/ORB, optional phase refine) -> blend -> optional sharpen.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests


DetectorName = Literal["akaze", "sift", "orb"]
BlendName = Literal["mean", "feather", "multiband", "seam"]

# Extensions we try when resolving files under --tiles-dir
TILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class MosaicRunConfig:
    """Everything run_mosaic() needs; built from CLI args or the Gradio UI."""

    base: str = ""
    params: str = ""
    crop: tuple[int, int, int, int] = (36, 36, 153, 153)
    xs: list[int] | None = None
    ys: list[int] | None = None
    tiles_dir: Path | str | None = None
    tile_pattern: str = "auto"
    tiles_precropped: bool = False
    nominal_dx: float | None = None
    nominal_dy: float | None = None
    strides: Literal["matched", "nominal"] = "matched"
    refine_phase: bool = False
    phase_max_adj: float = 6.0
    min_overlap: int = 16
    detector: DetectorName = "akaze"
    match_ratio: float = 0.75
    ransac_px: float = 3.0
    roi_frac: float = 0.55
    blend: BlendName = "multiband"
    feather_px: int = 28
    seam_smooth: int = 2
    multiband_levels: int = 4
    sharpen: float = 0.0
    sharpen_sigma: float = 1.2
    timeout: float = 30.0
    delay: float = 0.0
    workers: int = 4


def run_mosaic(cfg: MosaicRunConfig) -> tuple[np.ndarray, str]:
    """Load tiles, align, blend. Returns BGR image and a text log for the console/UI."""
    lines: list[str] = []
    xs, ys = cfg.xs or [], cfg.ys or []
    if not xs or not ys:
        raise ValueError("Empty X or Y range")

    n_tiles = len(xs) * len(ys)
    if cfg.tiles_dir:
        root = Path(cfg.tiles_dir)
        lines.append(f"Loading {len(xs)} x {len(ys)} = {n_tiles} tiles from {root} ...")
        tiles = load_tiles_from_folder(
            root,
            xs,
            ys,
            cfg.crop,
            cfg.tile_pattern,
            cfg.tiles_precropped,
            cfg.workers,
        )
    else:
        if not (cfg.params or "").strip():
            raise ValueError("params required when loading from API (omit only with tiles_dir)")
        session = requests.Session()
        session.headers.update({"User-Agent": "imageStitch-feature/1.0", "Accept": "image/*,*/*;q=0.8"})
        lines.append(f"Downloading {len(xs)} x {len(ys)} = {n_tiles} tiles ...")
        tiles = download_tiles(
            session,
            cfg.base,
            cfg.params,
            xs,
            ys,
            cfg.crop,
            cfg.timeout,
            cfg.delay,
            cfg.workers,
        )
    tw = tiles[(0, 0)].shape[1]
    th = tiles[(0, 0)].shape[0]
    ncols, nrows = len(xs), len(ys)

    n_blank = count_likely_blank_tiles(tiles, ncols, nrows)
    if n_blank > ncols * nrows // 4:
        lines.append(
            f"Warning: {n_blank}/{ncols * nrows} cropped tiles look nearly blank (uniform bright). "
            "The API often returns empty tiles when X/Y are outside the photo. "
            "Try --anchor-x 0 --anchor-y 0, or a smaller / different X/Y range from the viewer."
        )

    nom_dx = cfg.nominal_dx if cfg.nominal_dx is not None else max(32.0, min(float(tw) - 4.0, float(tw) * 0.38))
    nom_dy = cfg.nominal_dy if cfg.nominal_dy is not None else nom_dx
    det: DetectorName = cfg.detector  # type: ignore[assignment]

    if cfg.strides == "nominal":
        hx = [float(nom_dx)] * max(0, ncols - 1)
        vy = [float(nom_dy)] * max(0, nrows - 1)
        lines.append("Using nominal strides only (--strides nominal); skipping feature/phase estimation.")
    else:
        lines.append("Estimating horizontal strides (features) ...")
        hx = refine_horizontal_strides(tiles, ncols, nrows, nom_dx, det, cfg.match_ratio, cfg.ransac_px, cfg.roi_frac)
        lines.append("Estimating vertical strides ...")
        vy = refine_vertical_strides(tiles, ncols, nrows, nom_dy, det, cfg.match_ratio, cfg.ransac_px, cfg.roi_frac)
        if cfg.refine_phase:
            lines.append("Phase-correlation refinement ...")
            hx, vy = refine_strides_phase_global(tiles, hx, vy, ncols, nrows, tw, th, cfg.phase_max_adj)

    hx, vy, capped = cap_strides_for_overlap(hx, vy, tw, th, cfg.min_overlap)
    if capped:
        lines.append(
            f"Note: strides capped so overlap is at least {cfg.min_overlap}px "
            f"(max stride {tw - cfg.min_overlap}x{th - cfg.min_overlap} for {tw}x{th} tiles)."
        )

    if ncols > 1:
        lines.append("Horizontal strides: " + " ".join(f"{v:.2f}" for v in hx))
    if nrows > 1:
        lines.append("Vertical strides: " + " ".join(f"{v:.2f}" for v in vy))

    if hx and abs(float(np.mean(hx)) - float(nom_dx)) > 3.0:
        lines.append(
            "Warning: mean horizontal stride differs from --nominal-dx by >3px; "
            "misalignment often looks blurry. Try dropping --refine-phase, or set "
            "--multiband-levels 2 --blend feather, or tune --nominal-dx/--phase-max-adj."
        )
    if vy and abs(float(np.mean(vy)) - float(nom_dy)) > 3.0:
        lines.append("Warning: mean vertical stride differs from --nominal-dy by >3px.")

    xpos = prefix_positions(hx, ncols) if ncols > 1 else np.array([0.0])
    ypos = prefix_positions(vy, nrows) if nrows > 1 else np.array([0.0])
    est_w = int(np.ceil(float(xpos[-1]) + tw)) if ncols else tw
    est_h = int(np.ceil(float(ypos[-1]) + th)) if nrows else th
    lines.append(f"Estimated mosaic size: {est_w} x {est_h} px (blend={cfg.blend})")

    blend: BlendName = cfg.blend  # type: ignore[assignment]
    if blend == "mean":
        mosaic = blend_mosaic_mean(tiles, ncols, nrows, xpos, ypos)
    elif blend == "feather":
        mosaic = blend_mosaic_feather(tiles, ncols, nrows, xpos, ypos, cfg.feather_px)
    elif blend == "multiband":
        mosaic = blend_mosaic_multiband(tiles, ncols, nrows, xpos, ypos, cfg.multiband_levels, hx, vy)
    else:
        mosaic = blend_mosaic_seam(tiles, ncols, nrows, xpos, ypos, cfg.feather_px, cfg.seam_smooth)

    mosaic = ensure_u8_bgr(mosaic)
    if cfg.sharpen > 0:
        mosaic = unsharp_bgr(mosaic, cfg.sharpen_sigma, cfg.sharpen)

    return mosaic, "\n".join(lines)


@dataclass(frozen=True)
class TileJob:
    api_x: int
    api_y: int
    ix: int
    iy: int


def frange(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("step must be positive")
    out: list[int] = []
    v = start
    if start <= stop:
        while v <= stop:
            out.append(v)
            v += step
    else:
        while v >= stop:
            out.append(v)
            v -= step
    return out


def build_url(base: str, params: str, x: int, y: int, cache_bust: bool) -> str:
    sep = "&" if ("?" in base) else "?"
    q = f"X={x}&Y={y}"
    if params.strip():
        q += "&" + params.strip().lstrip("&")
    if cache_bust:
        q += f"&rand={random.random()}"
    return f"{base}{sep}{q}"


def fetch_bgr(session: requests.Session, url: str, timeout: float) -> np.ndarray:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    buf = np.frombuffer(r.content, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Could not decode image from {url[:80]}...")
    return im


def crop_inner_bgr(
    bgr: np.ndarray,
    left: int,
    upper: int,
    right: int,
    lower: int,
) -> np.ndarray:
    return bgr[upper:lower, left:right].copy()


def make_detector(name: DetectorName):
    if name == "akaze":
        return cv2.AKAZE_create()
    if name == "sift":
        return cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.03)
    if name == "orb":
        return cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, patchSize=31)
    raise ValueError(name)


def _ratio_match(desc1: np.ndarray, desc2: np.ndarray, ratio: float) -> list[cv2.DMatch]:
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    norm = cv2.NORM_HAMMING if desc1.dtype == np.uint8 else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(desc1, desc2, k=2)
    good: list[cv2.DMatch] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def estimate_pair_shift(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    detector_name: DetectorName,
    ratio: float,
    ransac_px: float,
    roi_frac: float,
) -> tuple[float, float] | None:
    h, w = left_bgr.shape[:2]
    x_cut = max(8, int(w * roi_frac))
    gray_l = cv2.cvtColor(left_bgr[:, w - x_cut :], cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_bgr[:, :x_cut], cv2.COLOR_BGR2GRAY)

    det = make_detector(detector_name)
    k1, d1 = det.detectAndCompute(gray_l, None)
    k2, d2 = det.detectAndCompute(gray_r, None)
    if k1 is None or k2 is None or d1 is None or d2 is None:
        return None

    x_off_l = float(w - x_cut)
    good = _ratio_match(d1, d2, ratio)
    if len(good) < 8:
        return None
    pts_r = np.float32([k2[m.trainIdx].pt for m in good])
    pts_l = np.float32([k1[m.queryIdx].pt for m in good])
    pts_l[:, 0] += x_off_l

    M, inliers = cv2.estimateAffinePartial2D(
        pts_r,
        pts_l,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_px,
        maxIters=5000,
        confidence=0.995,
    )
    if M is None:
        return None
    inl = inliers.ravel() == 1 if inliers is not None else np.zeros(len(pts_r), dtype=bool)
    if inl.sum() < 6:
        return None
    pr = pts_r[inl]
    pl = pts_l[inl]
    dx = float(np.median(pl[:, 0] - pr[:, 0]))
    dy = float(np.median(pl[:, 1] - pr[:, 1]))
    if not (2.0 < dx < float(w) - 2.0):
        return None
    if abs(dy) > 0.35 * h:
        return None
    return dx, dy


def estimate_pair_shift_vertical(
    top_bgr: np.ndarray,
    bottom_bgr: np.ndarray,
    detector_name: DetectorName,
    ratio: float,
    ransac_px: float,
    roi_frac: float,
) -> tuple[float, float] | None:
    h, w = top_bgr.shape[:2]
    y_cut = max(8, int(h * roi_frac))
    gray_t = cv2.cvtColor(top_bgr[h - y_cut :, :], cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(bottom_bgr[:y_cut, :], cv2.COLOR_BGR2GRAY)

    det = make_detector(detector_name)
    k1, d1 = det.detectAndCompute(gray_t, None)
    k2, d2 = det.detectAndCompute(gray_b, None)
    if k1 is None or k2 is None or d1 is None or d2 is None:
        return None

    y_off_t = float(h - y_cut)
    good = _ratio_match(d1, d2, ratio)
    if len(good) < 8:
        return None
    pts_r = np.float32([k2[m.trainIdx].pt for m in good])
    pts_l = np.float32([k1[m.queryIdx].pt for m in good])
    pts_l[:, 1] += y_off_t

    M, inliers = cv2.estimateAffinePartial2D(
        pts_r,
        pts_l,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_px,
        maxIters=5000,
        confidence=0.995,
    )
    if M is None:
        return None
    inl = inliers.ravel() == 1 if inliers is not None else np.zeros(len(pts_r), dtype=bool)
    if inl.sum() < 6:
        return None
    pr = pts_r[inl]
    pl = pts_l[inl]
    dx = float(np.median(pl[:, 0] - pr[:, 0]))
    dy = float(np.median(pl[:, 1] - pr[:, 1]))
    if not (2.0 < dy < float(h) - 2.0):
        return None
    if abs(dx) > 0.35 * w:
        return None
    return dx, dy


def _median_or(vals: list[float], fallback: float) -> float:
    if not vals:
        return fallback
    s = sorted(vals)
    return s[len(s) // 2]


def refine_horizontal_strides(
    tiles: dict[tuple[int, int], np.ndarray],
    ncols: int,
    nrows: int,
    nominal: float,
    detector: DetectorName,
    ratio: float,
    ransac_px: float,
    roi_frac: float,
) -> list[float]:
    strides: list[float] = []
    for ix in range(ncols - 1):
        row_vals: list[float] = []
        for iy in range(nrows):
            sh = estimate_pair_shift(
                tiles[(ix, iy)],
                tiles[(ix + 1, iy)],
                detector,
                ratio,
                ransac_px,
                roi_frac,
            )
            if sh is not None:
                row_vals.append(sh[0])
        strides.append(_median_or(row_vals, nominal))
    return strides


def refine_vertical_strides(
    tiles: dict[tuple[int, int], np.ndarray],
    ncols: int,
    nrows: int,
    nominal: float,
    detector: DetectorName,
    ratio: float,
    ransac_px: float,
    roi_frac: float,
) -> list[float]:
    strides: list[float] = []
    for iy in range(nrows - 1):
        col_vals: list[float] = []
        for ix in range(ncols):
            sv = estimate_pair_shift_vertical(
                tiles[(ix, iy)],
                tiles[(ix, iy + 1)],
                detector,
                ratio,
                ransac_px,
                roi_frac,
            )
            if sv is not None:
                col_vals.append(sv[1])
        strides.append(_median_or(col_vals, nominal))
    return strides


def phase_refine_horizontal_stride(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    stride: float,
    tw: int,
    th: int,
    max_adj: float,
) -> float:
    """Subpixel refinement of horizontal stride using windowed phase correlation."""
    ov = int(round(tw - stride))
    ov = max(16, min(ov, tw - 4))
    wpat = min(ov + 32, tw)
    La = left_bgr[:, -wpat:].astype(np.float32)
    Rb = right_bgr[:, :wpat].astype(np.float32)
    ga = cv2.cvtColor(La, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(Rb, cv2.COLOR_BGR2GRAY)
    ga -= float(np.mean(ga))
    gb -= float(np.mean(gb))
    win = cv2.createHanningWindow((wpat, th), cv2.CV_32F)
    ga = (ga * win).astype(np.float32)
    gb = (gb * win).astype(np.float32)
    (sx, _sy), resp = cv2.phaseCorrelate(ga, gb)
    if resp < 0.06:
        return stride
    adj = float(np.clip(sx, -max_adj, max_adj))
    return float(np.clip(stride - adj, 24.0, tw - 2.0))


def phase_refine_vertical_stride(
    top_bgr: np.ndarray,
    bottom_bgr: np.ndarray,
    stride: float,
    tw: int,
    th: int,
    max_adj: float,
) -> float:
    ov = int(round(th - stride))
    ov = max(16, min(ov, th - 4))
    hpat = min(ov + 32, th)
    Ta = top_bgr[-hpat:, :].astype(np.float32)
    Bb = bottom_bgr[:hpat, :].astype(np.float32)
    ga = cv2.cvtColor(Ta, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(Bb, cv2.COLOR_BGR2GRAY)
    ga -= float(np.mean(ga))
    gb -= float(np.mean(gb))
    win = cv2.createHanningWindow((tw, hpat), cv2.CV_32F)
    ga = (ga * win).astype(np.float32)
    gb = (gb * win).astype(np.float32)
    (_sx, sy), resp = cv2.phaseCorrelate(ga, gb)
    if resp < 0.06:
        return stride
    adj = float(np.clip(sy, -max_adj, max_adj))
    return float(np.clip(stride - adj, 24.0, th - 2.0))


def refine_strides_phase_global(
    tiles: dict[tuple[int, int], np.ndarray],
    hx: list[float],
    vy: list[float],
    ncols: int,
    nrows: int,
    tw: int,
    th: int,
    max_adj: float,
) -> tuple[list[float], list[float]]:
    hx2 = list(hx)
    for ix in range(ncols - 1):
        vals: list[float] = []
        for iy in range(nrows):
            v = phase_refine_horizontal_stride(
                tiles[(ix, iy)], tiles[(ix + 1, iy)], hx[ix], tw, th, max_adj
            )
            vals.append(v)
        hx2[ix] = _median_or(vals, hx[ix])
    vy2 = list(vy)
    for iy in range(nrows - 1):
        vals = []
        for ix in range(ncols):
            v = phase_refine_vertical_stride(
                tiles[(ix, iy)], tiles[(ix, iy + 1)], vy[iy], tw, th, max_adj
            )
            vals.append(v)
        vy2[iy] = _median_or(vals, vy[iy])
    return hx2, vy2


def prefix_positions(strides: list[float], n: int) -> np.ndarray:
    pos = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        pos[i] = pos[i - 1] + strides[i - 1]
    return pos


def cap_strides_for_overlap(
    hx: list[float],
    vy: list[float],
    tw: int,
    th: int,
    min_overlap: int,
) -> tuple[list[float], list[float], bool]:
    """
    Strides must be < tile size or tiles do not overlap and the canvas stays mostly
    empty (MultiBandBlender leaves gaps black). Cap stride to tw - min_overlap.
    """
    cap_x = float(max(4, tw - min_overlap))
    cap_y = float(max(4, th - min_overlap))
    changed = False
    hx2: list[float] = []
    for s in hx:
        sc = min(float(s), cap_x)
        if sc < float(s) - 0.01:
            changed = True
        hx2.append(sc)
    vy2: list[float] = []
    for s in vy:
        sc = min(float(s), cap_y)
        if sc < float(s) - 0.01:
            changed = True
        vy2.append(sc)
    return hx2, vy2, changed


def count_likely_blank_tiles(
    tiles: dict[tuple[int, int], np.ndarray],
    ncols: int,
    nrows: int,
) -> int:
    """Near-uniform bright tiles (common when API X/Y are outside the photo)."""
    n_blank = 0
    for iy in range(nrows):
        for ix in range(ncols):
            g = cv2.cvtColor(tiles[(ix, iy)], cv2.COLOR_BGR2GRAY)
            m, sd = float(np.mean(g)), float(np.std(g))
            if m > 248.0 and sd < 10.0:
                n_blank += 1
    return n_blank


def overlap_width(tw: int, stride: float) -> int:
    return max(1, int(round(tw - stride)))


def dp_seam_vertical_weights(
    left_strip: np.ndarray,
    right_strip: np.ndarray,
    smooth_px: int = 2,
) -> np.ndarray:
    """
    Vertical seam in horizontal overlap: returns weight for LEFT image, shape (H, W),
    values in [0,1]. W = overlap width; strips already aligned column-wise.
    """
    if left_strip.shape != right_strip.shape:
        raise ValueError("Strips must match for seam DP")
    h, w = left_strip.shape[:2]
    cost = np.linalg.norm(left_strip.astype(np.float32) - right_strip.astype(np.float32), axis=2)
    INF = 1e12
    dp = np.full((h, w), INF, dtype=np.float64)
    prev_x = np.zeros((h, w), dtype=np.int32)
    dp[0, :] = cost[0, :]
    for y in range(1, h):
        for x in range(w):
            best = INF
            bx = x
            for dx in (-1, 0, 1):
                px = x + dx
                if 0 <= px < w:
                    v = dp[y - 1, px] + cost[y, x]
                    if v < best:
                        best = v
                        bx = px
            dp[y, x] = best
            prev_x[y, x] = bx
    seam = np.zeros(h, dtype=np.int32)
    x = int(np.argmin(dp[h - 1]))
    seam[h - 1] = x
    for y in range(h - 2, -1, -1):
        x = prev_x[y + 1, x]
        seam[y] = x
    xx = np.arange(w, dtype=np.float64)[None, :]
    sm = max(float(smooth_px), 1.0)
    wL = 1.0 - np.clip((xx - (seam[:, None] - sm)) / (2.0 * sm + 1e-6), 0.0, 1.0)
    return wL.astype(np.float64)


def dp_seam_horizontal_weights(
    top_strip: np.ndarray,
    bottom_strip: np.ndarray,
    smooth_px: int = 2,
) -> np.ndarray:
    """Horizontal seam in vertical overlap: weight for TOP image, shape (H, W)."""
    if top_strip.shape != bottom_strip.shape:
        raise ValueError("Strips must match")
    h, w = top_strip.shape[:2]
    cost = np.linalg.norm(top_strip.astype(np.float32) - bottom_strip.astype(np.float32), axis=2)
    INF = 1e12
    dp = np.full((h, w), INF, dtype=np.float64)
    prev_y = np.zeros((h, w), dtype=np.int32)
    dp[:, 0] = cost[:, 0]
    for x in range(1, w):
        for y in range(h):
            best = INF
            by = y
            for dy in (-1, 0, 1):
                py = y + dy
                if 0 <= py < h:
                    v = dp[py, x - 1] + cost[y, x]
                    if v < best:
                        best = v
                        by = py
            dp[y, x] = best
            prev_y[y, x] = by
    seam = np.zeros(w, dtype=np.int32)
    y = int(np.argmin(dp[:, w - 1]))
    seam[w - 1] = y
    for x in range(w - 2, -1, -1):
        y = prev_y[seam[x + 1], x + 1]
        seam[x] = y
    yy = np.arange(h, dtype=np.float64)[:, None]
    sm = max(float(smooth_px), 1.0)
    wT = 1.0 - np.clip((yy - (seam[None, :] - sm)) / (2.0 * sm + 1e-6), 0.0, 1.0)
    return wT.astype(np.float64)


def blend_mosaic_mean(
    tiles: dict[tuple[int, int], np.ndarray],
    ncols: int,
    nrows: int,
    xpos: np.ndarray,
    ypos: np.ndarray,
) -> np.ndarray:
    tw = tiles[(0, 0)].shape[1]
    th = tiles[(0, 0)].shape[0]
    cw = int(np.ceil(float(xpos[-1]) + tw + 4))
    ch = int(np.ceil(float(ypos[-1]) + th + 4))
    acc = np.zeros((ch, cw, 3), dtype=np.float64)
    wgt = np.zeros((ch, cw), dtype=np.float64)
    for iy in range(nrows):
        for ix in range(ncols):
            im = tiles[(ix, iy)].astype(np.float64)
            x0 = int(np.floor(xpos[ix]))
            y0 = int(np.floor(ypos[iy]))
            h, w = im.shape[:2]
            acc[y0 : y0 + h, x0 : x0 + w] += im
            wgt[y0 : y0 + h, x0 : x0 + w] += 1.0
    wgt = np.maximum(wgt, 1e-6)
    return (acc / wgt[..., np.newaxis]).clip(0, 255).astype(np.uint8)


def _feather_tile_weight(h: int, w: int, feather: int) -> np.ndarray:
    m = np.ones((h, w), dtype=np.uint8)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float64)
    return np.clip(d / max(float(feather), 1.0), 0.0, 1.0)


def blend_mosaic_feather(
    tiles: dict[tuple[int, int], np.ndarray],
    ncols: int,
    nrows: int,
    xpos: np.ndarray,
    ypos: np.ndarray,
    feather_px: int,
) -> np.ndarray:
    tw = tiles[(0, 0)].shape[1]
    th = tiles[(0, 0)].shape[0]
    cw = int(np.ceil(float(xpos[-1]) + tw + 4))
    ch = int(np.ceil(float(ypos[-1]) + th + 4))
    acc = np.zeros((ch, cw, 3), dtype=np.float64)
    wsum = np.zeros((ch, cw), dtype=np.float64)
    fw = _feather_tile_weight(th, tw, feather_px)
    for iy in range(nrows):
        for ix in range(ncols):
            im = tiles[(ix, iy)].astype(np.float64)
            x0 = int(np.floor(xpos[ix]))
            y0 = int(np.floor(ypos[iy]))
            h, w = im.shape[:2]
            wt = fw[:, :, np.newaxis]
            acc[y0 : y0 + h, x0 : x0 + w] += im * wt
            wsum[y0 : y0 + h, x0 : x0 + w] += fw
    wsum = np.maximum(wsum, 1e-6)
    return (acc / wsum[..., np.newaxis]).clip(0, 255).astype(np.uint8)


def blend_mosaic_multiband(
    tiles: dict[tuple[int, int], np.ndarray],
    ncols: int,
    nrows: int,
    xpos: np.ndarray,
    ypos: np.ndarray,
    bands: int,
    hx: list[float],
    vy: list[float],
) -> np.ndarray:
    tw = tiles[(0, 0)].shape[1]
    th = tiles[(0, 0)].shape[0]
    safe_b = multiband_safe_bands(bands, tw, th, hx, vy)
    if safe_b < bands:
        ovh = min(max(1.0, tw - s) for s in hx) if hx else float(tw)
        ovv = min(max(1.0, th - s) for s in vy) if vy else float(th)
        print(
            f"Note: multiband bands {bands} -> {safe_b} (min overlap ~{min(ovh, ovv):.0f}px; extra bands blur JPEG seams).",
            file=sys.stderr,
        )
    corners: list[tuple[int, int]] = []
    sizes: list[tuple[int, int]] = []
    for iy in range(nrows):
        for ix in range(ncols):
            xi = int(round(float(xpos[ix])))
            yi = int(round(float(ypos[iy])))
            corners.append((xi, yi))
            sizes.append((tw, th))
    roi = cv2.detail.resultRoi(corners=corners, sizes=sizes)
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(safe_b)
    blender.prepare(roi)
    full_mask = np.full((th, tw), 255, dtype=np.uint8)
    for iy in range(nrows):
        for ix in range(ncols):
            im = tiles[(ix, iy)]
            xi = int(round(float(xpos[ix])))
            yi = int(round(float(ypos[iy])))
            blender.feed(cv2.UMat(im), cv2.UMat(full_mask), (xi, yi))
    res, _ = blender.blend(cv2.UMat(0), cv2.UMat(0))
    return ensure_u8_bgr(res.get())


def ensure_u8_bgr(img: np.ndarray) -> np.ndarray:
    """MultiBandBlender (and some paths) yield float; JPEG needs clean uint8 BGR."""
    if img is None:
        raise ValueError("empty image")
    if img.dtype == np.uint8:
        return img
    if img.dtype in (np.float32, np.float64):
        return np.clip(img, 0.0, 255.0).astype(np.uint8)
    return cv2.convertScaleAbs(img)


def multiband_safe_bands(
    requested: int,
    tw: int,
    th: int,
    hx: list[float],
    vy: list[float],
) -> int:
    """Cap Laplacian bands: too many relative to overlap smears detail."""
    ovh = min(max(1.0, tw - s) for s in hx) if hx else float(tw)
    ovv = min(max(1.0, th - s) for s in vy) if vy else float(th)
    ov = max(8.0, min(ovh, ovv))
    # ~1 band per ~20px of overlap keeps seams from going mushy on JPEG tiles
    cap = max(1, min(6, int(ov // 20)))
    return max(1, min(requested, cap, 8))


def unsharp_bgr(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0.0)


def blend_mosaic_seam(
    tiles: dict[tuple[int, int], np.ndarray],
    ncols: int,
    nrows: int,
    xpos: np.ndarray,
    ypos: np.ndarray,
    feather_px: int,
    seam_smooth: int,
) -> np.ndarray:
    """
    Per-overlap DP seam (vertical boundaries then horizontal), then distance-feather
    weights multiplied into a weighted average.
    """
    tw = tiles[(0, 0)].shape[1]
    th = tiles[(0, 0)].shape[0]
    cw = int(np.ceil(float(xpos[-1]) + tw + 4))
    ch = int(np.ceil(float(ypos[-1]) + th + 4))
    weights: dict[tuple[int, int], np.ndarray] = {}
    for iy in range(nrows):
        for ix in range(ncols):
            weights[(ix, iy)] = _feather_tile_weight(th, tw, feather_px)

    for iy in range(nrows):
        for ix in range(ncols - 1):
            s = float(xpos[ix + 1] - xpos[ix])
            ow = overlap_width(tw, s)
            if ow < 4:
                continue
            L = tiles[(ix, iy)][:, -ow:]
            R = tiles[(ix + 1, iy)][:, :ow]
            wL = dp_seam_vertical_weights(L, R, seam_smooth)
            wR = 1.0 - wL
            xL0 = int(np.floor(xpos[ix])) + tw - ow
            y0 = int(np.floor(ypos[iy]))
            xR0 = int(np.floor(xpos[ix + 1]))
            weights[(ix, iy)][:, -ow:] *= wL
            weights[(ix + 1, iy)][:, :ow] *= wR

    for iy in range(nrows - 1):
        for ix in range(ncols):
            s = float(ypos[iy + 1] - ypos[iy])
            oh = overlap_width(th, s)
            if oh < 4:
                continue
            T = tiles[(ix, iy)][-oh:, :]
            B = tiles[(ix, iy + 1)][:oh, :]
            wT = dp_seam_horizontal_weights(T, B, seam_smooth)
            wB = 1.0 - wT
            x0 = int(np.floor(xpos[ix]))
            yT0 = int(np.floor(ypos[iy])) + th - oh
            yB0 = int(np.floor(ypos[iy + 1]))
            weights[(ix, iy)][-oh:, :] *= wT
            weights[(ix, iy + 1)][:oh, :] *= wB

    for k in weights:
        weights[k] = np.clip(weights[k], 0.12, 1.0)

    acc = np.zeros((ch, cw, 3), dtype=np.float64)
    wsum = np.zeros((ch, cw), dtype=np.float64)
    for iy in range(nrows):
        for ix in range(ncols):
            im = tiles[(ix, iy)].astype(np.float64)
            wt = weights[(ix, iy)][:, :, np.newaxis]
            x0 = int(np.floor(xpos[ix]))
            y0 = int(np.floor(ypos[iy]))
            h, w = im.shape[:2]
            acc[y0 : y0 + h, x0 : x0 + w] += im * wt
            wsum[y0 : y0 + h, x0 : x0 + w] += weights[(ix, iy)]
    wsum = np.maximum(wsum, 1e-6)
    return (acc / wsum[..., np.newaxis]).clip(0, 255).astype(np.uint8)


def _tile_stem_candidates(pattern: str, api_x: int, api_y: int) -> list[str]:
    """Filename stems to try, in order, for one API (x, y) pair."""
    p = pattern.strip().lower()
    if p == "auto":
        return [
            f"crop_{api_x}_{api_y}",
            f"raw_{api_x}_{api_y}",
            f"tile_{api_x}_{api_y}",
            f"{api_x}_{api_y}",
            f"X{api_x}_Y{api_y}",
            f"x{api_x}_y{api_y}",
        ]
    if p == "raw":
        return [f"raw_{api_x}_{api_y}"]
    if p == "crop":
        return [f"crop_{api_x}_{api_y}"]
    if "{x}" in pattern or "{y}" in pattern:
        return [pattern.format(x=api_x, y=api_y)]
    return [pattern.replace("X", str(api_x)).replace("Y", str(api_y))]


def find_tile_file(
    root: Path,
    api_x: int,
    api_y: int,
    pattern: str = "auto",
) -> Path | None:
    """Pick the first existing file for this grid position, or None."""
    for stem in _tile_stem_candidates(pattern, api_x, api_y):
        for ext in TILE_EXTENSIONS:
            path = root / f"{stem}{ext}"
            if path.is_file():
                return path
    return None


def load_bgr_from_path(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Could not read image: {path}")
    return im


def _should_skip_crop(path: Path, pattern: str, tiles_precropped: bool) -> bool:
    # crop_* files from stitch_magnifier are already the interior
    if tiles_precropped:
        return True
    p = pattern.strip().lower()
    if p == "crop":
        return True
    if p == "auto" and path.stem.lower().startswith("crop_"):
        return True
    return False


def load_tiles_from_folder(
    root: Path | str,
    xs: list[int],
    ys: list[int],
    crop: tuple[int, int, int, int],
    pattern: str = "auto",
    tiles_precropped: bool = False,
    workers: int = 4,
) -> dict[tuple[int, int], np.ndarray]:
    """Build the same tile dict as download_tiles(), but from disk."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Tiles folder not found: {root}")

    jobs = [
        TileJob(api_x=x, api_y=y, ix=ix, iy=iy)
        for iy, y in enumerate(ys)
        for ix, x in enumerate(xs)
    ]
    out: dict[tuple[int, int], np.ndarray] = {}
    missing: list[str] = []

    def work(job: TileJob) -> tuple[int, int, np.ndarray]:
        path = find_tile_file(root, job.api_x, job.api_y, pattern)
        if path is None:
            raise FileNotFoundError(
                f"No tile for X={job.api_x} Y={job.api_y} under {root} (pattern={pattern!r})"
            )
        bgr = load_bgr_from_path(path)
        if _should_skip_crop(path, pattern, tiles_precropped):
            return job.ix, job.iy, bgr
        return job.ix, job.iy, crop_inner_bgr(bgr, crop[0], crop[1], crop[2], crop[3])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                ix, iy, im = fut.result()
                out[(ix, iy)] = im
            except FileNotFoundError:
                missing.append(f"X={job.api_x} Y={job.api_y}")

    if missing:
        hint = "Expected names like raw_{x}_{y}.jpg or crop_{x}_{y}.jpg (stitch_magnifier --tiles-dir)."
        raise FileNotFoundError(f"Missing {len(missing)} tile(s): {', '.join(missing[:8])}" + (f" ... (+{len(missing)-8})" if len(missing) > 8 else "") + f". {hint}")

    return out


def download_tiles(
    session: requests.Session,
    base: str,
    params: str,
    xs: list[int],
    ys: list[int],
    crop: tuple[int, int, int, int],
    timeout: float,
    delay_s: float,
    workers: int,
) -> dict[tuple[int, int], np.ndarray]:
    jobs = [
        TileJob(api_x=x, api_y=y, ix=ix, iy=iy)
        for iy, y in enumerate(ys)
        for ix, x in enumerate(xs)
    ]
    out: dict[tuple[int, int], np.ndarray] = {}

    def work(job: TileJob) -> tuple[int, int, np.ndarray]:
        url = build_url(base, params, job.api_x, job.api_y, True)
        raw = fetch_bgr(session, url, timeout)
        inner = crop_inner_bgr(raw, crop[0], crop[1], crop[2], crop[3])
        if delay_s > 0:
            time.sleep(delay_s)
        return job.ix, job.iy, inner

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            ix, iy, im = fut.result()
            out[(ix, iy)] = im

    return out


def grid_from_target(
    tw: int,
    th: int,
    target_w: int,
    target_h: int,
    nom_dx: float,
    nom_dy: float,
) -> tuple[int, int]:
    """How many columns/rows we need to reach at least target_w x target_h."""
    ncols = 1 if target_w <= tw else int(np.ceil((target_w - tw) / max(nom_dx, 1e-6))) + 1
    nrows = 1 if target_h <= th else int(np.ceil((target_h - th) / max(nom_dy, 1e-6))) + 1
    return max(1, ncols), max(1, nrows)



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="http://magnifier.flashphotography.com/MagnifyRender.ashx")
    p.add_argument(
        "--params",
        default="",
        help="Query fragment after X/Y (required for API; omit when using --tiles-dir).",
    )
    p.add_argument(
        "--tiles-dir",
        type=Path,
        default=None,
        help="Load tiles from folder instead of API (e.g. raw_X_Y.jpg from stitch_magnifier).",
    )
    p.add_argument(
        "--tile-pattern",
        default="auto",
        help="auto | raw | crop | custom template with {x} and {y} (e.g. tile_{x}_{y}).",
    )
    p.add_argument(
        "--tiles-precropped",
        action="store_true",
        help="Files are already interior-cropped; do not apply --crop.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--target-width",
        type=int,
        default=None,
        help="With --target-height/--anchor-* / --api-*: minimum stitched width (cropped px).",
    )
    g.add_argument(
        "--x-from",
        type=int,
        default=None,
        help="Explicit API grid (use with --x-to/--x-step and y-*).",
    )
    p.add_argument("--target-height", type=int, default=None)
    p.add_argument("--anchor-x", type=int, default=0)
    p.add_argument("--anchor-y", type=int, default=0)
    p.add_argument("--api-x-step", type=int, default=None, help="API X increment between adjacent columns (target mode).")
    p.add_argument("--api-y-step", type=int, default=None, help="API Y increment between adjacent rows (target mode).")
    p.add_argument("--x-to", type=int, default=None)
    p.add_argument("--x-step", type=int, default=None)
    p.add_argument("--y-from", type=int, default=None)
    p.add_argument("--y-to", type=int, default=None)
    p.add_argument("--y-step", type=int, default=None)
    p.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("L", "U", "R", "B"),
        default=[36, 36, 153, 153],
        help="PIL-style crop on raw 188px tile: left upper right lower (exclusive R/B). Four integers after --crop.",
    )
    p.add_argument("--nominal-dx", type=float, default=None)
    p.add_argument("--nominal-dy", type=float, default=None)
    p.add_argument("--detector", choices=("akaze", "sift", "orb"), default="akaze")
    p.add_argument("--match-ratio", type=float, default=0.75)
    p.add_argument("--ransac-px", type=float, default=3.0)
    p.add_argument("--roi-frac", type=float, default=0.55, help="Larger overlap strips help big mosaics (try 0.55-0.65).")
    p.add_argument(
        "--refine-phase",
        action="store_true",
        help="Refine each neighbor stride with windowed phase correlation (after features).",
    )
    p.add_argument("--phase-max-adj", type=float, default=6.0, help="Max subpixel adjustment from phase refine.")
    p.add_argument(
        "--min-overlap",
        type=int,
        default=16,
        metavar="PX",
        help="Clamp stride so neighbors overlap by at least this many pixels (avoids huge black gaps).",
    )
    p.add_argument(
        "--strides",
        choices=("matched", "nominal"),
        default="matched",
        help="matched: AKAZE/SIFT/ORB + optional phase (default). nominal: use --nominal-dx/dy only (faster, good if calibration is trusted).",
    )
    p.add_argument(
        "--blend",
        choices=("mean", "feather", "multiband", "seam"),
        default="multiband",
        help="mean | feather (distance weights) | multiband (Laplacian pyramid) | seam (DP + feather).",
    )
    p.add_argument("--feather-px", type=int, default=28, help="For feather / seam: falloff from tile edges.")
    p.add_argument("--seam-smooth", type=int, default=2, help="Pixels to soften DP seam mask transitions.")
    p.add_argument("--multiband-levels", type=int, default=4, help="Requested pyramid bands (capped by overlap; lower = sharper).")
    p.add_argument(
        "--sharpen",
        type=float,
        default=0.0,
        metavar="AMOUNT",
        help="Unsharp mask strength after blend (e.g. 0.35-0.7). 0 disables.",
    )
    p.add_argument("--sharpen-sigma", type=float, default=1.2, help="Gaussian sigma for unsharp (pixels).")
    p.add_argument("--out", type=Path, default=Path("mosaic_feature.jpg"))
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tiles", type=int, default=25000, help="Abort if computed grid exceeds this many tiles.")

    args = p.parse_args(argv)

    if args.tiles_dir is None and not (args.params or "").strip():
        p.error("--params is required unless --tiles-dir is set.")
    if args.tiles_dir is not None and not args.tiles_dir.is_dir():
        p.error(f"--tiles-dir is not a directory: {args.tiles_dir}")

    use_target = args.target_width is not None
    if use_target:
        if args.target_height is None or args.api_x_step is None or args.api_y_step is None:
            p.error("Target mode requires --target-width, --target-height, --api-x-step, and --api-y-step.")
        if args.x_from is not None:
            p.error("Do not pass --x-from with --target-width (mutually exclusive).")
    else:
        missing = [n for n, v in [
            ("--x-to", args.x_to), ("--x-step", args.x_step),
            ("--y-from", args.y_from), ("--y-to", args.y_to), ("--y-step", args.y_step),
        ] if v is None]
        if missing:
            p.error("Explicit grid requires --x-from --x-to --x-step --y-from --y-to --y-step (or use target mode).")

    if use_target:
        cl, cu, cr, cb = args.crop
        tw0, th0 = cr - cl, cb - cu
        nom_dx = args.nominal_dx if args.nominal_dx is not None else max(32.0, tw0 * 0.38)
        nom_dy = args.nominal_dy if args.nominal_dy is not None else nom_dx
        ncols, nrows = grid_from_target(tw0, th0, args.target_width, args.target_height, nom_dx, nom_dy)
        n_tiles = ncols * nrows
        if n_tiles > args.max_tiles:
            p.error(f"Grid {ncols}x{nrows} = {n_tiles} tiles exceeds --max-tiles {args.max_tiles}.")
        last_x = args.anchor_x + (ncols - 1) * args.api_x_step
        last_y = args.anchor_y + (nrows - 1) * args.api_y_step
        xs = frange(args.anchor_x, last_x, args.api_x_step)
        ys = frange(args.anchor_y, last_y, args.api_y_step)
        print(
            f"Target >={args.target_width}x{args.target_height} px -> grid {ncols}x{nrows} "
            f"({n_tiles} tiles), API X: {xs[0]}..{xs[-1]} step {args.api_x_step}, "
            f"Y: {ys[0]}..{ys[-1]} step {args.api_y_step}"
        )
    else:
        xs = frange(args.x_from, args.x_to, args.x_step)  # type: ignore[arg-type]
        ys = frange(args.y_from, args.y_to, args.y_step)  # type: ignore[arg-type]
        if len(xs) * len(ys) > args.max_tiles:
            p.error("Tile count exceeds --max-tiles.")

    if not xs or not ys:
        print("Empty X or Y range.", file=sys.stderr)
        return 2

    crop_t = (int(args.crop[0]), int(args.crop[1]), int(args.crop[2]), int(args.crop[3]))
    cfg = MosaicRunConfig(
        base=args.base,
        params=(args.params or "").strip(),
        crop=crop_t,
        xs=xs,
        ys=ys,
        tiles_dir=args.tiles_dir,
        tile_pattern=args.tile_pattern,
        tiles_precropped=args.tiles_precropped,
        nominal_dx=args.nominal_dx,
        nominal_dy=args.nominal_dy,
        strides=args.strides,
        refine_phase=args.refine_phase,
        phase_max_adj=args.phase_max_adj,
        min_overlap=args.min_overlap,
        detector=args.detector,
        match_ratio=args.match_ratio,
        ransac_px=args.ransac_px,
        roi_frac=args.roi_frac,
        blend=args.blend,
        feather_px=args.feather_px,
        seam_smooth=args.seam_smooth,
        multiband_levels=args.multiband_levels,
        sharpen=args.sharpen,
        sharpen_sigma=args.sharpen_sigma,
        timeout=args.timeout,
        delay=args.delay,
        workers=args.workers,
    )
    mosaic, log = run_mosaic(cfg)
    for line in log.splitlines():
        if line.startswith("Warning:") or line.startswith("Note:"):
            print(line, file=sys.stderr)
        else:
            print(line)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_path = str(args.out)
    if out_path.lower().endswith(".png"):
        cv2.imwrite(out_path, mosaic, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        cv2.imwrite(out_path, mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print(f"Wrote {args.out} ({mosaic.shape[1]} x {mosaic.shape[0]} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
