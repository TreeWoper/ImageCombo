"""
Download magnifier tiles and stitch them with measured overlap.

See README.md for setup, calibration, and all CLI flags.

Quick start: calibrate first, then stitch with the printed crop and steps:

  python stitch_magnifier.py --params "O=...&R=...&F=...&A=..." --calibrate --auto-stride
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import requests
from PIL import Image, ImageChops, ImageStat


@dataclass(frozen=True)
class TileJob:
    x: int
    y: int
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


def build_url(
    base: str,
    params: str,
    x: int,
    y: int,
    cache_bust: bool,
) -> str:
    # params should look like: O=..&R=..&F=..&A=..
    sep = "&" if ("?" in base) else "?"
    q = f"X={x}&Y={y}"
    if params.strip():
        q += "&" + params.strip().lstrip("&")
    if cache_bust:
        q += f"&rand={random.random()}"
    return f"{base}{sep}{q}"


def fetch_tile(session: requests.Session, url: str, timeout: float) -> Image.Image:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def crop_tile(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    l, t, r, b = box
    if not (0 <= l < r <= img.size[0] and 0 <= t < b <= img.size[1]):
        raise ValueError(
            f"Crop box {box} invalid for image size {img.size}. "
            "Measure borders in an editor or run --calibrate."
        )
    return img.crop(box)


def seam_score_h(a: Image.Image, b: Image.Image, box: tuple[int, int, int, int], strip: int) -> float:
    ca = a.crop(box)
    cb = b.crop(box)
    w = ca.size[0]
    if strip >= w:
        return float("inf")
    ra = ca.crop((w - strip, 0, w, w))
    lb = cb.crop((0, 0, strip, w))
    return float(sum(ImageStat.Stat(ImageChops.difference(ra, lb)).mean))


def seam_score_v(a: Image.Image, b: Image.Image, box: tuple[int, int, int, int], strip: int) -> float:
    ca = a.crop(box)
    cb = b.crop(box)
    h = ca.size[1]
    if strip >= h:
        return float("inf")
    ra = ca.crop((0, h - strip, h, h))
    lb = cb.crop((0, 0, h, strip))
    return float(sum(ImageStat.Stat(ImageChops.difference(ra, lb)).mean))


def _best_symmetric_crop(
    a: Image.Image,
    b: Image.Image | None,
    c: Image.Image | None,
    strip: int,
    left_lo: int = 18,
    left_hi: int = 94,
    inner_lo: int = 48,
    inner_hi: int = 124,
) -> tuple[float, tuple[int, int, int, int], float, float]:
    """Return (total_score, box, horiz_score, vert_score). Pass b=None to skip H, c=None to skip V."""
    best: tuple[float, tuple[int, int, int, int], float, float] | None = None
    for left in range(left_lo, left_hi + 1):
        for inner in range(inner_lo, inner_hi + 1):
            right = left + inner
            bottom = left + inner
            if right > a.size[0] or bottom > a.size[1]:
                continue
            box = (left, left, right, bottom)
            sh = seam_score_h(a, b, box, strip) if b is not None else 0.0
            sv = seam_score_v(a, c, box, strip) if c is not None else 0.0
            score = sh + sv
            if best is None or score < best[0]:
                best = (score, box, sh, sv)
    if best is None:
        raise RuntimeError("No valid symmetric crop for this tile size.")
    return best


def calibrate(
    session: requests.Session,
    base: str,
    params: str,
    x0: int,
    y0: int,
    dx: int,
    dy: int,
    timeout: float,
    strip: int,
) -> None:
    """Print heuristic crop + stride scores using four nearby tiles."""
    a = fetch_tile(session, build_url(base, params, x0, y0, True), timeout)
    b = fetch_tile(session, build_url(base, params, x0 + dx, y0, True), timeout)
    c = fetch_tile(session, build_url(base, params, x0, y0 + dy, True), timeout)

    print(f"Tile pixel size: {a.size[0]} x {a.size[1]}")
    score, box, sh, sv = _best_symmetric_crop(a, b, c, strip)
    inner = box[2] - box[0]
    print(f"Suggested crop (left upper right lower, exclusive right/bottom): {box}")
    print(f"  Cropped tile: {inner} x {inner} px")
    print(f"  Mean seam error - horizontal: {sh:.3f}, vertical: {sv:.3f}, sum: {score:.3f}")
    print("If this is still high, try --calibrate --auto-stride or adjust --strip.")


def calibrate_auto_stride(
    session: requests.Session,
    base: str,
    params: str,
    x0: int,
    y0: int,
    stride_min: int,
    stride_max: int,
    timeout: float,
    strip: int,
) -> None:
    """
    Jointly search API X/Y stride and symmetric crop. Wrong --x-step/--y-step is the
    usual cause of visible seams when the crop is already close.
    """
    if stride_min > stride_max:
        stride_min, stride_max = stride_max, stride_min

    print(f"Fetching anchor tile at X={x0} Y={y0} ...")
    a = fetch_tile(session, build_url(base, params, x0, y0, True), timeout)
    print(f"Tile pixel size: {a.size[0]} x {a.size[1]}")

    print(f"Fetching neighbors for strides {stride_min}..{stride_max} (one request per stride) ...")
    by_dx: dict[int, Image.Image] = {}
    by_dy: dict[int, Image.Image] = {}
    for d in range(stride_min, stride_max + 1):
        by_dx[d] = fetch_tile(session, build_url(base, params, x0 + d, y0, True), timeout)
        by_dy[d] = fetch_tile(session, build_url(base, params, x0, y0 + d, True), timeout)

    best: tuple[float, int, int, tuple[int, int, int, int], float, float] | None = None
    # Tighter crop search keeps joint (stride × crop) search fast enough.
    auto_left_lo, auto_left_hi = 20, 52
    auto_inner_lo, auto_inner_hi = 58, 78

    for dx in range(stride_min, stride_max + 1):
        b = by_dx[dx]
        for dy in range(stride_min, stride_max + 1):
            c = by_dy[dy]
            score, box, sh, sv = _best_symmetric_crop(
                a,
                b,
                c,
                strip,
                left_lo=auto_left_lo,
                left_hi=auto_left_hi,
                inner_lo=auto_inner_lo,
                inner_hi=auto_inner_hi,
            )
            if best is None or score < best[0]:
                best = (score, dx, dy, box, sh, sv)

    assert best is not None
    score, dx, dy, box, sh, sv = best
    inner = box[2] - box[0]
    print()
    print("Best joint match (lowest H+V seam score):")
    print(f"  --x-step {dx}  --y-step {dy}")
    print(f"  --crop {box[0]} {box[1]} {box[2]} {box[3]}")
    print(f"  Cropped tile: {inner} x {inner} px")
    print(f"  Mean seam error - horizontal: {sh:.3f}, vertical: {sv:.3f}, sum: {score:.3f}")
    if score > 80:
        print(
            "\nWarning: score is still high. Try --cal-strip 20 or 30, "
            "or set --cal-x0/--cal-y0 to a well-textured area (not blank sky).",
            file=sys.stderr,
        )


def smooth_seams_rgb(
    canvas: Image.Image,
    tw: int,
    th: int,
    ncols: int,
    nrows: int,
    paste_x: int,
    paste_y: int,
    passes: int,
) -> None:
    """Average RGB across tile boundaries to soften zipper/JPEG seams (in-place)."""
    if passes <= 0:
        return
    px = canvas.load()
    w, h = canvas.size

    def blend_pair(x1: int, y1: int, x2: int, y2: int) -> None:
        p1 = px[x1, y1]
        p2 = px[x2, y2]
        m = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2, (p1[2] + p2[2]) // 2)
        px[x1, y1] = m
        px[x2, y2] = m

    for _ in range(passes):
        for iy in range(nrows):
            row_y0 = iy * paste_y
            row_y1 = min(row_y0 + th, h) - 1
            for ix in range(1, ncols):
                sx = ix * paste_x
                if sx >= w:
                    continue
                for y in range(row_y0, row_y1 + 1):
                    blend_pair(sx - 1, y, sx, y)
        for ix in range(ncols):
            col_x0 = ix * paste_x
            col_x1 = min(col_x0 + tw, w) - 1
            for iy in range(1, nrows):
                sy = iy * paste_y
                if sy >= h:
                    continue
                for x in range(col_x0, col_x1 + 1):
                    blend_pair(x, sy - 1, x, sy)


def download_jobs(
    session: requests.Session,
    base: str,
    params: str,
    xs: Sequence[int],
    ys: Sequence[int],
    box: tuple[int, int, int, int],
    out_dir: Path | None,
    timeout: float,
    delay_s: float,
    workers: int,
) -> dict[tuple[int, int], Image.Image]:
    jobs: list[TileJob] = [
        TileJob(x=x, y=y, ix=ix, iy=iy) for iy, y in enumerate(ys) for ix, x in enumerate(xs)
    ]

    results: dict[tuple[int, int], Image.Image] = {}

    def work(job: TileJob) -> tuple[int, int, Image.Image]:
        url = build_url(base, params, job.x, job.y, True)
        img = fetch_tile(session, url, timeout)
        cropped = crop_tile(img, box)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            raw_path = out_dir / f"raw_{job.x}_{job.y}.jpg"
            crop_path = out_dir / f"crop_{job.x}_{job.y}.jpg"
            img.save(raw_path, quality=95)
            cropped.save(crop_path, quality=95)
        if delay_s > 0:
            time.sleep(delay_s)
        return job.ix, job.iy, cropped

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            ix, iy, cropped = fut.result()
            results[(ix, iy)] = cropped

    return results


def stitch_at_positions(
    tiles: dict[tuple[int, int], Image.Image],
    ncols: int,
    nrows: int,
    xpos: list[int],
    ypos: list[int],
) -> Image.Image:
    """Paste each tile at (xpos[ix], ypos[iy]) without blending."""
    if not tiles:
        raise ValueError("No tiles to stitch")
    tw, th = next(iter(tiles.values())).size
    cw = xpos[-1] + tw if ncols else tw
    ch = ypos[-1] + th if nrows else th
    canvas = Image.new("RGB", (cw, ch))
    for iy in range(nrows):
        for ix in range(ncols):
            im = tiles.get((ix, iy))
            if im is None:
                raise KeyError(f"Missing tile ix={ix}, iy={iy}")
            canvas.paste(im, (xpos[ix], ypos[iy]))
    return canvas


def stitch(
    tiles: dict[tuple[int, int], Image.Image],
    ncols: int,
    nrows: int,
    paste_x: int,
    paste_y: int,
) -> Image.Image:
    """Paste each tile at (ix*paste_x, iy*paste_y). Use paste_x==tw for edge-to-edge."""
    if not tiles:
        raise ValueError("No tiles to stitch")
    tw, th = next(iter(tiles.values())).size
    cw = (ncols - 1) * paste_x + tw if ncols > 0 else tw
    ch = (nrows - 1) * paste_y + th if nrows > 0 else th
    canvas = Image.new("RGB", (cw, ch))
    for iy in range(nrows):
        for ix in range(ncols):
            im = tiles.get((ix, iy))
            if im is None:
                raise KeyError(f"Missing tile ix={ix}, iy={iy}")
            canvas.paste(im, (ix * paste_x, iy * paste_y))
    return canvas


def _h_paste_seam_error(a: Image.Image, b: Image.Image, p: int) -> float:
    """Error if b is placed with its left edge p pixels to the right of a's left edge (overlap = tw - p)."""
    tw, th = a.size
    if p <= 0 or p >= tw:
        return float("inf")
    ov = tw - p
    if ov < 2:
        return float("inf")
    pa = a.crop((p, 0, tw, th))
    pb = b.crop((0, 0, ov, th))
    d = ImageChops.difference(pa, pb)
    return float(sum(ImageStat.Stat(d).mean))


def _v_paste_seam_error(a: Image.Image, b: Image.Image, q: int) -> float:
    tw, th = a.size
    if q <= 0 or q >= th:
        return float("inf")
    ov = th - q
    if ov < 2:
        return float("inf")
    pa = a.crop((0, q, tw, th))
    pb = b.crop((0, 0, tw, ov))
    d = ImageChops.difference(pa, pb)
    return float(sum(ImageStat.Stat(d).mean))


def _best_paste_1d(
    err_fn: Callable[[Image.Image, Image.Image, int], float],
    a: Image.Image,
    b: Image.Image,
    dim: int,
) -> tuple[int, float]:
    """Minimize err_fn(a,b,k) for k; among near-minima prefer larger k (less overlap, fewer false cycles)."""
    scores: list[tuple[float, int]] = []
    for k in range(4, dim - 2):
        e = float(err_fn(a, b, k))
        scores.append((e, k))
    if not scores:
        return dim, float("inf")
    min_e = min(s[0] for s in scores)
    tol = max(1.0, min_e * 0.08)
    good = [k for e, k in scores if e <= min_e + tol]
    return max(good), min_e


def _median_or(vals: list[int], fallback: int) -> int:
    if not vals:
        return fallback
    s = sorted(vals)
    return s[len(s) // 2]


def estimate_paste_strides(
    tiles: dict[tuple[int, int], Image.Image],
    ncols: int,
    nrows: int,
    tw: int,
    th: int,
    per_edge: bool,
) -> tuple[int, int] | tuple[list[int], list[int]]:
    """
    Horizontal paste px: left edge of column ix+1 is px pixels right of column ix.
    Vertical paste py: same for rows. Overlap is tw - px (when px < tw).

    If per_edge is True, returns (hx_edges, vy_edges) with len ncols-1 / nrows-1
    medians per boundary; caller uses cumulative positions. If False, returns a
    single (px, py) median over all pairs (backward compatible).
    """
    if per_edge:
        hx: list[int] = []
        for ix in range(ncols - 1):
            row_ps: list[int] = []
            for iy in range(nrows):
                a = tiles[(ix, iy)]
                b = tiles[(ix + 1, iy)]
                p, _e = _best_paste_1d(_h_paste_seam_error, a, b, tw)
                row_ps.append(p)
            hx.append(_median_or(row_ps, tw))
        vy: list[int] = []
        for iy in range(nrows - 1):
            col_qs: list[int] = []
            for ix in range(ncols):
                a = tiles[(ix, iy)]
                b = tiles[(ix, iy + 1)]
                q, _e = _best_paste_1d(_v_paste_seam_error, a, b, th)
                col_qs.append(q)
            vy.append(_median_or(col_qs, th))
        return hx, vy

    px_list: list[int] = []
    py_list: list[int] = []

    if ncols > 1:
        for iy in range(nrows):
            for ix in range(ncols - 1):
                a = tiles[(ix, iy)]
                b = tiles[(ix + 1, iy)]
                p, _e = _best_paste_1d(_h_paste_seam_error, a, b, tw)
                px_list.append(p)

    if nrows > 1:
        for ix in range(ncols):
            for iy in range(nrows - 1):
                a = tiles[(ix, iy)]
                b = tiles[(ix, iy + 1)]
                q, _e = _best_paste_1d(_v_paste_seam_error, a, b, th)
                py_list.append(q)

    px = _median_or(px_list, tw)
    py = _median_or(py_list, th)
    return px, py


def _prefix_positions(strides: list[int], n: int) -> list[int]:
    """Left/top edge for each index 0..n-1 from consecutive strides (len(strides) should be n-1)."""
    pos = [0]
    for i in range(n - 1):
        pos.append(pos[-1] + strides[i])
    return pos


def _uniform_positions(n: int, stride: int) -> list[int]:
    return [i * stride for i in range(n)]


def _lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], w: float) -> tuple[int, int, int]:
    w = min(1.0, max(0.0, w))
    return (
        int(round(a[0] + (b[0] - a[0]) * w)),
        int(round(a[1] + (b[1] - a[1]) * w)),
        int(round(a[2] + (b[2] - a[2]) * w)),
    )


def _blend_weight_1d(coord: int, overlap: int) -> float:
    """0 = favor neighbor, 1 = favor new tile, for coord in 0..overlap-1."""
    if overlap <= 0:
        return 1.0
    if overlap == 1:
        return 0.5
    return coord / (overlap - 1)


def stitch_overlap_blend(
    tiles: dict[tuple[int, int], Image.Image],
    ncols: int,
    nrows: int,
    xpos: list[int],
    ypos: list[int],
    tw: int,
    th: int,
) -> Image.Image:
    """
    Place tiles at xpos[ix], ypos[iy]. In overlaps with left/top neighbors, linearly
    blend toward the new tile (bilinear in the corner). Requires tw > stride so
    overlap width = tw - (xpos[ix] - xpos[ix-1]) is positive when ix > 0.
    """
    cw = xpos[-1] + tw if ncols else tw
    ch = ypos[-1] + th if nrows else th
    canvas = Image.new("RGB", (cw, ch))
    px_canvas = canvas.load()

    for iy in range(nrows):
        for ix in range(ncols):
            im = tiles[(ix, iy)]
            if im.size != (tw, th):
                raise ValueError(f"Tile ({ix},{iy}) size {im.size} != ({tw},{th})")
            bx, by = xpos[ix], ypos[iy]
            src = im.load()

            ov_h = max(0, tw - (bx - xpos[ix - 1])) if ix > 0 else 0
            ov_v = max(0, th - (by - ypos[iy - 1])) if iy > 0 else 0

            for y in range(th):
                cy = by + y
                for x in range(tw):
                    cx = bx + x
                    cb = src[x, y]
                    if ix == 0 and iy == 0:
                        px_canvas[cx, cy] = cb
                        continue

                    if iy > 0 and y < ov_v and ix > 0 and x < ov_h:
                        w_x = _blend_weight_1d(x, ov_h)
                        w_y = _blend_weight_1d(y, ov_v)
                        c_tl = px_canvas[cx - ov_h, cy - ov_v]
                        c_l = px_canvas[cx - ov_h, cy]
                        c_t = px_canvas[cx, cy - ov_v]
                        top = _lerp_rgb(c_tl, c_l, w_x)
                        bot = _lerp_rgb(c_t, cb, w_x)
                        px_canvas[cx, cy] = _lerp_rgb(top, bot, w_y)
                    elif iy > 0 and y < ov_v:
                        w_y = _blend_weight_1d(y, ov_v)
                        c_t = px_canvas[cx, cy - ov_v]
                        px_canvas[cx, cy] = _lerp_rgb(c_t, cb, w_y)
                    elif ix > 0 and x < ov_h:
                        w_x = _blend_weight_1d(x, ov_h)
                        c_l = px_canvas[cx - ov_h, cy]
                        px_canvas[cx, cy] = _lerp_rgb(c_l, cb, w_x)
                    else:
                        px_canvas[cx, cy] = cb

    return canvas


def parse_crop(s: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in s.replace(",", " ").split()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop needs four integers: left upper right lower")
    return (parts[0], parts[1], parts[2], parts[3])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--base",
        default="http://magnifier.flashphotography.com/MagnifyRender.ashx",
        help="MagnifyRender endpoint (without query string, or with ? if you prefer).",
    )
    p.add_argument(
        "--params",
        required=True,
        help='Fixed query fragment after X/Y, e.g. O=27341312&R=10103&F=0194&A=71714',
    )
    p.add_argument("--x-from", type=int, default=None)
    p.add_argument("--x-to", type=int, default=None)
    p.add_argument("--x-step", type=int, default=44)
    p.add_argument("--y-from", type=int, default=None)
    p.add_argument("--y-to", type=int, default=None)
    p.add_argument("--y-step", type=int, default=44)
    p.add_argument(
        "--crop",
        type=parse_crop,
        default="36 36 105 105",
        help="PIL crop box on each downloaded tile: left upper right lower (default tuned for 188px tiles).",
    )
    p.add_argument("--out", type=Path, default=Path("stitched.jpg"))
    p.add_argument("--tiles-dir", type=Path, default=None, help="If set, save each raw+cropped tile here.")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--delay", type=float, default=0.0, help="Per-tile delay after download (be polite).")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Do not stitch; fetch a few tiles and print suggested crop for given dx/dy.",
    )
    p.add_argument("--cal-x0", type=int, default=256)
    p.add_argument("--cal-y0", type=int, default=125)
    p.add_argument("--cal-dx", type=int, default=44)
    p.add_argument("--cal-dy", type=int, default=44)
    p.add_argument("--cal-strip", type=int, default=25, help="Band height/width used when scoring seams.")
    p.add_argument(
        "--auto-stride",
        action="store_true",
        help="With --calibrate: search --stride-min..--stride-max for best X/Y step and crop (fixes most seams).",
    )
    p.add_argument(
        "--stride-min",
        type=int,
        default=36,
        help="With --calibrate --auto-stride: smallest API stride to try (inclusive).",
    )
    p.add_argument(
        "--stride-max",
        type=int,
        default=54,
        help="With --calibrate --auto-stride: largest API stride to try (inclusive).",
    )
    p.add_argument(
        "--seam-smooth",
        type=int,
        default=0,
        metavar="N",
        help="After stitching, blend RGB across tile boundaries N times (0-3; try 1 to soften JPEG seams).",
    )
    p.add_argument(
        "--layout",
        choices=("auto", "abut"),
        default="auto",
        help="auto: overlap tiles by measured photo shift (default). abut: place tiles edge-to-edge at full crop size.",
    )
    p.add_argument(
        "--per-edge-strides",
        action="store_true",
        help="With --layout auto: one measured stride per column/row boundary (median across the other axis), "
        "then cumulative placement. Helps when effective shift drifts slightly across the grid.",
    )
    p.add_argument(
        "--overlap-blend",
        action="store_true",
        help="Blend RGB in tile overlaps (JPEG-friendly). Use with strides < tile size; "
        "usually sharper than --seam-smooth alone when tiles disagree in overlap.",
    )
    p.add_argument(
        "--paste-x",
        type=int,
        default=None,
        help="Override horizontal paste stride in pixels (left-edge to left-edge). Default: from --layout auto.",
    )
    p.add_argument(
        "--paste-y",
        type=int,
        default=None,
        help="Override vertical paste stride in pixels. Default: from --layout auto.",
    )

    args = p.parse_args(argv)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "imageStitch/1.0 (+https://github.com/)",
            "Accept": "image/*,*/*;q=0.8",
        }
    )

    if args.calibrate:
        if args.auto_stride:
            calibrate_auto_stride(
                session,
                args.base,
                args.params,
                args.cal_x0,
                args.cal_y0,
                args.stride_min,
                args.stride_max,
                args.timeout,
                args.cal_strip,
            )
        else:
            calibrate(
                session,
                args.base,
                args.params,
                args.cal_x0,
                args.cal_y0,
                args.cal_dx,
                args.cal_dy,
                args.timeout,
                args.cal_strip,
            )
        return 0

    if None in (args.x_from, args.x_to, args.y_from, args.y_to):
        p.error("Tile ranges --x-from/--x-to/--y-from/--y-to are required unless --calibrate is set.")

    if args.seam_smooth < 0 or args.seam_smooth > 3:
        p.error("--seam-smooth must be between 0 and 3.")

    xs = frange(args.x_from, args.x_to, args.x_step)
    ys = frange(args.y_from, args.y_to, args.y_step)
    if not xs or not ys:
        print("Empty X or Y range.", file=sys.stderr)
        return 2

    print(f"Grid: {len(xs)} x {len(ys)} = {len(xs) * len(ys)} tiles")
    tiles = download_jobs(
        session,
        args.base,
        args.params,
        xs,
        ys,
        args.crop,
        args.tiles_dir,
        args.timeout,
        args.delay,
        args.workers,
    )
    tw, th = next(iter(tiles.values())).size
    for _k, im in tiles.items():
        if im.size != (tw, th):
            print(
                f"WARNING: cropped tiles are not all {tw}x{th} (found {im.size}); "
                "row tops can look uneven. Check --crop and downloads.",
                file=sys.stderr,
            )
            break
    if args.layout == "abut" and args.per_edge_strides:
        p.error("--per-edge-strides is only meaningful with --layout auto.")
    if args.per_edge_strides and (args.paste_x is not None or args.paste_y is not None):
        p.error("Do not combine --per-edge-strides with --paste-x/--paste-y.")

    ncols, nrows = len(xs), len(ys)

    if args.layout == "abut":
        px, py = tw, th
        xpos = _uniform_positions(ncols, tw)
        ypos = _uniform_positions(nrows, th)
    elif args.per_edge_strides:
        est = estimate_paste_strides(tiles, ncols, nrows, tw, th, per_edge=True)
        assert isinstance(est[0], list)
        hx, vy = est
        xpos = _prefix_positions(hx, ncols)
        ypos = _prefix_positions(vy, nrows)
        px = _median_or(hx, tw) if hx else tw
        py = _median_or(vy, th) if vy else th
    else:
        est = estimate_paste_strides(tiles, ncols, nrows, tw, th, per_edge=False)
        assert isinstance(est[0], int)
        px, py = est
        if args.paste_x is not None:
            px = args.paste_x
        if args.paste_y is not None:
            py = args.paste_y
        if px <= 0 or py <= 0:
            p.error("--paste-x and --paste-y must be positive when using uniform strides.")
        xpos = _uniform_positions(ncols, px)
        ypos = _uniform_positions(nrows, py)

    if args.layout != "abut" and not args.per_edge_strides:
        if px <= 0 or py <= 0:
            p.error("Computed paste strides must be positive.")

    ovx = tw - (xpos[1] - xpos[0]) if ncols > 1 else 0
    ovy = th - (ypos[1] - ypos[0]) if nrows > 1 else 0
    if args.per_edge_strides:
        print(
            f"Per-edge strides: horizontal median {px} px, vertical median {py} px "
            f"(tile {tw}x{th}; typical overlap ~{max(0, tw - px)}x{max(0, th - py)})"
        )
    else:
        print(
            f"Paste stride: horizontal {xpos[1] - xpos[0] if ncols > 1 else tw} px, "
            f"vertical {ypos[1] - ypos[0] if nrows > 1 else th} px "
            f"(tile {tw}x{th}; overlap ~{max(0, ovx)}x{max(0, ovy)} when stride < tile size)"
        )

    if args.overlap_blend:
        mosaic = stitch_overlap_blend(tiles, ncols, nrows, xpos, ypos, tw, th)
    else:
        mosaic = stitch_at_positions(tiles, ncols, nrows, xpos, ypos)

    if args.seam_smooth > 0:
        h_ok = ncols <= 1 or len({xpos[i + 1] - xpos[i] for i in range(ncols - 1)}) == 1
        v_ok = nrows <= 1 or len({ypos[i + 1] - ypos[i] for i in range(nrows - 1)}) == 1
        if args.overlap_blend:
            print(
                "Note: --seam-smooth skipped (--overlap-blend already softens overlaps).",
                file=sys.stderr,
            )
        elif not (h_ok and v_ok):
            print(
                "Warning: --seam-smooth skipped (strides differ between columns or rows).",
                file=sys.stderr,
            )
        else:
            px_s = xpos[1] - xpos[0] if ncols > 1 else tw
            py_s = ypos[1] - ypos[0] if nrows > 1 else th
            smooth_seams_rgb(mosaic, tw, th, ncols, nrows, px_s, py_s, args.seam_smooth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(args.out, quality=95)
    print(f"Wrote {args.out} ({mosaic.size[0]} x {mosaic.size[1]} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
