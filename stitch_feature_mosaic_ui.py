"""
Gradio UI for stitch_feature_mosaic — tune options and preview the mosaic.

  python stitch_feature_mosaic_ui.py

See README.md for what each control does.
"""

from __future__ import annotations

import traceback

import cv2
import gradio as gr
import numpy as np

from stitch_feature_mosaic import MosaicRunConfig, frange, grid_from_target, run_mosaic


def _bgr_to_rgb_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _nominal_or_none(v: float) -> float | None:
    return None if v <= 0 else float(v)


def generate_ui(
    tile_source: str,
    base: str,
    params: str,
    tiles_dir: str,
    tile_pattern: str,
    tiles_precropped: bool,
    grid_mode: str,
    x_from: int,
    x_to: int,
    x_step: int,
    y_from: int,
    y_to: int,
    y_step: int,
    target_w: int,
    target_h: int,
    anchor_x: int,
    anchor_y: int,
    api_x_step: int,
    api_y_step: int,
    crop_l: int,
    crop_u: int,
    crop_r: int,
    crop_b: int,
    nominal_dx: float,
    nominal_dy: float,
    strides: str,
    refine_phase: bool,
    phase_max_adj: float,
    min_overlap: int,
    detector: str,
    match_ratio: float,
    ransac_px: float,
    roi_frac: float,
    blend: str,
    feather_px: int,
    seam_smooth: int,
    multiband_levels: int,
    sharpen: float,
    sharpen_sigma: float,
    workers: int,
    timeout: float,
    max_tiles: int,
) -> tuple[np.ndarray | None, str]:
    log_prefix = ""
    use_folder = tile_source == "Folder"
    params = (params or "").strip()
    folder = (tiles_dir or "").strip()
    if use_folder:
        if not folder:
            return None, "Enter a tiles folder path."
    elif not params:
        return None, "Enter params (same as CLI --params: query fragment after X and Y)."

    try:
        crop = (int(crop_l), int(crop_u), int(crop_r), int(crop_b))
        if crop[2] <= crop[0] or crop[3] <= crop[1]:
            return None, "Invalid crop: need R>L and B>U."

        if grid_mode == "Target canvas":
            if target_w <= 0 or target_h <= 0 or api_x_step == 0 or api_y_step == 0:
                return None, "Target mode needs positive width/height and non-zero API X/Y steps."
            cl, cu, cr, cb = crop
            tw0, th0 = cr - cl, cb - cu
            n_dx = nominal_dx if nominal_dx > 0 else max(32.0, tw0 * 0.38)
            n_dy = nominal_dy if nominal_dy > 0 else n_dx
            ncols, nrows = grid_from_target(tw0, th0, target_w, target_h, n_dx, n_dy)
            last_x = anchor_x + (ncols - 1) * api_x_step
            last_y = anchor_y + (nrows - 1) * api_y_step
            xs = frange(anchor_x, last_x, api_x_step)
            ys = frange(anchor_y, last_y, api_y_step)
            log_prefix = (
                f"Target >= {target_w}x{target_h} px -> grid {ncols}x{nrows} "
                f"({ncols * nrows} tiles), API X {xs[0]}..{xs[-1]} step {api_x_step}, "
                f"Y {ys[0]}..{ys[-1]} step {api_y_step}\n\n"
            )
        else:
            xs = frange(int(x_from), int(x_to), int(x_step))
            ys = frange(int(y_from), int(y_to), int(y_step))
            log_prefix = f"Explicit grid: {len(xs)} x {len(ys)} = {len(xs) * len(ys)} tiles\n\n"

        if not xs or not ys:
            return None, log_prefix + "Empty X or Y range."

        n_tiles = len(xs) * len(ys)
        if n_tiles > int(max_tiles):
            return None, log_prefix + f"Abort: {n_tiles} tiles exceeds max-tiles ({max_tiles})."

        cfg = MosaicRunConfig(
            base=base.strip(),
            params=params,
            crop=crop,
            xs=xs,
            ys=ys,
            tiles_dir=folder if use_folder else None,
            tile_pattern=tile_pattern,
            tiles_precropped=tiles_precropped,
            nominal_dx=_nominal_or_none(nominal_dx),
            nominal_dy=_nominal_or_none(nominal_dy),
            strides=strides,  # type: ignore[arg-type]
            refine_phase=refine_phase,
            phase_max_adj=float(phase_max_adj),
            min_overlap=int(min_overlap),
            detector=detector,  # type: ignore[arg-type]
            match_ratio=float(match_ratio),
            ransac_px=float(ransac_px),
            roi_frac=float(roi_frac),
            blend=blend,  # type: ignore[arg-type]
            feather_px=int(feather_px),
            seam_smooth=int(seam_smooth),
            multiband_levels=int(multiband_levels),
            sharpen=float(sharpen),
            sharpen_sigma=float(sharpen_sigma),
            timeout=float(timeout),
            delay=0.0,
            workers=int(workers),
        )
        mosaic, log = run_mosaic(cfg)
        rgb = _bgr_to_rgb_u8(mosaic)
        return rgb, log_prefix + log
    except Exception:
        return None, log_prefix + traceback.format_exc()


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Feature mosaic lab") as demo:
        gr.Markdown(
            "## Feature mosaic lab\n"
            "Choose **API** or **Folder** tiles, set the grid, adjust sliders, then **Generate**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                tile_source = gr.Radio(
                    choices=["API", "Folder"],
                    value="API",
                    label="Tile source",
                )
                base = gr.Textbox(
                    label="Base URL",
                    value="http://magnifier.flashphotography.com/MagnifyRender.ashx",
                )
                params = gr.Textbox(
                    label="Params (CLI --params)",
                    placeholder="e.g. O=...&F=...",
                    lines=2,
                )
                tiles_dir = gr.Textbox(
                    label="Tiles folder",
                    placeholder="tiles/  (raw_0_0.jpg, crop_0_0.jpg, ...)",
                    lines=1,
                )
                tile_pattern = gr.Dropdown(
                    choices=["auto", "raw", "crop"],
                    value="auto",
                    label="Filename pattern",
                    info="auto tries crop_*, raw_*, etc. (stitch_magnifier names)",
                )
                tiles_precropped = gr.Checkbox(
                    label="Tiles already cropped (skip crop step)",
                    value=False,
                )
                grid_mode = gr.Radio(
                    choices=["Explicit API grid", "Target canvas"],
                    value="Explicit API grid",
                    label="Grid mode",
                )
                gr.Markdown("**Explicit API grid** (x/y from, to, step)")
                with gr.Group():
                    with gr.Row():
                        x_from = gr.Number(label="x from", value=0, precision=0)
                        x_to = gr.Number(label="x to", value=40, precision=0)
                        x_step = gr.Number(label="x step", value=20, precision=0)
                    with gr.Row():
                        y_from = gr.Number(label="y from", value=0, precision=0)
                        y_to = gr.Number(label="y to", value=40, precision=0)
                        y_step = gr.Number(label="y step", value=20, precision=0)
                gr.Markdown("**Target canvas** (needs width, height, anchor, API steps)")
                with gr.Group():
                    with gr.Row():
                        target_w = gr.Number(label="target width (px)", value=400, precision=0)
                        target_h = gr.Number(label="target height (px)", value=400, precision=0)
                    with gr.Row():
                        anchor_x = gr.Number(label="anchor X", value=0, precision=0)
                        anchor_y = gr.Number(label="anchor Y", value=0, precision=0)
                    with gr.Row():
                        api_x_step = gr.Number(label="API X step", value=20, precision=0)
                        api_y_step = gr.Number(label="API Y step", value=20, precision=0)
                gr.Markdown("**Crop** (PIL L U R B on 188px tile)")
                with gr.Row():
                    crop_l = gr.Slider(0, 80, value=36, step=1, label="left")
                    crop_u = gr.Slider(0, 80, value=36, step=1, label="upper")
                    crop_r = gr.Slider(100, 188, value=153, step=1, label="right")
                    crop_b = gr.Slider(100, 188, value=153, step=1, label="lower")
                gr.Markdown("**Nominal strides** (0 = auto from tile / target sizing)")
                with gr.Row():
                    nominal_dx = gr.Slider(0, 120, value=0, step=0.5, label="nominal dx")
                    nominal_dy = gr.Slider(0, 120, value=0, step=0.5, label="nominal dy")
                strides = gr.Radio(
                    choices=["matched", "nominal"],
                    value="matched",
                    label="Strides",
                )
                refine_phase = gr.Checkbox(label="Refine phase (after features)", value=False)
                phase_max_adj = gr.Slider(0.5, 20, value=6, step=0.5, label="Phase max adj (px)")
                min_overlap = gr.Slider(4, 80, value=16, step=1, label="Min overlap (px)")
                detector = gr.Dropdown(["akaze", "sift", "orb"], value="akaze", label="Detector")
                match_ratio = gr.Slider(0.5, 0.95, value=0.75, step=0.05, label="Match ratio")
                ransac_px = gr.Slider(0.5, 12, value=3, step=0.5, label="RANSAC (px)")
                roi_frac = gr.Slider(0.35, 0.85, value=0.55, step=0.05, label="ROI frac")
                blend = gr.Dropdown(
                    ["mean", "feather", "multiband", "seam"],
                    value="multiband",
                    label="Blend",
                )
                feather_px = gr.Slider(4, 80, value=28, step=1, label="Feather px")
                seam_smooth = gr.Slider(0, 12, value=2, step=1, label="Seam smooth")
                multiband_levels = gr.Slider(1, 8, value=4, step=1, label="Multiband levels")
                sharpen = gr.Slider(0, 1.2, value=0, step=0.05, label="Sharpen")
                sharpen_sigma = gr.Slider(0.4, 4, value=1.2, step=0.1, label="Sharpen sigma")
                with gr.Row():
                    workers = gr.Slider(1, 16, value=4, step=1, label="Download workers")
                    timeout = gr.Slider(5, 120, value=30, step=1, label="HTTP timeout (s)")
                max_tiles = gr.Slider(50, 5000, value=800, step=50, label="Max tiles (safety cap)")
                go = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                out_img = gr.Image(label="Mosaic (RGB preview)", type="numpy")
                out_log = gr.Textbox(label="Log", lines=22)

        inputs = [
            tile_source,
            base,
            params,
            tiles_dir,
            tile_pattern,
            tiles_precropped,
            grid_mode,
            x_from,
            x_to,
            x_step,
            y_from,
            y_to,
            y_step,
            target_w,
            target_h,
            anchor_x,
            anchor_y,
            api_x_step,
            api_y_step,
            crop_l,
            crop_u,
            crop_r,
            crop_b,
            nominal_dx,
            nominal_dy,
            strides,
            refine_phase,
            phase_max_adj,
            min_overlap,
            detector,
            match_ratio,
            ransac_px,
            roi_frac,
            blend,
            feather_px,
            seam_smooth,
            multiband_levels,
            sharpen,
            sharpen_sigma,
            workers,
            timeout,
            max_tiles,
        ]
        go.click(generate_ui, inputs=inputs, outputs=[out_img, out_log])

    return demo


if __name__ == "__main__":
    build_demo().launch()
