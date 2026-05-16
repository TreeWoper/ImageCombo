# imageStitch

Tools for downloading **Flash Photography** magnifier tiles and stitching them into a single image. The magnifier API returns fixed-size JPEGs with a frame; the usable photo is usually a crop in the center (often `36 36 153 153` on 188×188 tiles).

Two stitchers are included:

| Script | Role |
|--------|------|
| `stitch_magnifier.py` | Download + paste with measured overlap; **calibration** to find crop and API step |
| `stitch_feature_mosaic.py` | Feature matching (AKAZE/SIFT/ORB), optional phase refine, richer blending |
| `stitch_feature_mosaic_ui.py` | Gradio UI to tune `stitch_feature_mosaic` interactively |

Typical workflow: calibrate with `stitch_magnifier.py`, download tiles (optional), then stitch with either script or the UI.

---

## Setup

From the project root:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

You need **Python 3.10+** and a working OpenCV build (`opencv-python`).

---

## Getting `params` from the viewer

In the magnifier, open a tile URL (or network tab). You want the query string **after** `X=` and `Y=`, for example:

```text
O=27341312&R=10103&F=0194&A=71714
```

Pass that string to `--params` (CLI) or the **Params** field (UI). `X` and `Y` are added automatically by the scripts.

---

## 1. `stitch_magnifier.py` — calibration and simple stitch

### Calibrate (recommended first)

Fetches a few neighbor tiles and prints suggested `--crop`, `--x-step`, and `--y-step`:

```bash
python stitch_magnifier.py --params "O=...&R=...&F=...&A=..." --calibrate --auto-stride
```

Use the printed values for a full download. Optional tuning:

| Option | Default | Meaning |
|--------|---------|---------|
| `--cal-x0`, `--cal-y0` | 256, 125 | Anchor API X/Y for calibration tiles |
| `--cal-dx`, `--cal-dy` | 44, 44 | API step between calibration pairs |
| `--cal-strip` | 25 | Band size when scoring seams |
| `--stride-min`, `--stride-max` | 36, 54 | API stride search range with `--auto-stride` |

### Full stitch

```bash
python stitch_magnifier.py ^
  --params "O=...&R=...&F=...&A=..." ^
  --x-from 0 --x-to 500 --x-step 44 ^
  --y-from 0 --y-to 700 --y-step 44 ^
  --crop 36 36 153 153 ^
  --layout auto ^
  --overlap-blend ^
  --out output/stitched.jpg
```

Save every tile while downloading (for reuse with the feature mosaic):

```bash
python stitch_magnifier.py ... --tiles-dir tiles/
```

Files are named `raw_{x}_{y}.jpg` and `crop_{x}_{y}.jpg`.

### All options (`stitch_magnifier.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--base` | MagnifyRender URL | Tile endpoint |
| `--params` | *(required)* | Query fragment after X/Y |
| `--x-from`, `--x-to`, `--x-step` | — | API X grid |
| `--y-from`, `--y-to`, `--y-step` | — | API Y grid |
| `--crop` | `36 36 105 105` | PIL crop: left upper right lower |
| `--out` | `stitched.jpg` | Output image path |
| `--tiles-dir` | — | Save raw + cropped tiles here |
| `--timeout` | 30 | HTTP timeout (seconds) |
| `--delay` | 0 | Pause after each download |
| `--workers` | 6 | Parallel downloads |
| `--calibrate` | off | Print crop/step hints only |
| `--auto-stride` | off | Search best API stride (with calibrate) |
| `--layout` | `auto` | `auto` = overlap by measured shift; `abut` = edge-to-edge |
| `--per-edge-strides` | off | One measured stride per column/row boundary |
| `--overlap-blend` | off | Blend RGB in overlaps (good for JPEG tiles) |
| `--paste-x`, `--paste-y` | — | Override paste stride in pixels |
| `--seam-smooth` | 0 | Soften tile boundaries (0–3) |

---

## 2. `stitch_feature_mosaic.py` — feature-based mosaic

Best when API X/Y are **jittery** but **repeatable** (same X,Y → same image). It measures real pixel offsets between neighbors, then blends.

### From the API

**Explicit grid** — you choose API indices:

```bash
python stitch_feature_mosaic.py ^
  --params "O=...&R=...&F=...&A=..." ^
  --x-from 0 --x-to 40 --x-step 20 ^
  --y-from 0 --y-to 40 --y-step 20 ^
  --crop 36 36 153 153 ^
  --strides matched ^
  --blend feather ^
  --out output/mosaic.jpg
```

**Target canvas** — you choose minimum output size; the script counts tiles:

```bash
python stitch_feature_mosaic.py ^
  --params "O=...&R=...&F=...&A=..." ^
  --target-width 500 --target-height 700 ^
  --anchor-x 0 --anchor-y 0 ^
  --api-x-step 20 --api-y-step 20 ^
  --nominal-dx 44 --nominal-dy 44 ^
  --crop 36 36 153 153 ^
  --out output/mosaic.jpg
```

### From a local folder

Use tiles saved by `stitch_magnifier.py --tiles-dir tiles/` (or your own names):

```bash
python stitch_feature_mosaic.py ^
  --tiles-dir tiles ^
  --tile-pattern auto ^
  --x-from 0 --x-to 40 --x-step 20 ^
  --y-from 0 --y-to 40 --y-step 20 ^
  --crop 36 36 153 153 ^
  --blend feather ^
  --out output/mosaic.jpg
```

Pre-cropped files only (`crop_0_0.jpg`, …):

```bash
python stitch_feature_mosaic.py ^
  --tiles-dir tiles ^
  --tile-pattern crop ^
  --tiles-precropped ^
  ...
```

### All options (`stitch_feature_mosaic.py`)

#### Tile source

| Option | Default | Description |
|--------|---------|-------------|
| `--base` | MagnifyRender URL | Used when downloading |
| `--params` | — | Required for API; omit with `--tiles-dir` |
| `--tiles-dir` | — | Load tiles from folder instead of API |
| `--tile-pattern` | `auto` | `auto`, `raw`, `crop`, or `tile_{x}_{y}` |
| `--tiles-precropped` | off | Skip `--crop` (files already interior) |

#### Grid (pick one mode)

| Mode | Flags |
|------|--------|
| Explicit | `--x-from`, `--x-to`, `--x-step`, `--y-from`, `--y-to`, `--y-step` |
| Target size | `--target-width`, `--target-height`, `--anchor-x`, `--anchor-y`, `--api-x-step`, `--api-y-step` |

| Option | Default | Description |
|--------|---------|-------------|
| `--max-tiles` | 25000 | Abort if grid is too large |

#### Geometry and matching

| Option | Default | Description |
|--------|---------|-------------|
| `--crop` | `36 36 153 153` | Four integers: left upper right lower |
| `--nominal-dx`, `--nominal-dy` | auto | Expected pixel stride (from calibration) |
| `--strides` | `matched` | `matched` = features (+ optional phase); `nominal` = trust nominals only |
| `--detector` | `akaze` | `akaze`, `sift`, or `orb` |
| `--match-ratio` | 0.75 | Descriptor match strictness |
| `--ransac-px` | 3.0 | RANSAC inlier threshold |
| `--roi-frac` | 0.55 | Overlap strip fraction for matching |
| `--refine-phase` | off | Subpixel phase correlation after features |
| `--phase-max-adj` | 6.0 | Max phase adjustment (px) |
| `--min-overlap` | 16 | Clamp strides so tiles still overlap |

#### Blending and output

| Option | Default | Description |
|--------|---------|-------------|
| `--blend` | `multiband` | `mean`, `feather`, `multiband`, `seam` |
| `--feather-px` | 28 | Edge falloff for feather / seam |
| `--seam-smooth` | 2 | Soften seam mask |
| `--multiband-levels` | 4 | Pyramid levels (lower = sharper, less blur) |
| `--sharpen` | 0 | Unsharp amount after blend (0 = off) |
| `--sharpen-sigma` | 1.2 | Unsharp Gaussian sigma |
| `--out` | `mosaic_feature.jpg` | Output path (.png supported) |
| `--timeout`, `--delay`, `--workers` | 30, 0, 4 | Download behavior |

---

## 3. `stitch_feature_mosaic_ui.py` — interactive lab

```bash
python stitch_feature_mosaic_ui.py
```

Open the URL printed in the terminal (usually `http://127.0.0.1:7860`).

- **Tile source**: API or **Folder** (path like `tiles/` relative to where you run the command).
- **Grid mode**: explicit API range or target canvas (same as CLI).
- Sliders mirror CLI options; click **Generate** to run (each run reloads tiles).

Use a **small grid** first while tuning, then scale up.

---

## Choosing parameters (quick guide)

| Situation | Suggestion |
|-----------|------------|
| Unknown crop / step | `stitch_magnifier.py --calibrate --auto-stride` |
| Trust calibration, fast stitch | `--strides nominal` + nominals from calibrate |
| API jitter, same X,Y repeatable | `--strides matched`, `--blend feather` or `seam` |
| Ghosting / blur in overlaps | Try `feather` or `seam`; lower `--multiband-levels`; check nominals |
| Mostly blank mosaic | Wrong API range or strides too large; lower range or `--min-overlap` |
| Re-stitch without re-downloading | `stitch_magnifier --tiles-dir tiles/` then `stitch_feature_mosaic --tiles-dir tiles` |

**Nominal dx/dy** = guessed pixel distance between tile centers in the **cropped** image. **API x-step / y-step** = how much the magnifier’s X/Y indices change per click — related but not the same number.

---

## Project layout

```text
imageStitch/
  README.md
  requirements.txt
  stitch_magnifier.py          # Calibrate + PIL stitch
  stitch_feature_mosaic.py     # OpenCV feature mosaic
  stitch_feature_mosaic_ui.py  # Gradio UI
  tiles/                       # Optional: saved tiles (gitignored)
  output/                      # Optional: results (gitignored)
```

---

## Note

- Tiles are **JPEG**; small differences at seams are normal. Overlap blending helps more than expecting pixel-perfect equality.
- Blank or white tiles usually mean X/Y are outside the photo — try `--anchor-x 0 --anchor-y 0` or a smaller range.
- This project is for educational / personal use with the Flash Photography magnifier; you are responsible for complying with their terms of use.
