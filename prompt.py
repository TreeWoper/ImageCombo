import requests
import numpy as np
from PIL import Image
from io import BytesIO

USABLE = 117

def get_tile_by_pixel(px, py):
    """
    px, py are pixel coordinates, step by 117
    """
    url = "http://magnifier.flashphotography.com/MagnifyRender.ashx"
    params = {
        'X': px, 'Y': py,
        'O': '27341312', 'R': '10103',
        'F': '0194', 'A': '71714',
        'rand': '0.123'
    }
    r = requests.get(url, params=params)
    img = np.array(Image.open(BytesIO(r.content)))
    return img[36:153, 36:153]  # crop to usable region

def check_connection_correct():
    tile_00  = get_tile_by_pixel(0,   0)
    tile_10  = get_tile_by_pixel(117, 0)
    tile_01  = get_tile_by_pixel(0,   117)
    tile_11  = get_tile_by_pixel(117, 117)

    # Quick 2x2 stitch
    quick_stitch = np.zeros((117*2, 117*2, 3), dtype=np.uint8)
    quick_stitch[0:117,   0:117]   = tile_00
    quick_stitch[0:117,   117:234] = tile_10
    quick_stitch[117:234, 0:117]   = tile_01
    quick_stitch[117:234, 117:234] = tile_11

    Image.fromarray(quick_stitch).save('quick_stitch_corrected.png')

    # Check edges
    diff_h = np.abs(tile_00[:, -5:].astype(float) - tile_10[:, :5].astype(float))
    diff_v = np.abs(tile_00[-5:, :].astype(float) - tile_01[:5, :].astype(float))

    print(f"Horizontal edge diff: max={diff_h.max():.1f} mean={diff_h.mean():.4f}")
    print(f"Vertical edge diff:   max={diff_v.max():.1f} mean={diff_v.mean():.4f}")

def check_jitter_type():
    """
    Request same pixel coordinate multiple times
    to see if jitter is random or fixed
    """
    print("=== Requesting tile (0,0) 5 times ===")
    tiles = [get_tile_by_pixel(50, 576) for _ in range(5)]
    
    for i, tile in enumerate(tiles[1:], 1):
        diff = np.abs(tiles[0].astype(float) - tile.astype(float))
        print(f"Request {i} vs first: max={diff.max():.1f} mean={diff.mean():.4f}")
    
    print()
    print("=== Checking if edge diff is consistent across requests ===")
    
    # Get tile_00 and tile_10 multiple times and check edge diff each time
    for i in range(5):
        t00 = get_tile_by_pixel(0, 0)
        t10 = get_tile_by_pixel(117, 0)
        diff = np.abs(t00[:, -1].astype(float) - t10[:, 0].astype(float))
        print(f"Request {i}: edge diff max={diff.max():.1f} mean={diff.mean():.4f}")

def check_true_overlap():
    """
    The real question is: does tile at X=117 start exactly
    where tile at X=0 ends, or is there a gap/overlap?
    """
    t00 = get_tile_by_pixel(0, 0)
    t10 = get_tile_by_pixel(117, 0)
    
    # If perfectly adjacent, last col of t00 should be
    # visually continuous with first col of t10
    # Save them side by side with just 1px boundary
    comparison = np.hstack([t00, t10])
    Image.fromarray(comparison).save('side_by_side.png')
    
    # Also try with overlap - what if step should not be 117
    # but something slightly different like 116 or 118?
    for step in range(110, 125):
        t_step = get_tile_by_pixel(step, 0)
        diff = np.abs(t00[:, -1].astype(float) - t_step[:, 0].astype(float))
        print(f"Step {step}: edge diff max={diff.max():.1f} mean={diff.mean():.4f}")

def refine_step():
    t00 = get_tile_by_pixel(0, 0)
    
    # Zoom in around 116 for horizontal
    print("=== Horizontal fine search around 116 ===")
    for step in range(113, 120):
        t = get_tile_by_pixel(step, 0)
        diff = np.abs(t00[:, -1].astype(float) - t[:, 0].astype(float))
        print(f"Step {step}: max={diff.max():.1f} mean={diff.mean():.4f}")
    
    print()
    
    # Check vertical step size too
    print("=== Vertical step search ===")
    for step in range(110, 125):
        t = get_tile_by_pixel(0, step)
        diff = np.abs(t00[-1, :].astype(float) - t[0, :].astype(float))
        print(f"Step {step}: max={diff.max():.1f} mean={diff.mean():.4f}")

    print()

    # Also check if the remaining diff at step 116
    # is consistent across different tile pairs
    print("=== Checking step 116 across multiple tile pairs ===")
    for x in range(5):
        ta = get_tile_by_pixel(x * 116, 0)
        tb = get_tile_by_pixel(x * 116 + 116, 0)
        diff = np.abs(ta[:, -1].astype(float) - tb[:, 0].astype(float))
        print(f"Tiles ({x},0)->({x+1},0): max={diff.max():.1f} mean={diff.mean():.4f}")

refine_step()