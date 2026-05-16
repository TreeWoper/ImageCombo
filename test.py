import requests
import shutil
import time
import os
from PIL import Image

# X = input()
# Y = input()


optput_path = 'data/test'
os.makedirs(optput_path, exist_ok=True)

def get_tile(x, y):
    file = optput_path + '/raw_' + str(x) + '_' + str(y) + '.jpg'
    url_path = 'http://magnifier.flashphotography.com/MagnifyRender.ashx?X='+ str(x) + '&Y=' + str(y) + '&O=27341312&R=10103&F=0194&A=71714'
    r = requests.get(url_path, stream=True)
    if r.status_code == 200:
        with open(file, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

def crop_tile(x1, y1, x2, y2, name='raw_0_0.jpg'):
    input_path = os.path.join('data/test', name)

    img = Image.open(input_path)
    cropped = img.crop((x1, y1, x2, y2))
    print(img.size)

    out_name = f"crop_{name}"
    output_path = os.path.join('data/test', out_name)
    cropped.save(output_path)

    print("Saved:", output_path, "size:", cropped.size)


def combine_two_tiles(tile1_path, tile2_path, output_path):
    img1 = Image.open(tile1_path)
    img2 = Image.open(tile2_path)

    w1, h1 = img1.size
    w2, h2 = img2.size

    # height must match — if not, resize or pad
    new_w = w1 + w2
    new_h = max(h1, h2)

    new_img = Image.new("RGB", (new_w, new_h))

    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (w1, 0))

    new_img.save(output_path)
    print("Saved:", output_path)

get_tile(59, 0)
crop_tile(0, 0, 59, 58)
crop_tile(0, 0, 59, 58, 'raw_59_0.jpg')
combine_two_tiles('data/test/crop_raw_0_0.jpg', 'data/test/crop_raw_59_0.jpg', 'data/test/combined.jpg')
