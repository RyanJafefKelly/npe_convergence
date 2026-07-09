#!/usr/bin/env python3
"""Build a Codex custom pet spritesheet from a green-background sprite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter


FRAME_W = 192
FRAME_H = 208
COLS = 8
ROWS = 9
SHEET_W = FRAME_W * COLS
SHEET_H = FRAME_H * ROWS


def remove_green_background(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    pixels = image.load()
    width, height = image.size

    sample_points = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
        (width // 2, 0),
        (width // 2, height - 1),
    ]
    samples = [pixels[x, y][:3] for x, y in sample_points]
    key = tuple(round(sum(color[i] for color in samples) / len(samples)) for i in range(3))

    # ChatGPT's chroma key is usually pure or near-pure green, but downloads can
    # contain antialiasing, compression, and faint generated shadows. Remove by
    # both distance-to-key and green dominance.
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            dist = math.sqrt((r - key[0]) ** 2 + (g - key[1]) ** 2 + (b - key[2]) ** 2)
            green_score = g - max(r, b)

            if dist <= 42 or (g > 135 and green_score > 38):
                pixels[x, y] = (r, g, b, 0)
                continue

            if dist <= 120 and g > r * 1.12 and g > b * 1.12:
                edge_alpha = int(a * min(1.0, max(0.0, (dist - 42) / 78)))
                pixels[x, y] = (r, min(g, max(r, b) + 16), b, edge_alpha)
                continue

            if green_score > 12:
                pixels[x, y] = (r, min(g, max(r, b) + 12), b, a)

    alpha = image.getchannel("A")
    alpha = alpha.point(lambda v: 0 if v < 36 else v)
    alpha = alpha.filter(ImageFilter.MedianFilter(3))
    image.putalpha(alpha)
    return image


def trim_transparent(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A").point(lambda v: 255 if v > 36 else 0)
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("No non-transparent pixels remain after background removal.")
    return image.crop(bbox)


def pixel_resize(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / image.width, max_h / image.height, 1.0)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.Resampling.NEAREST)


def shadow(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    blur = alpha.filter(ImageFilter.GaussianBlur(1.2))
    out = Image.new("RGBA", image.size, (0, 0, 0, 0))
    out.putalpha(blur.point(lambda v: min(90, int(v * 0.35))))
    return out


def composite_frame(base: Image.Image, frame: Image.Image, dx: int, dy: int) -> None:
    x = (FRAME_W - frame.width) // 2 + dx
    y = FRAME_H - frame.height - 12 + dy
    base.alpha_composite(frame, (x, y))


def build_sheet(sprite: Image.Image) -> Image.Image:
    sprite = pixel_resize(trim_transparent(remove_green_background(sprite)), 150, 170)
    mirrored = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    sprite_shadow = shadow(sprite)
    mirrored_shadow = shadow(mirrored)

    sheet = Image.new("RGBA", (SHEET_W, SHEET_H), (0, 0, 0, 0))

    for row in range(ROWS):
        for col in range(COLS):
            frame = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
            img = sprite
            img_shadow = sprite_shadow
            dx = 0
            dy = 0

            if row == 0:  # idle
                dy = [0, -1, -2, -1, 0, 1, 0, 0][col]
            elif row == 1:  # running right
                dx = [-2, -1, 0, 1, 2, 1, 0, -1][col]
                dy = [0, -2, 0, 2, 0, -2, 0, 2][col]
            elif row == 2:  # running left
                img = mirrored
                img_shadow = mirrored_shadow
                dx = [2, 1, 0, -1, -2, -1, 0, 1][col]
                dy = [0, -2, 0, 2, 0, -2, 0, 2][col]
            elif row == 3:  # waving
                dy = [0, -1, -2, -1, 0, 0, 0, 0][col]
            elif row == 4:  # jumping
                dy = [0, -8, -16, -8, 0, 0, 0, 0][col]
            elif row == 5:  # failed/sad
                dy = [3, 4, 5, 4, 3, 4, 5, 4][col]
            else:  # waiting/running/review placeholders
                dy = [0, -1, 0, 1, 0, -1, 0, 1][col]

            # A faint contact shadow helps the pet sit on the overlay without
            # baking in a large generated-image shadow.
            composite_frame(frame, img_shadow, dx, dy + 2)
            composite_frame(frame, img, dx, dy)
            sheet.alpha_composite(frame, (col * FRAME_W, row * FRAME_H))

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--pet-id", default="fiancee")
    parser.add_argument("--display-name", default="Fiancee")
    parser.add_argument("--description", default="A tiny pixel-art companion.")
    parser.add_argument("--out-root", type=Path, default=Path.home() / ".codex" / "pets")
    args = parser.parse_args()

    source = args.source.expanduser()
    pet_dir = args.out_root.expanduser() / args.pet_id
    pet_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(source)
    sheet = build_sheet(image)
    if sheet.size != (SHEET_W, SHEET_H):
        raise AssertionError(f"bad sheet size: {sheet.size}")

    sheet_path = pet_dir / "spritesheet.webp"
    sheet.save(sheet_path, "WEBP", lossless=True, method=6)

    manifest = {
        "displayName": args.display_name,
        "description": args.description,
        "spritesheetPath": "spritesheet.webp",
    }
    (pet_dir / "pet.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {sheet_path}")
    print(f"Wrote {pet_dir / 'pet.json'}")
    print(f"Spritesheet size: {sheet.size[0]}x{sheet.size[1]}")


if __name__ == "__main__":
    main()
