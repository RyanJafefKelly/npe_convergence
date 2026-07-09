#!/usr/bin/env python3
"""Assemble generated pose-row images into a Codex custom pet spritesheet."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFilter


FRAME_W = 192
FRAME_H = 208
COLS = 8
ROWS = 9
SHEET_W = FRAME_W * COLS
SHEET_H = FRAME_H * ROWS
DEFAULT_PET_DIR = Path.home() / ".codex" / "pets" / "angry-fiancee"


ROW_SLUGS = (
    "idle",
    "running-right",
    "running-left",
    "waving",
    "jumping",
    "failed-sad",
    "waiting",
    "running",
    "review-alert",
)


ROW_VERTICAL_OFFSETS = {
    "idle": [0, -1, -2, -1, 0, 1, 0, 0],
    "running-right": [0, -2, 0, 2, 0, -2, 0, 2],
    "running-left": [0, -2, 0, 2, 0, -2, 0, 2],
    "waving": [0, -1, -2, -1, 0, 0, -1, 0],
    "jumping": [0, -10, -28, -42, -28, -10, 0, 2],
    "failed-sad": [5, 6, 7, 6, 5, 6, 7, 6],
    "waiting": [0, 0, -1, 0, 0, 0, -1, 0],
    "running": [0, -2, 0, 2, 0, -2, 0, 2],
    "review-alert": [0, -1, -3, -1, 0, -1, -2, 0],
}


def sample_key(image: Image.Image) -> tuple[int, int, int]:
    image = image.convert("RGBA")
    pix = image.load()
    w, h = image.size
    points = []
    for x in [0, w // 2, w - 1]:
        points.extend([(x, 0), (x, h - 1)])
    for y in [h // 4, h // 2, (3 * h) // 4]:
        points.extend([(0, y), (w - 1, y)])
    samples = [pix[x, y][:3] for x, y in points]
    return tuple(round(sum(s[i] for s in samples) / len(samples)) for i in range(3))


def remove_green_background(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    key = sample_key(image)
    pix = image.load()
    w, h = image.size

    for y in range(h):
        for x in range(w):
            r, g, b, a = pix[x, y]
            if a == 0:
                continue
            dist = math.sqrt((r - key[0]) ** 2 + (g - key[1]) ** 2 + (b - key[2]) ** 2)
            green_score = g - max(r, b)
            key_is_green = key[1] > max(key[0], key[2]) + 40

            remove = False
            if dist <= 38:
                remove = True
            elif key_is_green and g > 120 and green_score > 34:
                remove = True
            elif g > 180 and r < 80 and b < 80:
                remove = True

            if remove:
                pix[x, y] = (r, g, b, 0)
                continue

            if key_is_green and dist <= 130 and g > r * 1.08 and g > b * 1.08:
                alpha = int(a * min(1.0, max(0.0, (dist - 38) / 92)))
                pix[x, y] = (r, min(g, max(r, b) + 12), b, alpha)
                continue

            if green_score > 8:
                pix[x, y] = (r, min(g, max(r, b) + 8), b, a)

    alpha = image.getchannel("A").point(lambda v: 0 if v < 28 else v)
    alpha = alpha.filter(ImageFilter.MedianFilter(3))
    image.putalpha(alpha)
    return image


def connected_components(image: Image.Image, min_area: int = 1) -> list[dict]:
    alpha = image.getchannel("A")
    w, h = image.size
    mask = alpha.load()
    seen = bytearray(w * h)
    components: list[dict] = []

    def idx(x: int, y: int) -> int:
        return y * w + x

    for sy in range(h):
        for sx in range(w):
            start = idx(sx, sy)
            if seen[start] or mask[sx, sy] <= 30:
                continue
            queue: deque[tuple[int, int]] = deque([(sx, sy)])
            seen[start] = 1
            comp: list[tuple[int, int]] = []
            min_x = max_x = sx
            min_y = max_y = sy
            while queue:
                x, y = queue.popleft()
                comp.append((x, y))
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    pos = idx(nx, ny)
                    if seen[pos] or mask[nx, ny] <= 30:
                        continue
                    seen[pos] = 1
                    queue.append((nx, ny))
            if len(comp) >= min_area:
                components.append(
                    {
                        "area": len(comp),
                        "bbox": (min_x, min_y, max_x + 1, max_y + 1),
                        "center": ((min_x + max_x + 1) / 2, (min_y + max_y + 1) / 2),
                        "pixels": comp,
                    }
                )

    return components


def keep_large_components(image: Image.Image, min_area: int = 24) -> Image.Image:
    image = image.convert("RGBA")
    w, h = image.size
    components = connected_components(image, min_area=min_area)
    keep = bytearray(w * h)

    def idx(x: int, y: int) -> int:
        return y * w + x

    if components:
        largest = max(components, key=lambda c: c["area"])
        lx1, ly1, lx2, ly2 = largest["bbox"]
        lcx, lcy = largest["center"]
        largest_area = largest["area"]
        expanded_largest = (lx1 - 55, ly1 - 55, lx2 + 55, ly2 + 55)

        for comp in components:
            x1, y1, x2, y2 = comp["bbox"]
            cx, cy = comp["center"]
            touches_edge = x1 <= 3 or y1 <= 3 or x2 >= w - 3 or y2 >= h - 3
            close_to_body = (
                expanded_largest[0] <= cx <= expanded_largest[2]
                and expanded_largest[1] <= cy <= expanded_largest[3]
            )
            substantial = comp["area"] >= max(24, largest_area * 0.012)
            keep_component = comp is largest or (close_to_body and substantial and not touches_edge)
            if not keep_component:
                continue
            for x, y in comp["pixels"]:
                keep[idx(x, y)] = 1

    pix = image.load()
    for y in range(h):
        for x in range(w):
            if not keep[idx(x, y)]:
                r, g, b, _ = pix[x, y]
                pix[x, y] = (r, g, b, 0)
    return image


def trim_alpha(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A").point(lambda v: 255 if v > 30 else 0)
    bbox = alpha.getbbox()
    if bbox is None:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    return image.crop(bbox)


def fit_sprite(image: Image.Image, max_w: int = 166, max_h: int = 184) -> Image.Image:
    image = trim_alpha(image)
    if image.width <= 1 or image.height <= 1:
        return image
    scale = min(max_w / image.width, max_h / image.height, 1.0)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.Resampling.NEAREST)


def frame_from_cell(cell: Image.Image, row_slug: str, col: int) -> Image.Image:
    sprite = keep_large_components(remove_green_background(cell))
    sprite = fit_sprite(sprite)

    frame = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    if sprite.width <= 1 or sprite.height <= 1:
        return frame
    x = (FRAME_W - sprite.width) // 2
    y = FRAME_H - sprite.height - 12 + ROW_VERTICAL_OFFSETS[row_slug][col]
    y = max(2, min(FRAME_H - sprite.height - 2, y))
    frame.alpha_composite(sprite, (x, y))
    return final_despeckle(frame)


def final_despeckle(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    pix = image.load()
    w, h = image.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = pix[x, y]
            if a == 0:
                continue
            if (g > 130 and g - max(r, b) > 30) or (g > 180 and r < 90 and b < 90):
                pix[x, y] = (r, min(g, max(r, b) + 8), b, 0 if a < 180 else a)
    return image


def row_path(source_dir: Path, row: int, slug: str) -> Path:
    candidates = [
        source_dir / "raw-rows" / f"{row:02d}-{slug}.png",
        source_dir / f"{row:02d}-{slug}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing generated row image for {row:02d}-{slug}")


def slice_row(row_image: Image.Image, row_slug: str) -> list[Image.Image]:
    row_image = row_image.convert("RGBA")
    cleaned = remove_green_background(row_image)
    min_body_area = max(250, int(row_image.width * row_image.height * 0.0008))
    components = [
        comp
        for comp in connected_components(cleaned, min_area=min_body_area)
        if (comp["bbox"][3] - comp["bbox"][1]) >= row_image.height * 0.12
    ]
    body_components = sorted(sorted(components, key=lambda c: c["area"], reverse=True)[:COLS], key=lambda c: c["center"][0])
    if len(body_components) == COLS:
        frames = []
        pad = 34
        for col, comp in enumerate(body_components):
            x1, y1, x2, y2 = comp["bbox"]
            crop = cleaned.crop(
                (
                    max(0, x1 - pad),
                    max(0, y1 - pad),
                    min(cleaned.width, x2 + pad),
                    min(cleaned.height, y2 + pad),
                )
            )
            frames.append(frame_from_cell(crop, row_slug, col))
        return frames

    frames: list[Image.Image] = []
    for col in range(COLS):
        left = round(col * row_image.width / COLS)
        right = round((col + 1) * row_image.width / COLS)
        cell = cleaned.crop((left, 0, right, row_image.height))
        frames.append(frame_from_cell(cell, row_slug, col))
    return frames


def save_previews(sheet: Image.Image, source_dir: Path) -> None:
    preview_dir = source_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(preview_dir / "spritesheet-preview.png")
    sample = sheet.crop((0, 0, FRAME_W, FRAME_H))
    sample.save(preview_dir / "frame-00-idle.png")
    scaled = sheet.resize((SHEET_W // 2, SHEET_H // 2), Image.Resampling.NEAREST)
    scaled.save(preview_dir / "spritesheet-preview-half.png")


def backup_existing(sheet_path: Path) -> Path | None:
    if not sheet_path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = sheet_path.with_name(f"{sheet_path.stem}.backup-{timestamp}{sheet_path.suffix}")
    shutil.copy2(sheet_path, backup_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pet-dir", type=Path, default=DEFAULT_PET_DIR)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    pet_dir = args.pet_dir.expanduser()
    source_dir = (args.source_dir or pet_dir / "source-poses").expanduser()
    frames_dir = source_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    sheet = Image.new("RGBA", (SHEET_W, SHEET_H), (0, 0, 0, 0))
    for row, slug in enumerate(ROW_SLUGS):
        img = Image.open(row_path(source_dir, row, slug))
        frames = slice_row(img, slug)
        for col, frame in enumerate(frames):
            frame_path = frames_dir / f"{row:02d}-{slug}-{col:02d}.png"
            frame.save(frame_path)
            sheet.alpha_composite(frame, (col * FRAME_W, row * FRAME_H))

    sheet = final_despeckle(sheet)
    if sheet.size != (SHEET_W, SHEET_H):
        raise AssertionError(f"bad sheet size: {sheet.size}")

    sheet_path = pet_dir / "spritesheet.webp"
    backup_path = None if args.no_backup else backup_existing(sheet_path)
    sheet.save(sheet_path, "WEBP", lossless=True, method=6)
    save_previews(sheet, source_dir)

    manifest_path = pet_dir / "pet.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("spritesheetPath") != "spritesheet.webp":
        raise AssertionError(f"pet.json spritesheetPath is {manifest.get('spritesheetPath')!r}, expected 'spritesheet.webp'")

    print(f"wrote {sheet_path}")
    if backup_path:
        print(f"backup {backup_path}")
    print(f"size {sheet.size[0]}x{sheet.size[1]}")
    print(f"preview {source_dir / 'preview' / 'spritesheet-preview.png'}")
    print(f"sample {source_dir / 'preview' / 'frame-00-idle.png'}")


if __name__ == "__main__":
    main()
