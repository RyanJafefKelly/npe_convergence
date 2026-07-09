#!/usr/bin/env python3
"""Generate pose-row source images for a Codex custom pet.

This intentionally asks the image model for one animation row at a time rather
than for the final Codex spritesheet. The assembly step is handled locally by
assemble_codex_pet.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import string
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


API_URL = "https://api.openai.com/v1/images/edits"
DEFAULT_PET_DIR = Path.home() / ".codex" / "pets" / "angry-fiancee"
DEFAULT_MODELS = ("gpt-image-2", "gpt-image-1.5", "gpt-image-1")


@dataclass(frozen=True)
class PoseSpec:
    row: int
    slug: str
    title: str
    action: str


POSES = (
    PoseSpec(0, "idle", "idle breathing and subtle bobbing", "standing in place, angry breathing/bobbing loop, tiny shoulder and head motion"),
    PoseSpec(1, "running-right", "running to the right", "running toward screen right with a clear eight-frame stride cycle, legs and arms changing pose"),
    PoseSpec(2, "running-left", "running to the left", "running toward screen left with a clear eight-frame stride cycle, legs and arms changing pose"),
    PoseSpec(3, "waving", "angry waving", "standing while angrily waving one raised hand, arm positions changing across the loop"),
    PoseSpec(4, "jumping", "jumping", "jumping cycle from crouch to airborne to landing, angry face maintained"),
    PoseSpec(5, "failed-sad", "failed or sad slumped pose", "slumped disappointed failure animation, head and shoulders droop while staying visibly angry-sad"),
    PoseSpec(6, "waiting", "waiting impatiently", "impatient waiting loop with foot tapping and tense posture"),
    PoseSpec(7, "running", "running in place", "frustrated running-in-place loop facing mostly forward, energetic arm and leg motion"),
    PoseSpec(8, "review-alert", "review or alert attention pose", "alert attention loop, upright angry stance, one hand or body snap indicating review needed"),
)


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def build_prompt(pose: PoseSpec) -> str:
    return f"""
Use case: stylized-concept
Asset type: source pose row for a tiny Codex custom pet spritesheet.
Primary request: Create exactly 8 consistent full-body pixel-art animation frames in one horizontal row for this pose group: {pose.title}.
Reference image: Use the uploaded image only as the identity and style reference.
Character invariants: same character as reference; same outfit; same glasses; same hairstyle; same angry fiancee expression; same full-body chibi pixel-art scale; same proportions in every frame.
Pose/action: {pose.action}.
Layout: one single horizontal strip with exactly 8 evenly spaced full-body frames, no more and no fewer. Each frame should be separated by empty background only. No panel borders, no grid lines, no labels, no numbering, no captions, no text, no watermark.
Camera/style: straight-on orthographic pixel art, crisp sprite-game look, consistent lighting, consistent scale, centered character in each frame, generous padding around the character.
Background: perfectly flat solid #00ff00 chroma-key background across the entire image.
Background constraints: no shadows, no cast shadow, no contact shadow, no gradients, no texture, no floor plane, no reflections, no props, and do not use #00ff00 anywhere in the character.
Output constraints: the character must be fully visible in every frame, never cropped, and all 8 frames must be useful as animation frames after local background removal.
""".strip()


def multipart_body(fields: dict[str, str], files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    boundary = "----codex-pet-" + "".join(random.choice(string.ascii_letters + string.digits) for _ in range(24))
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )

    for field, path in files:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'.encode(),
                f"Content-Type: {mime}\r\n\r\n".encode(),
                path.read_bytes(),
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def post_image_edit(api_key: str, fields: dict[str, str], image_path: Path) -> dict:
    attempts: list[tuple[str, dict[str, str]]] = []
    base = dict(fields)
    attempts.append(("image[]", base))
    attempts.append(("image", base))

    # Some model revisions reject newer optional knobs. Keep the prompt/model/size
    # stable and progressively remove optional controls if needed.
    stripped = {k: v for k, v in base.items() if k not in {"input_fidelity", "quality", "output_format"}}
    attempts.append(("image[]", stripped))
    attempts.append(("image", stripped))

    last_error: str | None = None
    for image_field, request_fields in attempts:
        body, boundary = multipart_body(request_fields, [(image_field, image_path)])
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(payload)
                last_error = json.dumps(parsed.get("error", parsed), indent=2)
            except json.JSONDecodeError:
                last_error = payload

            # Retry on parameter/field compatibility failures, but fail fast for
            # auth, billing, or policy errors that another field name will not fix.
            if exc.code in {400, 404, 422}:
                continue
            die(f"OpenAI image request failed with HTTP {exc.code}: {last_error}")
        except urllib.error.URLError as exc:
            die(f"OpenAI image request failed: {exc}")

    die(f"OpenAI image request failed after compatibility retries: {last_error}")


def decode_image_response(data: dict, out_path: Path) -> None:
    items = data.get("data") or []
    if not items:
        die(f"OpenAI response did not include image data: {json.dumps(data)[:500]}")
    first = items[0]
    if first.get("b64_json"):
        out_path.write_bytes(base64.b64decode(first["b64_json"]))
        return
    if first.get("url"):
        with urllib.request.urlopen(first["url"], timeout=300) as response:
            out_path.write_bytes(response.read())
        return
    die(f"OpenAI response did not include b64_json or url: {json.dumps(first)[:500]}")


def model_fields(model: str, prompt: str, size: str, quality: str) -> dict[str, str]:
    fields = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "output_format": "png",
        "input_fidelity": "high",
    }
    return fields


def selected_poses(slugs: Iterable[str]) -> list[PoseSpec]:
    wanted = set(slugs)
    if not wanted:
        return list(POSES)
    by_slug = {p.slug: p for p in POSES}
    missing = sorted(wanted - set(by_slug))
    if missing:
        die(f"unknown pose slug(s): {', '.join(missing)}")
    return [p for p in POSES if p.slug in wanted]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pet-dir", type=Path, default=DEFAULT_PET_DIR)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--size", default="1536x1024")
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--force", action="store_true", help="Regenerate rows even when output files already exist.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        die("OPENAI_API_KEY is not set")

    pet_dir = args.pet_dir.expanduser()
    out_dir = (args.out_dir or pet_dir / "source-poses").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = (args.reference or out_dir / "identity_reference.png").expanduser()
    if not reference.exists():
        die(f"reference image does not exist: {reference}")

    rows_dir = out_dir / "raw-rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, str | int]] = []
    for pose in selected_poses(args.only):
        prompt = build_prompt(pose)
        prompt_path = prompt_dir / f"{pose.row:02d}-{pose.slug}.txt"
        prompt_path.write_text(prompt + "\n")
        out_path = rows_dir / f"{pose.row:02d}-{pose.slug}.png"
        if out_path.exists() and not args.force:
            print(f"skip existing {out_path}")
            summary.append({"row": pose.row, "slug": pose.slug, "path": str(out_path), "model": "existing"})
            continue

        row_done = False
        last_error: str | None = None
        for model in args.models:
            print(f"generating row {pose.row} {pose.slug} with {model}", flush=True)
            fields = model_fields(model, prompt, args.size, args.quality)
            try:
                data = post_image_edit(api_key, fields, reference)
            except SystemExit as exc:
                last_error = str(exc)
                # Try the next model for model-specific failures.
                if model != args.models[-1]:
                    continue
                raise

            decode_image_response(data, out_path)
            metadata = {
                "row": pose.row,
                "slug": pose.slug,
                "model": model,
                "size": args.size,
                "quality": args.quality,
                "reference": str(reference),
                "output": str(out_path),
                "usage": data.get("usage"),
                "created": data.get("created"),
            }
            (meta_dir / f"{pose.row:02d}-{pose.slug}.json").write_text(json.dumps(metadata, indent=2) + "\n")
            summary.append({"row": pose.row, "slug": pose.slug, "path": str(out_path), "model": model})
            row_done = True
            time.sleep(1)
            break

        if not row_done:
            die(f"could not generate {pose.slug}: {last_error}")

    (out_dir / "generation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_dir / 'generation-summary.json'}")


if __name__ == "__main__":
    main()
