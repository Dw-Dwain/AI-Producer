"""
studio/generation/video_output_manager.py
Phase 5: LTX-2 video file saving — MP4 export, metadata JSON, prompt .txt sidecar,
and first-frame thumbnail PNG.
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _safe_dirname(name: str) -> str:
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.")
    return "".join(c if c in keep else "_" for c in (name or "")).strip()


def get_video_output_dir(
    studio_root: str,
    project_name: str = "",
    character_name: str = "",
) -> str:
    """
    Return (and create) the output directory for a project+character combination.

    Layout::

        <studio_root>/output/videos/<project>/<character>/
    """
    safe_proj = _safe_dirname(project_name)  or "_unassigned"
    safe_char = _safe_dirname(character_name) or "_unassigned"
    out_dir = os.path.join(studio_root, "output", "videos", safe_proj, safe_char)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def build_video_metadata(
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = -1,
    pipeline: str = "",
    preset: str = "",
    width: int = 768,
    height: int = 512,
    fps: int = 24,
    num_frames: int = 65,
    steps: int = 20,
    guidance_scale: float = 3.0,
    lora_path: str = "",
    lora_weight: float = 1.0,
    reference_image_path: str = "",
    character_name: str = "",
    location_name: str = "",
    project_name: str = "",
) -> dict:
    """Return a flat metadata dict covering all generation parameters."""
    return {
        "prompt":               prompt,
        "negative_prompt":      negative_prompt,
        "seed":                 seed,
        "pipeline":             pipeline,
        "preset":               preset,
        "width":                width,
        "height":               height,
        "fps":                  fps,
        "num_frames":           num_frames,
        "steps":                steps,
        "guidance_scale":       guidance_scale,
        "lora_path":            lora_path,
        "lora_weight":          lora_weight,
        "reference_image_path": reference_image_path,
        "character":            character_name,
        "location":             location_name,
        "project":              project_name,
        "generated_at":         datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_video(
    frames: list,
    fps: int,
    output_dir: str,
    metadata: dict,
) -> tuple:
    """
    Save a list of PIL frames as an MP4 and write sidecar files.

    Tries ``imageio`` (with ffmpeg) first; falls back to writing
    individual PNG frames if imageio-ffmpeg is unavailable.

    Returns:
        (mp4_path, thumbnail_path, duration_seconds)
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    seed_str  = str(metadata.get("seed", "noseed"))
    base_name = f"ltx_{timestamp}_seed{seed_str}"

    mp4_path   = os.path.join(output_dir, base_name + ".mp4")
    json_path  = os.path.join(output_dir, base_name + ".json")
    txt_path   = os.path.join(output_dir, base_name + "_prompt.txt")
    thumb_path: str | None = None
    duration   = len(frames) / max(fps, 1)

    # ---- Save MP4 ----
    mp4_saved = False
    if frames:
        mp4_saved = _save_mp4_imageio(frames, fps, mp4_path)
        if not mp4_saved:
            mp4_path = _save_frames_fallback(frames, output_dir, base_name)
            if mp4_path:
                mp4_saved = True

    if not mp4_saved:
        logger.error("Could not save video — no frames or encoder unavailable.")
        mp4_path = os.path.join(output_dir, base_name + "_FAILED.txt")
        with open(mp4_path, "w") as f:
            f.write("Video generation failed to save.\n")

    # ---- Write JSON metadata sidecar ----
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning(f"JSON sidecar write failed: {exc}")

    # ---- Write prompt .txt sidecar ----
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(metadata.get("prompt", "") + "\n")
            neg = metadata.get("negative_prompt", "")
            if neg:
                f.write(f"\n[NEGATIVE]\n{neg}\n")
    except Exception as exc:
        logger.warning(f"Prompt sidecar write failed: {exc}")

    # ---- First-frame thumbnail ----
    if frames:
        try:
            thumb = frames[0].copy()
            thumb.thumbnail((320, 320))
            thumb_path = os.path.join(output_dir, base_name + "_thumb.png")
            thumb.save(thumb_path, format="PNG")
        except Exception as exc:
            logger.warning(f"Thumbnail generation failed: {exc}")
            thumb_path = None

    logger.info(f"Video saved: {mp4_path} ({duration:.2f}s)")
    return mp4_path, thumb_path, duration


def _save_mp4_imageio(frames: list, fps: int, mp4_path: str) -> bool:
    """Attempt to write frames as MP4 using imageio + imageio-ffmpeg."""
    try:
        import imageio
        import numpy as np

        writer = imageio.get_writer(
            mp4_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        for frame in frames:
            arr = np.array(frame.convert("RGB"))
            writer.append_data(arr)
        writer.close()
        logger.info(f"MP4 written via imageio: {mp4_path}")
        return True
    except Exception as exc:
        logger.warning(f"imageio MP4 write failed: {exc}")
        return False


def _save_frames_fallback(frames: list, output_dir: str, base_name: str) -> str | None:
    """Fallback: save individual frames as PNGs in a subfolder."""
    try:
        frames_dir = os.path.join(output_dir, base_name + "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(os.path.join(frames_dir, f"frame_{i:05d}.png"))
        logger.info(f"Frames saved to folder (no MP4 encoder): {frames_dir}")
        # Return path to frames dir as the "file_path" stored in DB
        return frames_dir
    except Exception as exc:
        logger.error(f"Frame fallback failed: {exc}")
        return None
