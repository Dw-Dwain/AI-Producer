"""
studio/generation/output_manager.py
Phase 4: Image file saving, thumbnail creation, and metadata sidecar writing.
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_output_dir(studio_root: str, project_name: str = "", character_name: str = "") -> str:
    """
    Return (and create) the output directory for a project+character combination.

    Layout::

        <studio_root>/output/<project>/<character>/

    If project_name or character_name is empty, falls back to '_unassigned'.
    """
    safe_project   = _safe_dirname(project_name)   or "_unassigned"
    safe_character = _safe_dirname(character_name) or "_unassigned"

    out_dir = os.path.join(studio_root, "output", safe_project, safe_character)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _safe_dirname(name: str) -> str:
    """Strip characters that are unsafe for directory names."""
    if not name:
        return ""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.")
    return "".join(c if c in keep else "_" for c in name).strip()


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_metadata(
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = -1,
    model: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    guidance_scale: float = 3.5,
    character_name: str = "",
    location_name: str = "",
    project_name: str = "",
) -> dict:
    """Return a flat dict describing the generation parameters."""
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "model": model,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "character": character_name,
        "location": location_name,
        "project": project_name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_image(pil_image, output_dir: str, metadata: dict) -> tuple[str, str | None]:
    """
    Save a PIL image as PNG and write a companion JSON sidecar.

    Returns:
        (file_path, thumbnail_path)
        thumbnail_path is None if thumbnail creation failed.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    seed_str   = str(metadata.get("seed", "noseed"))
    base_name  = f"flux_{timestamp}_seed{seed_str}"

    png_path  = os.path.join(output_dir, base_name + ".png")
    json_path = os.path.join(output_dir, base_name + ".json")
    thumb_path: str | None = None

    # Save full-resolution PNG
    pil_image.save(png_path, format="PNG")
    logger.info(f"Image saved: {png_path}")

    # Write sidecar JSON
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning(f"Could not write metadata sidecar: {exc}")

    # Generate thumbnail (256 px wide, proportional)
    try:
        thumb = pil_image.copy()
        thumb.thumbnail((256, 256))
        thumb_path = os.path.join(output_dir, base_name + "_thumb.png")
        thumb.save(thumb_path, format="PNG")
    except Exception as exc:
        logger.warning(f"Thumbnail creation failed: {exc}")
        thumb_path = None

    return png_path, thumb_path
