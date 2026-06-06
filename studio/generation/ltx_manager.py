"""
studio/generation/ltx_manager.py
Phase 5: LTX-2 video model loader and multi-pipeline inference engine.

Supported pipelines:
    distilled     — fast single-stage text-to-video (or image-to-video)
    two_stage     — base generation + latent upsampling
    two_stage_hq  — two_stage with more steps in stage 2 for higher fidelity

No models are auto-downloaded. All paths are user-supplied.
diffusers / torch are imported lazily so the studio starts without them.
"""

import logging
import os
import random

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level pipeline cache
# Keys: 'distilled' | 'two_stage' | 'two_stage_hq'
# Values: dict with loaded pipeline components
# ---------------------------------------------------------------------------
_cached_state: dict = {}       # {'name': str, 'base': pipe, 'upscaler': pipe|None}
_cached_pipeline_name: str | None = None


# ---------------------------------------------------------------------------
# PRESET DEFINITIONS
# ---------------------------------------------------------------------------
PRESETS = {
    "draft": {
        "width": 768, "height": 512, "fps": 24, "num_frames": 65,
        "steps": 20, "guidance_scale": 3.0, "pipeline": "distilled",
    },
    "production": {
        "width": 1024, "height": 576, "fps": 24, "num_frames": 97,
        "steps": 40, "guidance_scale": 3.5, "pipeline": "two_stage",
    },
}


# ---------------------------------------------------------------------------
# MODEL DETECTION
# ---------------------------------------------------------------------------

def _probe_path(path: str, label: str) -> dict:
    """Return detection info for a single model path."""
    if not path or not os.path.exists(path):
        return {"found": False, "path": path or "", "label": label, "files": []}

    if os.path.isfile(path):
        found = path.endswith((".safetensors", ".bin", ".ckpt", ".pt"))
        return {"found": found, "path": path, "label": label,
                "files": [os.path.basename(path)]}

    # Directory
    try:
        entries = os.listdir(path)
    except PermissionError:
        return {"found": False, "path": path, "label": label, "files": []}

    model_exts = (".safetensors", ".bin", ".ckpt", ".pt")
    has_index   = "model_index.json" in entries
    has_weights = any(f.endswith(model_exts) for f in entries)

    # Recurse one level (transformer/, scheduler/, text_encoder/, etc.)
    collected: list[str] = [f for f in entries if any(f.endswith(e) for e in model_exts + (".json",))]
    for entry in entries:
        sub = os.path.join(path, entry)
        if os.path.isdir(sub):
            try:
                sub_files = os.listdir(sub)
                wt = [f"{entry}/{f}" for f in sub_files if f.endswith(model_exts)]
                if wt:
                    has_weights = True
                    collected.extend(wt)
            except PermissionError:
                pass

    return {
        "found": has_index or has_weights,
        "path": path,
        "label": label,
        "files": collected[:20],
    }


def detect_models(ltx_path: str, gemma_path: str,
                  upscaler_path: str, lora_dir: str) -> dict:
    """
    Validate local model paths for all four model types.

    Returns::

        {
            "ltx":      {"found": bool, "path": str, "label": str, "files": [...]},
            "gemma":    {...},
            "upscaler": {...},
            "loras":    {..., "lora_files": [...]},
        }
    """
    result = {
        "ltx":      _probe_path(ltx_path,      "LTX-Video"),
        "gemma":    _probe_path(gemma_path,    "Gemma (Prompt Enhancer)"),
        "upscaler": _probe_path(upscaler_path, "Upscaler"),
    }

    # LoRA directory: list .safetensors files
    lora_files: list[str] = []
    if lora_dir and os.path.isdir(lora_dir):
        try:
            lora_files = sorted(
                f for f in os.listdir(lora_dir) if f.endswith(".safetensors")
            )
        except PermissionError:
            pass
    result["loras"] = {
        "found": bool(lora_files),
        "path": lora_dir or "",
        "label": "LoRA Directory",
        "files": lora_files,
        "lora_files": lora_files,
    }
    return result


def list_loras(lora_dir: str) -> list[str]:
    """Return sorted list of LoRA .safetensors filenames in lora_dir."""
    if not lora_dir or not os.path.isdir(lora_dir):
        return []
    try:
        return sorted(f for f in os.listdir(lora_dir) if f.endswith(".safetensors"))
    except PermissionError:
        return []


# ---------------------------------------------------------------------------
# PIPELINE LOADING
# ---------------------------------------------------------------------------

def _dtype(dtype_str: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(dtype_str, torch.bfloat16)


def _auto_offload(pipe, device: str, vram_threshold_gb: float = 16.0) -> bool:
    """Apply cpu offload if VRAM < threshold. Returns True if offloaded."""
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if vram < vram_threshold_gb:
                logger.info(f"VRAM {vram:.1f} GB < {vram_threshold_gb} GB — enabling cpu offload")
                pipe.enable_model_cpu_offload()
                return True
    except Exception:
        pass
    return False


def load_pipeline(
    pipeline_name: str,
    ltx_path: str,
    upscaler_path: str = "",
    gemma_path: str = "",
    device: str = "cuda",
    dtype_str: str = "bfloat16",
    lora_path: str = "",
    lora_weight: float = 1.0,
):
    """
    Load the requested LTX pipeline into the module-level cache.

    Args:
        pipeline_name: 'distilled' | 'two_stage' | 'two_stage_hq'
        ltx_path:      Path to LTX-Video base model (dir or .safetensors)
        upscaler_path: Path to upscaler model (required for two_stage variants)
        gemma_path:    Optional Gemma model path for prompt enhancement
        device:        'cuda' | 'cpu'
        dtype_str:     'bfloat16' | 'float16' | 'float32'
        lora_path:     Optional LoRA .safetensors file path
        lora_weight:   LoRA scale / weight

    Returns:
        (state_dict | None, status_message)
    """
    global _cached_state, _cached_pipeline_name

    # Return cached if already loaded with same name
    if _cached_pipeline_name == pipeline_name and _cached_state:
        logger.info(f"Using cached pipeline: {pipeline_name}")
        return _cached_state, f"✅ Pipeline already loaded: **{pipeline_name}**"

    # Lazy import
    try:
        import torch
        from diffusers import LTXPipeline, LTXImageToVideoPipeline
    except ImportError as exc:
        msg = (
            "⚠️ **diffusers / torch not installed.**\n\n"
            "Install with:\n```\npip install torch diffusers transformers accelerate imageio imageio-ffmpeg\n```\n\n"
            f"Error: `{exc}`"
        )
        logger.warning(msg)
        return None, msg

    if not ltx_path:
        return None, "❌ LTX Model Path is empty. Set it in the Model Manager."

    torch_dtype = _dtype(dtype_str)

    try:
        # ---- Stage 1: Load base LTX pipeline ----
        logger.info(f"Loading LTX base pipeline from: {ltx_path}")
        base_pipe = LTXPipeline.from_pretrained(
            ltx_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )

        # Apply LoRA if supplied
        if lora_path and os.path.isfile(lora_path):
            logger.info(f"Loading LoRA weights: {lora_path} @ {lora_weight}")
            try:
                base_pipe.load_lora_weights(lora_path)
                base_pipe.fuse_lora(lora_scale=float(lora_weight))
            except Exception as lora_exc:
                logger.warning(f"LoRA load failed (continuing without): {lora_exc}")

        # Auto-offload or move to device
        if not _auto_offload(base_pipe, device):
            base_pipe = base_pipe.to(device)

        # ---- Stage 2: Load upscaler pipeline (two_stage variants) ----
        upscaler_pipe = None
        if pipeline_name in ("two_stage", "two_stage_hq"):
            if not upscaler_path:
                logger.warning("Two-stage pipeline selected but no upscaler path — will skip stage 2")
            else:
                logger.info(f"Loading LTX upscaler from: {upscaler_path}")
                try:
                    # Try dedicated upscale pipeline first, fall back to base
                    try:
                        from diffusers import LTXLatentUpsamplePipeline
                        upscaler_pipe = LTXLatentUpsamplePipeline.from_pretrained(
                            upscaler_path,
                            torch_dtype=torch_dtype,
                            local_files_only=True,
                        )
                    except (ImportError, Exception):
                        logger.warning("LTXLatentUpsamplePipeline unavailable — loading as LTXPipeline upscaler")
                        upscaler_pipe = LTXPipeline.from_pretrained(
                            upscaler_path,
                            torch_dtype=torch_dtype,
                            local_files_only=True,
                        )
                    if not _auto_offload(upscaler_pipe, device):
                        upscaler_pipe = upscaler_pipe.to(device)
                except Exception as up_exc:
                    logger.error(f"Upscaler load failed: {up_exc}")
                    upscaler_pipe = None

        # Cache state
        _cached_state = {
            "name": pipeline_name,
            "base": base_pipe,
            "upscaler": upscaler_pipe,
            "device": device,
            "dtype_str": dtype_str,
        }
        _cached_pipeline_name = pipeline_name

        # VRAM report
        vram_info = ""
        try:
            if device == "cuda" and torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
                total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                vram_info = f" | VRAM: {alloc:.1f}/{total:.1f} GB"
        except Exception:
            pass

        stage2_note = " (+ upscaler)" if upscaler_pipe else ""
        msg = f"✅ **{pipeline_name}**{stage2_note} loaded{vram_info}"
        logger.info(msg)
        return _cached_state, msg

    except Exception as exc:
        logger.error(f"Pipeline load failed: {exc}", exc_info=True)
        return None, f"❌ Failed to load **{pipeline_name}**: `{exc}`"


def unload_pipeline() -> str:
    """Release all cached pipelines and free GPU memory."""
    global _cached_state, _cached_pipeline_name
    if not _cached_state:
        return "ℹ️ No pipeline currently loaded."
    name = _cached_pipeline_name or "unknown"
    _cached_state = {}
    _cached_pipeline_name = None
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info(f"Pipeline unloaded: {name}")
    return f"✅ **{name}** unloaded. GPU memory released."


def get_loaded_pipeline_name() -> str | None:
    return _cached_pipeline_name


def get_vram_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return f"{alloc:.1f} / {total:.1f} GB"
    except Exception:
        pass
    return "N/A"


# ---------------------------------------------------------------------------
# VIDEO GENERATION
# ---------------------------------------------------------------------------

def generate_video(
    state: dict,
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    fps: int = 24,
    num_frames: int = 65,
    steps: int = 20,
    guidance_scale: float = 3.0,
    seed: int = -1,
    pipeline_name: str = "distilled",
    reference_image=None,   # PIL.Image or None
):
    """
    Run LTX video generation.

    Two-stage pipelines:
      - Stage 1: generate at base resolution (width/2, height/2 for two_stage_hq)
      - Stage 2: upscale latents to target resolution

    Returns:
        (frames: list[PIL.Image], actual_seed: int, error_str | None)
    """
    if not state or "base" not in state:
        return None, -1, "❌ No pipeline loaded. Load a pipeline via the Model Manager."

    try:
        import torch

        actual_seed = seed if seed != -1 else random.randint(0, 2 ** 32 - 1)
        generator = torch.Generator().manual_seed(actual_seed)

        base_pipe     = state["base"]
        upscaler_pipe = state.get("upscaler")

        # Ensure dimensions are multiples of 32 (LTX requirement)
        w = max(256, (width  // 32) * 32)
        h = max(256, (height // 32) * 32)

        # Ensure num_frames satisfies (n-1) % 8 == 0
        nf = num_frames
        if (nf - 1) % 8 != 0:
            nf = ((nf - 1) // 8) * 8 + 1
            if nf < 1:
                nf = 1

        logger.info(
            f"Generating video | pipeline={pipeline_name} | seed={actual_seed} "
            f"| {w}x{h} | fps={fps} | frames={nf} | steps={steps}"
        )

        common_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

        if pipeline_name == "distilled":
            # Single stage — text-to-video or image-to-video
            if reference_image is not None:
                try:
                    from diffusers import LTXImageToVideoPipeline
                    i2v_pipe = LTXImageToVideoPipeline(**{
                        k: getattr(base_pipe, k) for k in
                        ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
                        if hasattr(base_pipe, k)
                    })
                    result = i2v_pipe(
                        image=reference_image,
                        width=w, height=h,
                        num_frames=nf,
                        **common_kwargs,
                    )
                except Exception as i2v_err:
                    logger.warning(f"I2V failed ({i2v_err}), falling back to T2V")
                    result = base_pipe(width=w, height=h, num_frames=nf, **common_kwargs)
            else:
                result = base_pipe(width=w, height=h, num_frames=nf, **common_kwargs)

            frames = result.frames[0] if hasattr(result, "frames") else result.images

        elif pipeline_name in ("two_stage", "two_stage_hq"):
            # Stage 1: generate at reduced resolution
            stage1_w = w // 2 if pipeline_name == "two_stage_hq" else w
            stage1_h = h // 2 if pipeline_name == "two_stage_hq" else h
            stage1_steps = max(10, steps // 2)
            stage2_steps = steps - stage1_steps

            stage1_w = max(256, (stage1_w // 32) * 32)
            stage1_h = max(256, (stage1_h // 32) * 32)

            logger.info(f"Stage 1: {stage1_w}x{stage1_h} | {stage1_steps} steps")
            result1 = base_pipe(
                width=stage1_w, height=stage1_h,
                num_frames=nf,
                num_inference_steps=stage1_steps,
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(actual_seed),
                output_type="latent" if upscaler_pipe else "pil",
            )

            if upscaler_pipe and hasattr(result1, "frames") and result1.frames is not None:
                # Stage 2: upscale latents → final resolution
                logger.info(f"Stage 2 upscale: {w}x{h} | {stage2_steps} steps")
                try:
                    result2 = upscaler_pipe(
                        latents=result1.frames,
                        prompt=prompt,
                        width=w, height=h,
                        num_frames=nf,
                        num_inference_steps=stage2_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator().manual_seed(actual_seed),
                        output_type="pil",
                    )
                    frames = result2.frames[0] if hasattr(result2, "frames") else result2.images
                except Exception as s2_err:
                    logger.warning(f"Stage 2 failed ({s2_err}), using stage 1 output")
                    # Fall back: decode stage 1 latents or use pil directly
                    frames = _decode_fallback(base_pipe, result1, stage1_w, stage1_h)
            else:
                frames = result1.frames[0] if hasattr(result1, "frames") else result1.images
        else:
            return None, -1, f"❌ Unknown pipeline: {pipeline_name}"

        logger.info(f"Generation complete. Frames: {len(frames)}")
        return frames, actual_seed, None

    except Exception as exc:
        logger.error(f"Video generation failed: {exc}", exc_info=True)
        return None, -1, f"❌ Generation failed: `{exc}`"


def _decode_fallback(pipe, result, w, h):
    """Try to extract PIL frames from a latent or pil result."""
    try:
        if hasattr(result, "frames") and result.frames is not None:
            if hasattr(result.frames, "shape"):  # tensor
                decoded = pipe.decode_latents(result.frames)
                # decoded shape: (batch, frames, h, w, c) or similar
                import numpy as np
                from PIL import Image
                arr = decoded[0]  # first batch
                frames = []
                for f in arr:
                    if f.max() <= 1.0:
                        f = (f * 255).clip(0, 255).astype("uint8")
                    frames.append(Image.fromarray(f))
                return frames
            else:
                return result.frames[0]
        if hasattr(result, "images"):
            return result.images
    except Exception:
        pass
    return []
