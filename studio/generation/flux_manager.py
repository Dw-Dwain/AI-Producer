"""
studio/generation/flux_manager.py
Phase 4: Flux model loader and inference engine.

No models are downloaded here. Users supply their own local paths.
Diffusers / torch are imported lazily so the studio launches even
when those libraries are not installed.
"""

import logging
import random

logger = logging.getLogger(__name__)

# Module-level pipeline cache: only one pipeline is resident at a time.
_cached_pipeline = None
_cached_model_name = None  # 'flux_dev' | 'flux_kontext'


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def detect_models(dev_path: str, kontext_path: str) -> dict:
    """
    Validate that each local path actually contains Flux model artefacts.

    Returns::

        {
            "flux_dev":     {"found": bool, "path": str, "files": [...]},
            "flux_kontext": {"found": bool, "path": str, "files": [...]},
        }
    """
    import os

    result = {}
    for name, path in [("flux_dev", dev_path), ("flux_kontext", kontext_path)]:
        if not path or not os.path.exists(path):
            result[name] = {"found": False, "path": path or "", "files": []}
            continue

        # Accept either a directory (diffusers format) or a bare safetensors file.
        if os.path.isfile(path):
            result[name] = {
                "found": path.endswith((".safetensors", ".bin", ".ckpt")),
                "path": path,
                "files": [os.path.basename(path)],
            }
        else:
            # Directory — look for diffusers model_index.json or safetensors files
            files = os.listdir(path)
            known_exts = (".safetensors", ".bin", ".ckpt", ".json", ".pt")
            model_files = [f for f in files if any(f.endswith(e) for e in known_exts)]
            has_index = "model_index.json" in files
            has_weights = any(f.endswith((".safetensors", ".bin", ".ckpt")) for f in files)

            # Also check sub-directories (transformer/, scheduler/, etc.)
            for entry in files:
                sub = os.path.join(path, entry)
                if os.path.isdir(sub):
                    sub_files = os.listdir(sub)
                    if any(f.endswith((".safetensors", ".bin")) for f in sub_files):
                        has_weights = True
                        model_files += [f"{entry}/{f}" for f in sub_files if f.endswith((".safetensors", ".bin"))]

            result[name] = {
                "found": has_index or has_weights,
                "path": path,
                "files": model_files[:20],  # cap list length for display
            }

    return result


def load_model(
    model_name: str,
    dev_path: str,
    kontext_path: str,
    device: str = "cuda",
    dtype_str: str = "bfloat16",
):
    """
    Load (or return cached) a Flux pipeline into memory.

    Args:
        model_name:  'flux_dev' or 'flux_kontext'
        dev_path:    Local path to Flux Dev model
        kontext_path: Local path to Flux Kontext model
        device:      'cuda' | 'cpu' | 'mps'
        dtype_str:   'bfloat16' | 'float16' | 'float32'

    Returns:
        (pipeline_object, status_message)
        On failure: (None, error_string)
    """
    global _cached_pipeline, _cached_model_name

    # Return cached pipeline if it matches
    if _cached_pipeline is not None and _cached_model_name == model_name:
        logger.info(f"Using cached pipeline: {model_name}")
        return _cached_pipeline, f"✅ Model already loaded: **{model_name}**"

    # Lazy import — graceful failure when diffusers/torch not installed
    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError as exc:
        msg = (
            "⚠️ **diffusers / torch not installed.**\n\n"
            "Install them in your environment:\n"
            "```\npip install torch diffusers transformers accelerate\n```\n\n"
            f"Original error: `{exc}`"
        )
        logger.warning(msg)
        return None, msg

    # Resolve dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Resolve path
    path = dev_path if model_name == "flux_dev" else kontext_path
    if not path:
        return None, f"❌ No path configured for **{model_name}**. Set it in Model Manager."

    # Pick pipeline class
    try:
        if model_name == "flux_kontext":
            # Flux Kontext support (requires diffusers >= 0.33 or a custom fork)
            try:
                from diffusers import FluxKontextPipeline
                PipelineCls = FluxKontextPipeline
            except ImportError:
                # Fall back to standard FluxPipeline if Kontext not available
                logger.warning("FluxKontextPipeline not found; falling back to FluxPipeline")
                PipelineCls = FluxPipeline
        else:
            PipelineCls = FluxPipeline

        logger.info(f"Loading {model_name} from {path} | dtype={dtype_str} | device={device}")
        pipe = PipelineCls.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            local_files_only=True,  # never auto-download
        )

        # VRAM check: auto-offload when GPU VRAM < 24 GB
        offloaded = False
        if device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb < 24:
                logger.info(f"VRAM = {vram_gb:.1f} GB < 24 GB — enabling cpu_offload")
                pipe.enable_model_cpu_offload()
                offloaded = True

        if not offloaded:
            pipe = pipe.to(device)

        _cached_pipeline = pipe
        _cached_model_name = model_name

        vram_info = ""
        if device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            vram_info = f" | VRAM used: {allocated:.2f} GB"

        msg = f"✅ **{model_name}** loaded successfully{vram_info}"
        logger.info(msg)
        return pipe, msg

    except Exception as exc:
        logger.error(f"Failed to load {model_name}: {exc}", exc_info=True)
        return None, f"❌ Failed to load **{model_name}**: `{exc}`"


def unload_model():
    """
    Release the cached pipeline and free GPU memory.
    Returns a status message string.
    """
    global _cached_pipeline, _cached_model_name

    if _cached_pipeline is None:
        return "ℹ️ No model currently loaded."

    name = _cached_model_name
    _cached_pipeline = None
    _cached_model_name = None

    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info(f"Model unloaded: {name}")
    return f"✅ **{name}** unloaded. GPU memory released."


def get_loaded_model_name() -> str | None:
    """Return the currently-loaded model name, or None."""
    return _cached_model_name


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    seed: int = -1,
    guidance_scale: float = 3.5,
):
    """
    Run inference using an already-loaded pipeline.

    Args:
        pipe:            The diffusers FluxPipeline (or Kontext variant)
        prompt:          Positive prompt text
        negative_prompt: Negative prompt text
        width / height:  Output image size (must be multiples of 64)
        steps:           Number of denoising steps
        seed:            Fixed seed for reproducibility; -1 = random
        guidance_scale:  CFG-like scale (Flux uses 0 by default for Dev)

    Returns:
        (PIL.Image | None, actual_seed: int, error_str | None)
    """
    if pipe is None:
        return None, -1, "❌ No model loaded. Load a model first via the Model Manager."

    try:
        import torch

        actual_seed = seed if seed != -1 else random.randint(0, 2 ** 32 - 1)
        generator = torch.Generator().manual_seed(actual_seed)

        # Build kwargs — Flux Dev ignores negative_prompt but we include it
        # for compatibility with downstream pipelines that do support it.
        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        logger.info(
            f"Generating image | seed={actual_seed} | {width}x{height} | steps={steps}"
        )
        output = pipe(**kwargs)
        pil_image = output.images[0]
        logger.info("Generation complete.")
        return pil_image, actual_seed, None

    except Exception as exc:
        logger.error(f"Generation failed: {exc}", exc_info=True)
        return None, -1, f"❌ Generation failed: `{exc}`"
