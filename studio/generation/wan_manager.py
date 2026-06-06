import logging
import os
import random

logger = logging.getLogger(__name__)

_cached_state: dict = {}
_cached_pipeline_name: str | None = None

def _probe_path(path: str, label: str) -> dict:
    if not path or not os.path.exists(path):
        return {"found": False, "path": path or "", "label": label, "files": []}
    if os.path.isfile(path):
        found = path.endswith((".safetensors", ".bin", ".ckpt", ".pt"))
        return {"found": found, "path": path, "label": label, "files": [os.path.basename(path)]}
    try:
        entries = os.listdir(path)
    except PermissionError:
        return {"found": False, "path": path, "label": label, "files": []}
    model_exts = (".safetensors", ".bin", ".ckpt", ".pt")
    has_index = "model_index.json" in entries
    has_weights = any(f.endswith(model_exts) for f in entries)
    return {"found": has_index or has_weights, "path": path, "label": label, "files": entries[:20]}

def detect_models(wan_path: str) -> dict:
    return {"wan": _probe_path(wan_path, "Wan 2.2 Base Model")}

def _dtype(dtype_str: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_str, torch.bfloat16)

def _auto_offload(pipe, device: str, vram_threshold_gb: float = 16.0) -> bool:
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram < vram_threshold_gb:
                logger.info(f"VRAM {vram:.1f} GB < {vram_threshold_gb} GB — enabling cpu offload")
                pipe.enable_model_cpu_offload()
                return True
    except Exception:
        pass
    return False

def load_pipeline(pipeline_name: str, wan_path: str, device: str = "cuda", dtype_str: str = "bfloat16"):
    global _cached_state, _cached_pipeline_name
    if _cached_pipeline_name == pipeline_name and _cached_state:
        return _cached_state, f"✅ Pipeline already loaded: **{pipeline_name}**"

    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        msg = f"⚠️ **diffusers / torch not installed.**\nError: `{exc}`"
        return None, msg

    if not wan_path:
        return None, "❌ Wan Model Path is empty. Set it in the Model Manager."

    torch_dtype = _dtype(dtype_str)

    try:
        logger.info(f"Loading Wan 2.2 pipeline from: {wan_path}")
        # Use DiffusionPipeline to handle either T2V or I2V variants seamlessly
        base_pipe = DiffusionPipeline.from_pretrained(wan_path, torch_dtype=torch_dtype, local_files_only=True)
        if not _auto_offload(base_pipe, device):
            base_pipe = base_pipe.to(device)

        _cached_state = {"name": pipeline_name, "base": base_pipe, "device": device, "dtype_str": dtype_str}
        _cached_pipeline_name = pipeline_name
        return _cached_state, f"✅ **{pipeline_name}** loaded (Wan 2.2)"
    except Exception as exc:
        logger.error(f"Wan load failed: {exc}")
        return None, f"❌ Failed to load **{pipeline_name}**: `{exc}`"

def unload_pipeline() -> str:
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
    return f"✅ **{name}** unloaded (Wan)."

def get_loaded_pipeline_name() -> str | None:
    return _cached_pipeline_name

def generate_video(state: dict, prompt: str, negative_prompt: str = "", width: int = 768, height: int = 512, fps: int = 24, num_frames: int = 65, steps: int = 20, guidance_scale: float = 3.0, seed: int = -1, pipeline_name: str = "text2video", reference_image=None):
    if not state or "base" not in state:
        return None, -1, "❌ No Wan pipeline loaded."
    try:
        import torch
        actual_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
        generator = torch.Generator().manual_seed(actual_seed)
        pipe = state["base"]

        # Ensure frame count and resolutions meet typical boundaries
        nf = num_frames
        if (nf - 1) % 8 != 0:
            nf = ((nf - 1) // 8) * 8 + 1
            if nf < 1: nf = 1
        
        w = max(256, (width // 32) * 32)
        h = max(256, (height // 32) * 32)

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
            width=w, height=h, num_frames=nf
        )

        if pipeline_name == "image2video" and reference_image is not None:
            kwargs["image"] = reference_image
            
        result = pipe(**kwargs)
        frames = result.frames[0] if hasattr(result, "frames") else result.images
        return frames, actual_seed, None
    except Exception as exc:
        return None, -1, f"❌ Wan generation failed: `{exc}`"
