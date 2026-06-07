import gc
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


# Wan 2.2 A14B is a Mixture-of-Experts model: two ~14B experts (~56GB bf16)
# plus an ~11GB UMT5 text encoder. Keeping all of that resident overflows even
# an 80GB A100 once generation activations are added. model_cpu_offload swaps
# the experts in/out (one ~28GB expert on GPU at a time), which fits 80GB with
# room to spare. So we offload on anything below ~120GB — i.e. everything except
# H200/B300-class cards that can hold the whole MoE at once.
def _auto_offload(pipe, device: str, vram_threshold_gb: float = 120.0) -> bool:
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


def _align_frames(num_frames: int) -> int:
    # Wan 2.2 requires (num_frames - 1) % 4 == 0
    nf = max(5, num_frames)
    remainder = (nf - 1) % 4
    if remainder != 0:
        nf = nf + (4 - remainder)
    return nf


def load_pipeline(pipeline_name: str, wan_path: str, device: str = "cuda", dtype_str: str = "bfloat16"):
    global _cached_state, _cached_pipeline_name
    if _cached_pipeline_name == pipeline_name and _cached_state:
        return _cached_state, f"✅ Pipeline already loaded: **{pipeline_name}**"

    try:
        import torch
    except ImportError as exc:
        return None, f"⚠️ **torch not installed.** Error: `{exc}`"

    try:
        if pipeline_name == "image2video":
            from diffusers import WanImageToVideoPipeline
            pipe_cls = WanImageToVideoPipeline
        else:
            from diffusers import WanPipeline
            pipe_cls = WanPipeline
    except ImportError:
        # Fallback for older diffusers versions — generic loader
        logger.warning("WanPipeline not found in this diffusers version — falling back to DiffusionPipeline")
        from diffusers import DiffusionPipeline
        pipe_cls = DiffusionPipeline

    if not wan_path:
        return None, "❌ Wan Model Path is empty. Set it in the Model Manager."

    torch_dtype = _dtype(dtype_str)

    try:
        logger.info(f"Loading Wan 2.2 [{pipeline_name}] from: {wan_path}")
        pipe = pipe_cls.from_pretrained(wan_path, torch_dtype=torch_dtype, local_files_only=True)
        if not _auto_offload(pipe, device):
            pipe = pipe.to(device)

        _cached_state = {
            "name": pipeline_name,
            "base": pipe,
            "device": device,
            "dtype_str": dtype_str,
        }
        _cached_pipeline_name = pipeline_name
        return _cached_state, f"✅ **{pipeline_name}** loaded (Wan 2.2)"
    except Exception as exc:
        logger.error(f"Wan load failed: {exc}", exc_info=True)
        return None, f"❌ Failed to load **{pipeline_name}**: `{exc}`"


def unload_pipeline() -> str:
    global _cached_state, _cached_pipeline_name
    if not _cached_state:
        return "ℹ️ No Wan pipeline currently loaded."
    name = _cached_pipeline_name or "unknown"
    _cached_state = {}
    _cached_pipeline_name = None
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return f"✅ **{name}** unloaded (Wan)."


def get_loaded_pipeline_name() -> str | None:
    return _cached_pipeline_name


def get_state() -> dict:
    return _cached_state


def generate_video(
    state: dict,
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 480,
    fps: int = 16,
    num_frames: int = 81,
    steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = -1,
    pipeline_name: str = "text2video",
    reference_image=None,
):
    if not state or "base" not in state:
        return None, -1, "❌ No Wan pipeline loaded."
    try:
        import torch
        actual_seed = seed if seed != -1 else random.randint(0, 2 ** 32 - 1)
        generator = torch.Generator(device=state.get("device", "cpu")).manual_seed(actual_seed)
        pipe = state["base"]

        nf = _align_frames(num_frames)
        # Wan resolution must be divisible by 32
        w = max(256, (width // 32) * 32)
        h = max(256, (height // 32) * 32)

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
            width=w,
            height=h,
            num_frames=nf,
        )

        if pipeline_name == "image2video" and reference_image is not None:
            # Accept either a file path string or a PIL Image directly
            if isinstance(reference_image, str) and os.path.isfile(reference_image):
                from PIL import Image as PILImage
                img = PILImage.open(reference_image).convert("RGB")
                # Resize to match generation resolution so I2V doesn't distort
                img = img.resize((w, h), PILImage.LANCZOS)
                reference_image = img
            kwargs["image"] = reference_image

        result = pipe(**kwargs)
        frames = result.frames[0] if hasattr(result, "frames") else result.images
        return frames, actual_seed, None
    except Exception as exc:
        logger.error(f"Wan generation failed: {exc}", exc_info=True)
        return None, -1, f"❌ Wan generation failed: `{exc}`"
