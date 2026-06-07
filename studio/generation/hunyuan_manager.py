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


def detect_models(hunyuan_path: str) -> dict:
    return {"hunyuan": _probe_path(hunyuan_path, "Hunyuan Video Base Model")}


def _dtype(dtype_str: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_str, torch.bfloat16)


def _auto_offload(pipe, device: str, vram_threshold_gb: float = 40.0) -> bool:
    # Hunyuan is large (~13B params) — offload unless we have serious VRAM
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if vram < vram_threshold_gb:
                logger.info(f"VRAM {vram:.1f} GB < {vram_threshold_gb} GB — enabling sequential cpu offload for Hunyuan")
                pipe.enable_sequential_cpu_offload()
                return True
    except Exception:
        pass
    return False


def _align_frames(num_frames: int) -> int:
    # Hunyuan Video requires (num_frames - 1) % 4 == 0
    nf = max(5, num_frames)
    remainder = (nf - 1) % 4
    if remainder != 0:
        nf = nf + (4 - remainder)
    return nf


def load_pipeline(pipeline_name: str, hunyuan_path: str, device: str = "cuda", dtype_str: str = "bfloat16"):
    global _cached_state, _cached_pipeline_name
    if _cached_pipeline_name == pipeline_name and _cached_state:
        return _cached_state, f"✅ Pipeline already loaded: **{pipeline_name}**"

    try:
        import torch
    except ImportError as exc:
        return None, f"⚠️ **torch not installed.** Error: `{exc}`"

    try:
        from diffusers import HunyuanVideoPipeline
        from diffusers.models import AutoencoderKLHunyuanVideo
        pipe_cls = HunyuanVideoPipeline
    except ImportError:
        logger.warning("HunyuanVideoPipeline not found in this diffusers version — falling back to DiffusionPipeline")
        from diffusers import DiffusionPipeline
        pipe_cls = DiffusionPipeline
        AutoencoderKLHunyuanVideo = None

    if not hunyuan_path:
        return None, "❌ Hunyuan Model Path is empty. Set it in the Model Manager."

    # Hunyuan requires bfloat16 — float32 will OOM on any practical GPU
    if dtype_str == "float32":
        logger.warning("Hunyuan Video: float32 requested but overriding to bfloat16 to prevent OOM")
        dtype_str = "bfloat16"
    torch_dtype = _dtype(dtype_str)

    try:
        logger.info(f"Loading Hunyuan Video [{pipeline_name}] from: {hunyuan_path}")
        pipe = pipe_cls.from_pretrained(
            hunyuan_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        if not _auto_offload(pipe, device):
            pipe = pipe.to(device)

        # Enable VAE slicing to reduce VRAM peak during decode
        try:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        except AttributeError:
            pass

        _cached_state = {
            "name": pipeline_name,
            "base": pipe,
            "device": device,
            "dtype_str": dtype_str,
        }
        _cached_pipeline_name = pipeline_name
        return _cached_state, f"✅ **{pipeline_name}** loaded (Hunyuan Video)"
    except Exception as exc:
        logger.error(f"Hunyuan load failed: {exc}", exc_info=True)
        return None, f"❌ Failed to load **{pipeline_name}**: `{exc}`"


def unload_pipeline() -> str:
    global _cached_state, _cached_pipeline_name
    if not _cached_state:
        return "ℹ️ No Hunyuan pipeline currently loaded."
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
    return f"✅ **{name}** unloaded (Hunyuan)."


def get_loaded_pipeline_name() -> str | None:
    return _cached_pipeline_name


def get_state() -> dict:
    return _cached_state


def generate_video(
    state: dict,
    prompt: str,
    negative_prompt: str = "",
    width: int = 720,
    height: int = 1280,
    fps: int = 24,
    num_frames: int = 129,
    steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = -1,
    pipeline_name: str = "text2video",
    reference_image=None,
):
    if not state or "base" not in state:
        return None, -1, "❌ No Hunyuan pipeline loaded."
    try:
        import torch
        actual_seed = seed if seed != -1 else random.randint(0, 2 ** 32 - 1)
        generator = torch.Generator(device=state.get("device", "cpu")).manual_seed(actual_seed)
        pipe = state["base"]

        nf = _align_frames(num_frames)
        # Hunyuan resolution must be divisible by 16
        w = max(256, (width // 16) * 16)
        h = max(256, (height // 16) * 16)

        kwargs = dict(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
            width=w,
            height=h,
            num_frames=nf,
        )

        # Hunyuan base model does not use negative_prompt in the same way;
        # only pass it if the loaded pipeline supports it
        if negative_prompt:
            try:
                import inspect
                if "negative_prompt" in inspect.signature(pipe.__call__).parameters:
                    kwargs["negative_prompt"] = negative_prompt
            except Exception:
                pass

        # I2V is not yet in the standard diffusers HunyuanVideoPipeline;
        # silently ignore reference_image to avoid breaking jobs
        if pipeline_name == "image2video" and reference_image is not None:
            logger.warning("Hunyuan image2video not supported in current diffusers — running text2video instead")

        result = pipe(**kwargs)
        frames = result.frames[0] if hasattr(result, "frames") else result.images
        return frames, actual_seed, None
    except Exception as exc:
        logger.error(f"Hunyuan generation failed: {exc}", exc_info=True)
        return None, -1, f"❌ Hunyuan generation failed: `{exc}`"
