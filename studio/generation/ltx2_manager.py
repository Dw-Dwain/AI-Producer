"""
studio/generation/ltx2_manager.py

LTX-2.3 audio+video generation via subprocess isolation.

LTX-2.3 cannot share the app's Python environment — it requires its own
`uv`-managed venv with torch ~2.7 / CUDA >=12.7 and the Lightricks LTX-2
package. So instead of importing it in-process (like Wan/Hunyuan), we shell
out to the LTX-2 venv's interpreter and run the pipeline as a module, exactly
like lipsync_manager spawns lip-sync engines.

LTX-2.3 produces a single MP4 with synchronized audio+video, so shots made
this way do NOT need the separate TTS + lip-sync passes.

Configuration (stored in DB settings, set via the UI Model Manager):
    ltx2_repo_dir          /workspace/LTX-2
    ltx2_venv_python       /workspace/LTX-2/.venv/bin/python
    ltx2_checkpoint_path   /workspace/models/LTX-2.3/<checkpoint>.safetensors
    ltx2_upsampler_path    /workspace/models/LTX-2.3/<spatial_upsampler>.safetensors
    ltx2_distilled_lora    /workspace/models/LTX-2.3/<distilled_lora>.safetensors
    ltx2_gemma_root        /workspace/models/LTX-2.3/gemma   (or a separate Gemma 3 dir)
    ltx2_module            ltx_pipelines.ti2vid_two_stages   (pipeline module)
    ltx2_quantization      fp8-cast   (optional; "" to disable)
"""

import json
import logging
import os
import subprocess
from datetime import datetime

logger = logging.getLogger("studio.ltx2")

# Available LTX-2 pipeline modules (the `python -m <module>` target).
PIPELINE_MODULES = {
    "two_stage": "ltx_pipelines.ti2vid_two_stages",   # text/image -> video, 2-stage upsample (best)
    "one_stage": "ltx_pipelines.ti2vid_one_stage",    # single stage (faster)
    "distilled": "ltx_pipelines.distilled",           # fastest
}

# Default DB-setting keys -> fallbacks
_DEFAULTS = {
    "ltx2_repo_dir":        "/workspace/LTX-2",
    # venv is built on local disk (UV_PROJECT_ENVIRONMENT) because MooseFS can't
    # handle a venv build; see MASTER_GUIDE §8.
    "ltx2_venv_python":     "/root/ltx2-venv/bin/python",
    "ltx2_checkpoint_path": "",
    "ltx2_upsampler_path":  "",
    "ltx2_distilled_lora":  "",
    "ltx2_gemma_root":      "",
    "ltx2_module":          "ltx_pipelines.ti2vid_two_stages",
    "ltx2_quantization":    "fp8-cast",
}


def _setting(db, key: str) -> str:
    if db is None:
        return _DEFAULTS.get(key, "")
    return db.get_setting(key, _DEFAULTS.get(key, "")) or _DEFAULTS.get(key, "")


def get_config(db) -> dict:
    """Return the current LTX-2 configuration dict (for the UI / detection)."""
    return {k: _setting(db, k) for k in _DEFAULTS}


def detect_models(db) -> dict:
    """Probe configured LTX-2 paths for the Model Manager status panel."""
    cfg = get_config(db)
    def probe(path, label, is_dir=False):
        ok = bool(path) and (os.path.isdir(path) if is_dir else os.path.exists(path))
        return {"found": ok, "path": path or "", "label": label, "files": []}
    return {
        "repo":       probe(cfg["ltx2_repo_dir"], "LTX-2 Repo", is_dir=True),
        "venv":       probe(cfg["ltx2_venv_python"], "LTX-2 venv Python"),
        "checkpoint": probe(cfg["ltx2_checkpoint_path"], "LTX-2.3 Checkpoint"),
        "upsampler":  probe(cfg["ltx2_upsampler_path"], "Spatial Upsampler"),
        "lora":       probe(cfg["ltx2_distilled_lora"], "Distilled LoRA"),
        "gemma":      probe(cfg["ltx2_gemma_root"], "Gemma 3 Encoder", is_dir=True),
    }


# In-process state interface parity with the other managers (no-ops here,
# because LTX-2 loads inside the subprocess, not in this process).
def get_loaded_pipeline_name() -> str | None:
    return None


def get_state() -> dict:
    return {}


def unload_pipeline() -> str:
    return "ℹ️ LTX-2.3 runs per-job as a subprocess — nothing to unload."


class LTX2Manager:
    def __init__(self, studio_root: str, db=None):
        self.studio_root = studio_root
        self.db = db
        self.output_root = os.path.join(studio_root, "output", "videos", "_ltx2")
        os.makedirs(self.output_root, exist_ok=True)

    # -- validation -------------------------------------------------------
    def _require(self, cfg: dict):
        repo = cfg["ltx2_repo_dir"]
        venv = cfg["ltx2_venv_python"]
        ckpt = cfg["ltx2_checkpoint_path"]
        gemma = cfg["ltx2_gemma_root"]
        if not repo or not os.path.isdir(repo):
            raise FileNotFoundError(f"LTX-2 repo dir not found: '{repo}'. Set ltx2_repo_dir in Settings.")
        if not venv or not os.path.isfile(venv):
            raise FileNotFoundError(f"LTX-2 venv python not found: '{venv}'. Run `uv sync` in the repo and set ltx2_venv_python.")
        if not ckpt or not os.path.isfile(ckpt):
            raise FileNotFoundError(f"LTX-2.3 checkpoint not found: '{ckpt}'. Set ltx2_checkpoint_path in Settings.")
        if not gemma or not os.path.isdir(gemma):
            raise FileNotFoundError(f"Gemma 3 encoder dir not found: '{gemma}'. Set ltx2_gemma_root in Settings.")

    # -- command construction --------------------------------------------
    def _build_command(self, cfg: dict, job: dict, output_path: str) -> list:
        # Module follows the job's pipeline choice (distilled/one_stage/two_stage),
        # falling back to the configured default module.
        module = PIPELINE_MODULES.get(
            job.get("pipeline", ""),
            cfg.get("ltx2_module") or "ltx_pipelines.ti2vid_two_stages",
        )
        cmd = [cfg["ltx2_venv_python"], "-m", module]

        # Flag names verified against ltx_pipelines/utils/args.py
        # (default_2_stage_arg_parser).
        cmd += ["--checkpoint-path", cfg["ltx2_checkpoint_path"]]
        cmd += ["--gemma-root", cfg["ltx2_gemma_root"]]

        # Two-stage needs the spatial upsampler + distilled LoRA for stage 2.
        if "two_stage" in module:
            if cfg.get("ltx2_upsampler_path"):
                cmd += ["--spatial-upsampler-path", cfg["ltx2_upsampler_path"]]
            if cfg.get("ltx2_distilled_lora"):
                # --distilled-lora takes the path; stage strengths are separate
                # flags with sane defaults in the parser.
                cmd += ["--distilled-lora", cfg["ltx2_distilled_lora"]]

        # Prompt / negative
        cmd += ["--prompt", job.get("prompt") or ""]
        if job.get("negative_prompt"):
            cmd += ["--negative-prompt", job["negative_prompt"]]

        # Resolution / timing / steps / seed
        if job.get("width"):      cmd += ["--width", str(job["width"])]
        if job.get("height"):     cmd += ["--height", str(job["height"])]
        if job.get("num_frames"): cmd += ["--num-frames", str(job["num_frames"])]
        if job.get("fps"):        cmd += ["--frame-rate", str(job["fps"])]
        if job.get("steps"):      cmd += ["--num-inference-steps", str(job["steps"])]
        if job.get("seed") is not None and int(job.get("seed", -1)) >= 0:
            cmd += ["--seed", str(job["seed"])]

        # Image conditioning (image-to-video). --image takes:
        #   <path> <target_frame> <strength> <noise>
        # We condition the first frame at full strength by default; tune via
        # ltx2_extra_args / the I2V conditioning string if needed.
        ref = job.get("reference_image_path")
        if ref and os.path.isfile(str(ref)):
            cmd += ["--image", ref, "0", "1.0", "0"]

        # Memory: --offload helps fit on a single 80GB card; fp8 quant reduces
        # weight footprint. Both configurable.
        if str(_setting(self.db, "ltx2_offload")).lower() in ("1", "true", "yes", "on"):
            cmd += ["--offload"]
        quant = cfg.get("ltx2_quantization")
        if quant:
            cmd += ["--quantization", quant]

        # Free-form extra args passthrough (space-separated), set via Settings
        extra = _setting(self.db, "ltx2_extra_args") if self.db else ""
        if extra:
            cmd += extra.split()

        cmd += ["--output-path", output_path]
        return cmd

    # -- main entrypoint --------------------------------------------------
    def generate(self, job: dict) -> tuple[str, dict]:
        """
        Run an LTX-2.3 generation job as a subprocess.
        Returns (output_mp4_path, metadata). Raises on failure.
        """
        cfg = get_config(self.db)
        self._require(cfg)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        seed_str = str(job.get("seed", "noseed"))
        output_path = os.path.join(self.output_root, f"ltx2_{timestamp}_seed{seed_str}.mp4")

        cmd = self._build_command(cfg, job, output_path)
        logger.info(f"LTX-2.3 command: {' '.join(cmd)} (cwd={cfg['ltx2_repo_dir']})")

        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            cwd=cfg["ltx2_repo_dir"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            check=False,
        )
        if result.stdout:
            logger.info(f"[LTX-2 STDOUT]\n{result.stdout[-4000:]}")
        if result.stderr:
            logger.warning(f"[LTX-2 STDERR]\n{result.stderr[-4000:]}")

        if result.returncode != 0:
            raise RuntimeError(
                f"LTX-2.3 generation failed (exit {result.returncode}).\n"
                f"{result.stderr[-2000:] or result.stdout[-2000:]}"
            )
        if not os.path.isfile(output_path):
            raise FileNotFoundError(
                f"LTX-2.3 reported success but no output file at {output_path}. "
                f"Check the --output-path flag name against `--help`."
            )

        metadata = {
            "engine": "LTX-2.3",
            "module": cfg.get("ltx2_module"),
            "prompt": job.get("prompt"),
            "negative_prompt": job.get("negative_prompt"),
            "seed": job.get("seed"),
            "width": job.get("width"),
            "height": job.get("height"),
            "num_frames": job.get("num_frames"),
            "fps": job.get("fps"),
            "has_audio": True,
            "output_path": output_path,
        }
        try:
            with open(output_path + ".json", "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
        except Exception as exc:
            logger.warning(f"Could not write LTX-2 metadata sidecar: {exc}")

        return output_path, metadata
