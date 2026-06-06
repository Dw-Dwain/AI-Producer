"""
Engine-switchable lip sync execution layer.

Spawns child subprocesses running Wav2Lip, MuseTalk, or SyncTalk
using their respective configured Python executables and codebases.
"""
import os
import shutil
import subprocess
import logging
import json
from datetime import datetime

logger = logging.getLogger("studio.lipsync")

class LipSyncManager:
    SUPPORTED_ENGINES = ("Wav2Lip", "SyncTalk", "MuseTalk")

    def __init__(self, studio_root: str, db=None):
        self.studio_root = studio_root
        self.output_root = os.path.join(studio_root, "output", "_lipsync")
        os.makedirs(self.output_root, exist_ok=True)
        
        self.db = db
        if not self.db:
            try:
                from studio.database.db_manager import DatabaseManager
                self.db = DatabaseManager()
            except ImportError:
                self.db = None

    def process(self, engine: str, input_video_path: str, input_audio_path: str, engine_config: dict | None = None) -> tuple[str | None, dict]:
        engine = engine or "Wav2Lip"
        if engine not in self.SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported lip sync engine: {engine}")
        if not input_video_path or not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Source video for lip sync was not found at: {input_video_path}")
        if not input_audio_path or not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Source audio for lip sync was not found at: {input_audio_path}")

        # Fetch model config from settings
        python_path = "python"
        code_dir = ""
        checkpoint_path = ""

        if self.db:
            python_path = self.db.get_setting(f"lipsync_{engine.lower()}_python_path", "python")
            code_dir = self.db.get_setting(f"lipsync_{engine.lower()}_code_dir", "")
            checkpoint_path = self.db.get_setting(f"lipsync_{engine.lower()}_checkpoint_path", "")

        # Fallback to defaults if not set
        if not code_dir:
            code_dir = os.path.join(self.studio_root, "models", engine) if os.name == "nt" else f"/workspace/{engine}"
        if not checkpoint_path:
            if engine == "Wav2Lip":
                checkpoint_path = os.path.join(code_dir, "checkpoints", "wav2lip_gan.pth")
            elif engine == "MuseTalk":
                checkpoint_path = os.path.join(code_dir, "models", "musetalk", "musetalk.json")
            elif engine == "SyncTalk":
                checkpoint_path = os.path.join(code_dir, "checkpoints", "synctalk.pth")

        # Verify directories exist before running
        if not os.path.isdir(code_dir):
            raise FileNotFoundError(
                f"Code directory for {engine} not found at '{code_dir}'. "
                f"Please ensure it is installed and configured correctly in Settings."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint for {engine} not found at '{checkpoint_path}'. "
                f"Please verify model files and configuration paths."
            )

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(input_video_path)[1] or ".mp4"
        output_path = os.path.join(self.output_root, f"{engine.lower()}_{timestamp}{ext}")

        # Build command based on engine type
        cmd = []
        temp_out_dir = None

        if engine == "Wav2Lip":
            cmd = [
                python_path,
                "inference.py",
                "--checkpoint", checkpoint_path,
                "--face", input_video_path,
                "--audio", input_audio_path,
                "--outfile", output_path
            ]
        elif engine == "MuseTalk":
            temp_out_dir = os.path.join(self.output_root, f"musetalk_temp_{timestamp}")
            os.makedirs(temp_out_dir, exist_ok=True)
            cmd = [
                python_path,
                "inference.py",
                "--video_path", input_video_path,
                "--audio_path", input_audio_path,
                "--result_dir", temp_out_dir
            ]
        elif engine == "SyncTalk":
            cmd = [
                python_path,
                "inference.py",
                "--video", input_video_path,
                "--audio", input_audio_path,
                "--output", output_path
            ]

        # Dynamically append user-supplied configuration flags
        if engine_config:
            for k, v in engine_config.items():
                flag = k if k.startswith("-") else f"--{k}"
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag)
                elif isinstance(v, list):
                    cmd.append(flag)
                    cmd.extend(str(item) for item in v)
                else:
                    cmd.append(flag)
                    cmd.append(str(v))

        logger.info(f"Running lip sync engine {engine} command: {' '.join(cmd)} (CWD: {code_dir})")

        # Inherit env variables to pass CUDA/RunPod parameters
        env = os.environ.copy()

        # Run process
        result = subprocess.run(
            cmd,
            cwd=code_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            check=False
        )

        # Log process stdout/stderr
        if result.stdout:
            logger.info(f"[{engine} STDOUT]\n{result.stdout}")
        if result.stderr:
            logger.warning(f"[{engine} STDERR]\n{result.stderr}")

        # Check return code
        if result.returncode != 0:
            # Clean up temp folder if it exists
            if temp_out_dir and os.path.exists(temp_out_dir):
                shutil.rmtree(temp_out_dir, ignore_errors=True)
            raise RuntimeError(
                f"Lip sync engine {engine} exited with error code {result.returncode}.\n"
                f"Details:\n{result.stderr or result.stdout}"
            )

        # Extract MuseTalk output from temp directory
        if engine == "MuseTalk" and temp_out_dir:
            try:
                mp4_files = [os.path.join(temp_out_dir, f) for f in os.listdir(temp_out_dir) if f.endswith(".mp4")]
                if not mp4_files:
                    raise FileNotFoundError(f"MuseTalk execution succeeded but no output video was created in {temp_out_dir}")
                # Get latest modified file
                mp4_files.sort(key=os.path.getmtime, reverse=True)
                shutil.move(mp4_files[0], output_path)
            finally:
                # Clean up temporary dir
                shutil.rmtree(temp_out_dir, ignore_errors=True)

        metadata = {
            "engine": engine,
            "engine_config": engine_config or {},
            "source_video_path": input_video_path,
            "source_audio_path": input_audio_path,
            "output_video_path": output_path,
            "status": "completed",
            "execution": "runtime_subprocess",
            "returncode": result.returncode,
        }

        # Write metadata sidecar file next to output
        try:
            with open(output_path + ".json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
        except Exception as exc:
            logger.warning(f"Could not save lip sync JSON metadata: {exc}")

        return output_path, metadata
