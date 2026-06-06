import os
import shutil
import sqlite3
import threading
import logging
from pathlib import Path

logger = logging.getLogger("studio.models")

def check_gpu_status() -> dict:
    """Check PyTorch GPU visibility and CUDA diagnostics."""
    gpu_info = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "N/A",
        "gpu_available": False
    }
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        gpu_info["device_count"] = torch.cuda.device_count()
        if gpu_info["cuda_available"] and gpu_info["device_count"] > 0:
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["gpu_available"] = True
    except ImportError:
        pass
    return gpu_info


def check_storage_status(studio_root: Path, db_path: Path, min_free_gb: float = 10.0) -> dict:
    """Evaluate free disk storage space and active database sizes on disk."""
    db_size_bytes = db_path.stat().st_size if db_path.exists() else 0
    db_size_mb = db_size_bytes / (1024 * 1024)
    
    # Check disk usage where outputs reside
    total, used, free = shutil.disk_usage(studio_root)
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    
    threshold_exceeded = free_gb < min_free_gb
    
    return {
        "total_gb": round(total_gb, 2),
        "free_gb": round(free_gb, 2),
        "db_size_mb": round(db_size_mb, 2),
        "warning": threshold_exceeded,
        "warning_message": f"⚠️ Low Disk Space Alert: Only {free_gb:.2f} GB free (threshold: {min_free_gb} GB)." if threshold_exceeded else None
    }


def verify_database_schema(db_path: Path) -> dict:
    """Audit database tables integrity and check that all required tables exist."""
    required_tables = {
        "projects", "episodes", "scenes", "shots", "characters",
        "locations", "render_queue", "audio_queue", "lip_sync_queue",
        "settings", "generated_videos", "generated_images", "lipsync_jobs"
    }
    
    audit = {
        "connected": False,
        "missing_tables": [],
        "valid": False,
        "table_count": 0
    }
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        audit["connected"] = True
        
        cursor = conn.cursor()
        rows = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        existing_tables = {row["name"] for row in rows}
        audit["table_count"] = len(existing_tables)
        
        missing = required_tables - existing_tables
        audit["missing_tables"] = list(missing)
        audit["valid"] = len(missing) == 0
    except Exception as e:
        logger.error(f"Database diagnostics error: {e}")
        audit["error"] = str(e)
    finally:
        if conn:
            conn.close()
            
    return audit


def check_worker_status() -> dict:
    """Verify if the background RenderWorker thread is active."""
    threads = threading.enumerate()
    worker_thread = next((t for t in threads if t.name == "RenderWorker"), None)
    
    return {
        "running": worker_thread is not None,
        "thread_name": worker_thread.name if worker_thread else None,
        "daemon": worker_thread.daemon if worker_thread else False,
        "status_text": "Active" if worker_thread else "Stopped"
    }


def scan_models(studio_root: Path, model_dir: Path, db=None) -> dict:
    """
    Scans model directory and database settings configurations to detect:
    Flux Dev, Flux Kontext, LTX Video, Wan 2.2, Hunyuan Video, Kokoro TTS,
    MuseTalk, Wav2Lip, SyncTalk and LoRAs.
    """
    discovery = {}

    def get_configured_or_fallback(db_key, default_subpath):
        val = ""
        if db:
            try:
                val = db.get_setting(db_key, "")
            except Exception:
                pass
        return Path(val) if val else (model_dir / default_subpath)

    # 1. FLUX DEV
    flux_dev = get_configured_or_fallback("flux_dev_path", "flux/flux1-dev.sft")
    discovery["Flux Dev"] = {
        "path": str(flux_dev),
        "status": "Available" if flux_dev.is_file() else "Missing",
        "version": "v1.0 (Dev)" if flux_dev.is_file() else "N/A"
    }

    # 2. FLUX KONTEXT
    flux_kontext = get_configured_or_fallback("flux_kontext_path", "flux/flux1-kontext.sft")
    discovery["Flux Kontext"] = {
        "path": str(flux_kontext),
        "status": "Available" if flux_kontext.is_file() else "Missing",
        "version": "v1.0 (Kontext)" if flux_kontext.is_file() else "N/A"
    }

    # 3. LTX VIDEO
    ltx = get_configured_or_fallback("ltx_model_path", "ltx/ltx-video-2b.sft")
    discovery["LTX-Video"] = {
        "path": str(ltx),
        "status": "Available" if ltx.is_file() else "Missing",
        "version": "v0.9.1 (2B)" if ltx.is_file() else "N/A"
    }

    # 4. WAN 2.2
    wan = get_configured_or_fallback("wan_model_path", "wan/wan2.2-i2v.safetensors")
    discovery["Wan 2.2"] = {
        "path": str(wan),
        "status": "Available" if wan.is_file() else "Missing",
        "version": "v2.2 (14B)" if wan.is_file() else "N/A"
    }

    # 5. HUNYUAN VIDEO
    hunyuan = get_configured_or_fallback("hunyuan_model_path", "hunyuan/hunyuan_video.sft")
    discovery["Hunyuan Video"] = {
        "path": str(hunyuan),
        "status": "Available" if hunyuan.is_file() else "Missing",
        "version": "v1.0" if hunyuan.is_file() else "N/A"
    }

    # 6. KOKORO TTS
    try:
        import kokoro
        discovery["Kokoro TTS"] = {
            "path": "Python Package",
            "status": "Available",
            "version": getattr(kokoro, "__version__", "Installed")
        }
    except ImportError:
        discovery["Kokoro TTS"] = {
            "path": "Kokoro pip package",
            "status": "Missing",
            "version": "N/A"
        }

    # 7. Wav2Lip
    w2l_dir = get_configured_or_fallback("lipsync_wav2lip_code_dir", "Wav2Lip")
    w2l_chk = get_configured_or_fallback("lipsync_wav2lip_checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth")
    discovery["Wav2Lip"] = {
        "path": str(w2l_chk),
        "status": "Available" if w2l_dir.is_dir() and w2l_chk.is_file() else "Missing",
        "version": "GAN-v1" if w2l_chk.is_file() else "N/A"
    }

    # 8. MuseTalk
    mt_dir = get_configured_or_fallback("lipsync_musetalk_code_dir", "MuseTalk")
    mt_chk = get_configured_or_fallback("lipsync_musetalk_checkpoint_path", "MuseTalk/models/musetalk/musetalk.json")
    discovery["MuseTalk"] = {
        "path": str(mt_chk),
        "status": "Available" if mt_dir.is_dir() and mt_chk.is_file() else "Missing",
        "version": "v1.0" if mt_chk.is_file() else "N/A"
    }

    # 9. SyncTalk
    st_dir = get_configured_or_fallback("lipsync_synctalk_code_dir", "SyncTalk")
    st_chk = get_configured_or_fallback("lipsync_synctalk_checkpoint_path", "SyncTalk/checkpoints/synctalk.pth")
    discovery["SyncTalk"] = {
        "path": str(st_chk),
        "status": "Available" if st_dir.is_dir() and st_chk.is_file() else "Missing",
        "version": "v1.0" if st_chk.is_file() else "N/A"
    }

    # 10. LoRAs Scanning
    lora_dir = model_dir / "loras"
    loras = []
    if lora_dir.is_dir():
        for f in lora_dir.glob("*.safetensors"):
            size_mb = f.stat().st_size / (1024 * 1024)
            loras.append({
                "name": f.name,
                "size_mb": round(size_mb, 2)
            })
            
    discovery["LoRAs"] = {
        "path": str(lora_dir),
        "count": len(loras),
        "status": "Available" if lora_dir.is_dir() else "Missing",
        "items": loras
    }

    # Validate file sizes to verify if model downloads were corrupted/empty
    for name, item in discovery.items():
        if name == "Kokoro TTS" or name == "LoRAs":
            continue
        p = Path(item["path"])
        if item["status"] == "Available":
            if p.exists() and p.stat().st_size < 1000:  # File is under 1KB
                item["status"] = "Invalid (File size too small)"
            elif not p.exists():
                item["status"] = "Missing"

    return discovery


def run_system_startup_checks(studio_root: Path, db_path: Path, model_dir: Path) -> dict:
    """Executes the full diagnostic suite on startup to ensure deployment readiness."""
    report = {
        "gpu": check_gpu_status(),
        "storage": check_storage_status(studio_root, db_path),
        "db": verify_database_schema(db_path),
        "worker": check_worker_status(),
        "ready": False,
        "critical_errors": []
    }
    
    # Evaluate readiness
    if not report["db"]["connected"]:
        report["critical_errors"].append("❌ CRITICAL: Database connection failed.")
    elif not report["db"]["valid"]:
        report["critical_errors"].append(f"❌ CRITICAL: Database schema corrupted. Missing tables: {', '.join(report['db']['missing_tables'])}")
        
    if not studio_root.exists():
        report["critical_errors"].append("❌ CRITICAL: STUDIO_ROOT directory does not exist.")
        
    report["ready"] = len(report["critical_errors"]) == 0
    return report
