import logging
import sys
from pathlib import Path

def setup_centralized_logging(log_dir: Path):
    """
    Sets up centralized and separated log files for different application modules:
    - app.log (Server/UI execution)
    - worker.log (Background worker loop)
    - generation.log (FLUX, LTX, WAN, Hunyuan generation engines)
    - lipsync.log (Wav2Lip, MuseTalk, SyncTalk subprocesses)
    - models.log (Model scanner operations)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
    
    # Define handlers configurations
    log_targets = {
        "studio.app": log_dir / "app.log",
        "studio.worker": log_dir / "worker.log",
        "studio.generation": log_dir / "generation.log",
        "studio.lipsync": log_dir / "lipsync.log",
        "studio.models": log_dir / "models.log",
    }
    
    for logger_name, log_path in log_targets.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Avoid routing back to root handlers duplication
        
        if logger.handlers:
            logger.handlers.clear()
            
        # Write to specific isolated file
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Parallel stream stdout for container logs / runpod output checks
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    # Configure root logger to output to app.log by default
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.handlers:
        root_logger.handlers.clear()
        
    root_file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    root_file_handler.setFormatter(formatter)
    root_logger.addHandler(root_file_handler)
    
    root_stream_handler = logging.StreamHandler(sys.stdout)
    root_stream_handler.setFormatter(formatter)
    root_logger.addHandler(root_stream_handler)

    logging.info(f"Centralized logs initialized under {log_dir}")
