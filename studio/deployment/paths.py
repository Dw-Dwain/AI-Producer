import os
from pathlib import Path

# RunPod/Linux production mode flag
# "production" defaults to standard RunPod workspace layout
STUDIO_ENV = os.getenv("STUDIO_ENV", "development").lower()

# Resolve Studio Root directory (dynamic parent of this deployment folder)
# __file__ is studio/deployment/paths.py, so parent.parent is studio/
default_root = Path(__file__).resolve().parent.parent
override_root = os.getenv("STUDIO_ROOT")
STUDIO_ROOT = Path(override_root).resolve() if override_root else default_root

# Database path configuration
default_db_path = STUDIO_ROOT / "database" / "studio.db"
override_db_path = os.getenv("STUDIO_DB_PATH")
DB_PATH = Path(override_db_path).resolve() if override_db_path else default_db_path

# Output directory configuration
override_output = os.getenv("STUDIO_OUTPUT_DIR")
OUTPUT_DIR = Path(override_output).resolve() if override_output else (STUDIO_ROOT / "output")

# Log files root directory configuration
# Defaults to standard /workspace/studio/logs on RunPod production instances
override_logs = os.getenv("STUDIO_LOG_DIR")
if override_logs:
    LOG_DIR = Path(override_logs).resolve()
else:
    if STUDIO_ENV == "production" or os.path.exists("/workspace"):
        LOG_DIR = Path("/workspace/studio/logs")
    else:
        LOG_DIR = STUDIO_ROOT / "logs"

# Model directory scanner path
override_models = os.getenv("STUDIO_MODEL_DIR")
MODEL_DIR = Path(override_models).resolve() if override_models else (STUDIO_ROOT / "models")

# Backup storage location
override_backups = os.getenv("STUDIO_BACKUP_DIR")
BACKUP_DIR = Path(override_backups).resolve() if override_backups else (STUDIO_ROOT / "backups")


def ensure_directories():
    """Create all required runtime directory structures dynamically if missing."""
    STUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pre-production asset folders
    (STUDIO_ROOT / "characters").mkdir(parents=True, exist_ok=True)
    (STUDIO_ROOT / "locations").mkdir(parents=True, exist_ok=True)
    (STUDIO_ROOT / "projects").mkdir(parents=True, exist_ok=True)
    (STUDIO_ROOT / "audio_assets").mkdir(parents=True, exist_ok=True)
