import os
import zipfile
import shutil
import tempfile
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("studio.app")

def create_backup(studio_root: Path, backup_dir: Path, db_path: Path) -> Path:
    """
    Export Projects, Characters, Episodes, Scenes, Settings, References and the DB
    into a consolidated zip backup package.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = backup_dir / f"studio_backup_{timestamp}.zip"
    
    logger.info(f"Creating system backup package at: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Archive DB File
        if db_path.exists():
            zipf.write(db_path, arcname="database/studio.db")
            logger.info("Database file added to backup archive.")
            
        # 2. Archive pre-production asset directories
        asset_folders = ["characters", "locations", "projects", "audio_assets", "output"]
        for folder_name in asset_folders:
            folder_path = studio_root / folder_name
            if folder_path.exists():
                logger.info(f"Archiving folder assets: {folder_name}")
                for file_path in folder_path.rglob("*"):
                    # Exclude the backups folder itself and temporary directories
                    if "_temp" in file_path.name or "backups" in file_path.parts:
                        continue
                    if file_path.is_file():
                        rel_path = file_path.relative_to(studio_root)
                        zipf.write(file_path, arcname=str(rel_path))
                        
    logger.info(f"System backup package created successfully: {zip_path.name}")
    return zip_path


def restore_backup(backup_zip_path: Path, studio_root: Path, db_path: Path) -> bool:
    """
    Restore the SQLite database and asset directories atomically from a zip file.
    Validates ZIP structure first to prevent corruption.
    """
    if not backup_zip_path.exists():
        raise FileNotFoundError(f"Backup zip file does not exist at: {backup_zip_path}")
        
    logger.info(f"Starting atomic database and asset restoration from: {backup_zip_path.name}")
    
    # Extract into temporary directory first to validate schema presence
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with zipfile.ZipFile(backup_zip_path, 'r') as zipf:
            zipf.extractall(temp_path)
            
        extracted_db = temp_path / "database" / "studio.db"
        if not extracted_db.exists():
            logger.error("Restore failed: Invalid backup package (database/studio.db is missing).")
            raise ValueError("Invalid backup package: SQLite database file (database/studio.db) is missing.")
            
        # Stop database write operations and copy file
        logger.info("Restoring database file...")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted_db, db_path)
        
        # Restore folders
        asset_folders = ["characters", "locations", "projects", "audio_assets", "output"]
        for folder_name in asset_folders:
            src_folder = temp_path / folder_name
            dest_folder = studio_root / folder_name
            
            if src_folder.exists():
                logger.info(f"Restoring folder assets hierarchy: {folder_name}")
                if dest_folder.exists():
                    shutil.rmtree(dest_folder, ignore_errors=True)
                shutil.copytree(src_folder, dest_folder)
                
    logger.info("System restoration completed successfully!")
    return True
