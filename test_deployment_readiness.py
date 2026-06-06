import os
import sys
import shutil
import tempfile
import logging
from pathlib import Path
from studio.database.db_manager import DatabaseManager

def test_deployment_readiness_pipeline():
    print("=== STARTING DEPLOYMENT READINESS TESTS ===")
    
    # 1. Test Path Abstraction and Environment Overrides
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    os.environ["STUDIO_ENV"] = "production"
    os.environ["STUDIO_ROOT"] = str(temp_path)
    os.environ["STUDIO_DB_PATH"] = str(temp_path / "custom_db" / "studio_prod.db")
    os.environ["STUDIO_OUTPUT_DIR"] = str(temp_path / "custom_outputs")
    os.environ["STUDIO_LOG_DIR"] = str(temp_path / "custom_logs")
    os.environ["STUDIO_MODEL_DIR"] = str(temp_path / "custom_models")
    os.environ["STUDIO_BACKUP_DIR"] = str(temp_path / "custom_backups")
    
    # Reload/import paths module to evaluate overridden env paths
    import importlib
    import studio.deployment.paths as paths
    importlib.reload(paths)
    
    assert paths.STUDIO_ROOT == temp_path
    assert paths.DB_PATH == temp_path / "custom_db" / "studio_prod.db"
    assert paths.OUTPUT_DIR == temp_path / "custom_outputs"
    assert paths.LOG_DIR == temp_path / "custom_logs"
    assert paths.MODEL_DIR == temp_path / "custom_models"
    assert paths.BACKUP_DIR == temp_path / "custom_backups"
    
    # Ensure directories are created successfully
    paths.ensure_directories()
    assert (temp_path / "custom_db").is_dir()
    assert (temp_path / "custom_outputs").is_dir()
    assert (temp_path / "custom_logs").is_dir()
    assert (temp_path / "custom_models").is_dir()
    assert (temp_path / "custom_backups").is_dir()
    print("Paths abstraction and folder auto-generation validated.")

    # 2. Test Centralized Logging setup
    import studio.deployment.logging_setup as logging_setup
    logging_setup.setup_centralized_logging(paths.LOG_DIR)
    
    logger_app = logging.getLogger("studio.app")
    logger_worker = logging.getLogger("studio.worker")
    logger_gen = logging.getLogger("studio.generation")
    logger_lipsync = logging.getLogger("studio.lipsync")
    logger_models = logging.getLogger("studio.models")
    
    logger_app.info("Test App Log Event")
    logger_worker.info("Test Worker Log Event")
    logger_gen.info("Test Gen Log Event")
    logger_lipsync.info("Test Lipsync Log Event")
    logger_models.info("Test Models Log Event")
    
    # Assert logs are isolated in their respective files
    assert (paths.LOG_DIR / "app.log").is_file()
    assert (paths.LOG_DIR / "worker.log").is_file()
    assert (paths.LOG_DIR / "generation.log").is_file()
    assert (paths.LOG_DIR / "lipsync.log").is_file()
    assert (paths.LOG_DIR / "models.log").is_file()
    
    with open(paths.LOG_DIR / "app.log", "r", encoding="utf-8") as f:
        content = f.read()
        assert "Test App Log Event" in content
        assert "Test Worker Log Event" not in content
        
    with open(paths.LOG_DIR / "worker.log", "r", encoding="utf-8") as f:
        content = f.read()
        assert "Test Worker Log Event" in content
        assert "Test App Log Event" not in content
        
    print("Centralized logs separation validated.")

    # 3. Test Diagnostics System Health Checks
    from studio.deployment.diagnostics import check_gpu_status, check_storage_status, verify_database_schema, check_worker_status, scan_models, run_system_startup_checks
    
    # Initialize a valid database
    db = DatabaseManager(db_path=str(paths.DB_PATH))
    
    # Storage Checks
    storage_info = check_storage_status(paths.STUDIO_ROOT, paths.DB_PATH)
    assert storage_info["total_gb"] > 0
    assert storage_info["free_gb"] > 0
    assert storage_info["db_size_mb"] >= 0
    
    # Database table audit
    db_audit = verify_database_schema(paths.DB_PATH)
    assert db_audit["connected"] is True
    assert db_audit["valid"] is True
    assert db_audit["table_count"] >= 13
    
    # Model scanning
    models_scanned = scan_models(paths.STUDIO_ROOT, paths.MODEL_DIR, db=db)
    assert "Flux Dev" in models_scanned
    assert "LTX-Video" in models_scanned
    assert "Wav2Lip" in models_scanned
    assert "LoRAs" in models_scanned
    
    # Startup validation checks report
    startup_report = run_system_startup_checks(paths.STUDIO_ROOT, paths.DB_PATH, paths.MODEL_DIR)
    assert startup_report["ready"] is True
    assert len(startup_report["critical_errors"]) == 0
    print("Diagnostics hardware, storage, and tables audit validated.")

    # 4. Test Backup and Restore system
    # Populate test data
    with db._get_connection() as conn:
        conn.execute("INSERT INTO projects (name, description) VALUES ('Prod Ready Project', 'Deployment Hardening')")
        conn.commit()
        
    # Write a test asset file in output dir
    dummy_asset_path = paths.STUDIO_ROOT / "characters" / "e2e_character.png"
    dummy_asset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dummy_asset_path, "w") as f:
        f.write("DUMMY IMAGE BYTES")
        
    from studio.deployment.backup import create_backup, restore_backup
    backup_zip = create_backup(paths.STUDIO_ROOT, paths.BACKUP_DIR, paths.DB_PATH)
    assert backup_zip.is_file()
    assert backup_zip.stat().st_size > 0
    print(f"Created system backup package: {backup_zip.name}")
    
    # Clear the existing project & dummy file to verify restore works
    with db._get_connection() as conn:
        conn.execute("DELETE FROM projects")
        conn.commit()
    os.remove(dummy_asset_path)
    
    # Assert database is indeed empty before restore
    assert len(db.list_projects()) == 0
    
    # Restore
    restore_success = restore_backup(backup_zip, paths.STUDIO_ROOT, paths.DB_PATH)
    assert restore_success is True
    
    # Reload DB and check project is restored
    db_restored = DatabaseManager(db_path=str(paths.DB_PATH))
    projects_list = db_restored.list_projects()
    assert len(projects_list) == 1
    assert projects_list[0]["name"] == "Prod Ready Project"
    
    # Check dummy file restored
    assert dummy_asset_path.is_file()
    print("Database and folder assets backup/restore verification validated.")

    # 5. Test Worker startup queue recovery
    # Put a job into running state
    with db_restored._get_connection() as conn:
        conn.execute("INSERT INTO render_queue (prompt, status) VALUES ('Test Interrupted Render', 'running')")
        conn.execute("INSERT INTO audio_queue (dialogue_text, voice_id, status) VALUES ('Test Interrupted Audio', 'am_adam', 'running')")
        conn.execute("INSERT INTO lip_sync_queue (engine, status) VALUES ('Wav2Lip', 'running')")
        conn.commit()
        
    from studio.generation.worker import recover_unfinished_jobs
    recover_unfinished_jobs(db_restored)
    
    # Assert they are recovered to queued state
    with db_restored._get_connection() as conn:
        cursor = conn.cursor()
        render_job = cursor.execute("SELECT status FROM render_queue ORDER BY id DESC LIMIT 1").fetchone()
        audio_job = cursor.execute("SELECT status FROM audio_queue ORDER BY id DESC LIMIT 1").fetchone()
        lipsync_job = cursor.execute("SELECT status FROM lip_sync_queue ORDER BY id DESC LIMIT 1").fetchone()
        
        assert render_job["status"] == "queued"
        assert audio_job["status"] == "queued"
        assert lipsync_job["status"] == "queued"
        
    print("Queue crash-recovery system validated.")
    print("=== ALL DEPLOYMENT READINESS TESTS PASSED ===")
    
    # Clear env
    del os.environ["STUDIO_ENV"]
    del os.environ["STUDIO_ROOT"]
    del os.environ["STUDIO_DB_PATH"]
    del os.environ["STUDIO_OUTPUT_DIR"]
    del os.environ["STUDIO_LOG_DIR"]
    del os.environ["STUDIO_MODEL_DIR"]
    del os.environ["STUDIO_BACKUP_DIR"]
    
    # Cleanup temp folder
    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_deployment_readiness_pipeline()
