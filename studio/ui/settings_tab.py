import os
import gradio as gr
from studio.database.db_manager import DatabaseManager
from studio.deployment.paths import STUDIO_ROOT, DB_PATH, LOG_DIR

def create_settings_tab(db: DatabaseManager):
    with gr.TabItem("Settings & Status"):
        gr.Markdown("### ⚙️ System Settings & Diagnostic Console")
        
        # 🏥 System Health & Model Discovery Dashboard
        with gr.Accordion("🏥 System Health & Model Discovery Dashboard", open=True):
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown("#### 💻 Hardware & Storage status")
                    gpu_status_box = gr.Markdown("*Run diagnostic audit to check GPU/CUDA visibility.*")
                    storage_status_box = gr.Markdown("*Run diagnostic audit to check disk space and db file sizes.*")
                with gr.Column(variant="panel"):
                    gr.Markdown("#### 🧠 Model Discovery Checkpoints")
                    model_status_box = gr.Markdown("*Run diagnostic audit to scan configured model files.*")
            btn_refresh_health = gr.Button("🔍 Run System Diagnostic Audit", variant="primary")
            
        with gr.Row():
            # Left side: Path Info & Configurations
            with gr.Column(variant="panel"):
                gr.Markdown("#### 📂 Environment Directories")
                studio_root_box = gr.Textbox(label="STUDIO_ROOT Path", value=str(STUDIO_ROOT), interactive=False)
                db_path_box = gr.Textbox(label="Database File Path", value=str(DB_PATH), interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("#### 📦 Database & Assets Backup System")
                with gr.Row():
                    btn_create_backup = gr.Button("📤 Create ZIP Backup", variant="secondary")
                    btn_restore_backup = gr.Button("📥 Restore Selected Backup", variant="stop")
                
                # Fetch initial choices safely
                initial_choices = []
                backups_folder = STUDIO_ROOT / "backups"
                if backups_folder.is_dir():
                    initial_choices = sorted([f.name for f in backups_folder.glob("*.zip")], reverse=True)
                
                backup_dropdown = gr.Dropdown(
                    label="Available Backup Packages",
                    choices=initial_choices,
                    value=initial_choices[0] if initial_choices else None
                )
                backup_msg = gr.Markdown("*Database snapshot back up folder: `studio/backups`*")
                
                gr.Markdown("---")
                gr.Markdown("#### 💾 Session Configuration Keys")
                
                settings_list = gr.Dataframe(
                    headers=["Setting Key", "Setting Value"],
                    datatype=["str", "str"],
                    row_count=5,
                    col_count=2,
                    interactive=False
                )
                
                btn_refresh_settings = gr.Button("Refresh Environment Info", variant="secondary")
                btn_clear_session = gr.Button("Reset Session Context", variant="stop")
                settings_msg = gr.Markdown("")

            # Right side: Centralized Log Terminal
            with gr.Column(variant="panel"):
                gr.Markdown("#### 💻 Live System Console Logs")
                log_source = gr.Dropdown(
                    label="Log Source",
                    choices=[
                        "Application (app.log)", 
                        "Worker (worker.log)", 
                        "Generation (generation.log)", 
                        "Lip Sync (lipsync.log)", 
                        "Models (models.log)"
                    ],
                    value="Application (app.log)"
                )
                log_box = gr.Textbox(
                    label="Log File Content (Last 100 Lines)",
                    value="[INFO] System startup diagnostic initialized...\n",
                    lines=18,
                    max_lines=28,
                    interactive=False
                )
                btn_refresh_logs = gr.Button("Reload Logs", variant="secondary")

        # ==========================================
        # EVENT FUNCTIONS
        # ==========================================
        def read_logs_fn(source):
            mapping = {
                "Application (app.log)": LOG_DIR / "app.log",
                "Worker (worker.log)": LOG_DIR / "worker.log",
                "Generation (generation.log)": LOG_DIR / "generation.log",
                "Lip Sync (lipsync.log)": LOG_DIR / "lipsync.log",
                "Models (models.log)": LOG_DIR / "models.log",
            }
            log_path = mapping.get(source, LOG_DIR / "app.log")
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    last_lines = lines[-100:] # Get last 100 lines
                    return "".join(last_lines)
                except Exception as e:
                    return f"[ERROR] Failed to read log file: {e}"
            return f"[INFO] Log file {log_path.name} not found yet. It will be generated when events occur."

        def run_diagnostics_fn():
            from studio.deployment.diagnostics import check_gpu_status, check_storage_status, verify_database_schema, check_worker_status, scan_models
            
            # GPU
            gpu = check_gpu_status()
            gpu_md = f"**CUDA Available:** {'✅ Yes' if gpu['cuda_available'] else '❌ No'}\n\n"
            gpu_md += f"**GPU Count:** `{gpu['device_count']}`\n\n"
            gpu_md += f"**Active Device:** `{gpu['device_name']}`"
            
            # Storage & DB
            storage = check_storage_status(STUDIO_ROOT, DB_PATH)
            db_audit = verify_database_schema(DB_PATH)
            worker = check_worker_status()
            
            storage_md = f"**Free Disk Space:** `{storage['free_gb']} GB` / `{storage['total_gb']} GB`\n\n"
            storage_md += f"**Database Size:** `{storage['db_size_mb']} MB`\n\n"
            storage_md += f"**Database Status:** {'✅ Schema Valid' if db_audit['valid'] else '❌ Schema Corrupt'}\n\n"
            storage_md += f"**Background Worker:** {'✅ Active' if worker['running'] else '❌ Stopped'}"
            if storage['warning']:
                storage_md += f"\n\n🚨 *Warning: {storage['warning_message']}*"
            if not db_audit['valid']:
                storage_md += f"\n\n🚨 *Database tables missing: {', '.join(db_audit['missing_tables'])}*"
                
            # Models Scanner
            models = scan_models(STUDIO_ROOT, STUDIO_ROOT / "models", db=db)
            models_md = ""
            for name, meta in models.items():
                if name == "LoRAs":
                    models_md += f"- **LoRAs:** Status: `{meta['status']}` (Count: `{meta['count']}`)\n"
                else:
                    models_md += f"- **{name}:** Status: `{meta['status']}` | Version: `{meta['version']}`\n"
                    
            return gpu_md, storage_md, models_md

        def load_settings_table():
            settings = db.get_all_settings()
            table_data = [[k, v] for k, v in settings.items()]
            if not table_data:
                table_data = [["No keys", "No session saved"]]
            return table_data

        def refresh_backups_fn():
            from studio.deployment.paths import BACKUP_DIR
            if BACKUP_DIR.is_dir():
                zips = sorted([f.name for f in BACKUP_DIR.glob("*.zip")], reverse=True)
                return gr.update(choices=zips, value=zips[0] if zips else None)
            return gr.update(choices=[], value=None)

        def create_backup_fn():
            from studio.deployment.backup import create_backup
            from studio.deployment.paths import BACKUP_DIR
            try:
                zip_path = create_backup(STUDIO_ROOT, BACKUP_DIR, DB_PATH)
                zips = sorted([f.name for f in BACKUP_DIR.glob("*.zip")], reverse=True)
                return gr.update(choices=zips, value=zip_path.name), f"✅ Backup created: `{zip_path.name}`"
            except Exception as e:
                return gr.update(), f"❌ Backup failed: {e}"

        def restore_backup_fn(zip_name):
            if not zip_name:
                return "❌ Please select a backup package first."
            from studio.deployment.backup import restore_backup
            from studio.deployment.paths import BACKUP_DIR
            zip_path = BACKUP_DIR / zip_name
            try:
                restore_backup(zip_path, STUDIO_ROOT, DB_PATH)
                return f"✅ Restore successful from `{zip_name}`! Refresh app page to load restored state."
            except Exception as e:
                return f"❌ Restore failed: {e}"

        def reset_session_fn(source):
            with db._get_connection() as conn:
                conn.execute("DELETE FROM settings WHERE key IN ('session_project', 'session_episode', 'session_scene')")
                conn.commit()
            
            import logging
            logger = logging.getLogger("studio.app")
            logger.info("Session context reset by user.")
            
            return load_settings_table(), "✅ Session memory cleared. Active focuses will be empty on reload.", read_logs_fn(source)

        def refresh_tab_fn(source):
            return load_settings_table(), "✅ Environment paths and configurations loaded.", read_logs_fn(source), refresh_backups_fn()

        # Wire up event listeners
        btn_refresh_logs.click(fn=read_logs_fn, inputs=[log_source], outputs=[log_box])
        log_source.change(fn=read_logs_fn, inputs=[log_source], outputs=[log_box])
        
        btn_refresh_settings.click(fn=refresh_tab_fn, inputs=[log_source], outputs=[settings_list, settings_msg, log_box, backup_dropdown])
        btn_clear_session.click(fn=reset_session_fn, inputs=[log_source], outputs=[settings_list, settings_msg, log_box])
        btn_refresh_health.click(fn=run_diagnostics_fn, inputs=None, outputs=[gpu_status_box, storage_status_box, model_status_box])
        
        btn_create_backup.click(fn=create_backup_fn, inputs=None, outputs=[backup_dropdown, backup_msg])
        btn_restore_backup.click(fn=restore_backup_fn, inputs=[backup_dropdown], outputs=[backup_msg])
