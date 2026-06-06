import os
import sqlite3
import json
from datetime import datetime

# Determine project root path dynamically via deployment paths
from studio.deployment.paths import STUDIO_ROOT, DB_PATH
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = str(DB_PATH)
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")


class DatabaseManager:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        # Create database directories if they don't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Read and run schema setup
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema_sql = f.read()
            with self._get_connection() as conn:
                conn.executescript(schema_sql)
                conn.commit()
            self._migrate_db()
            self._init_default_settings()
        else:
            raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")

    def _init_default_settings(self):
        default_keys = {
            "lipsync_wav2lip_python_path": "python",
            "lipsync_wav2lip_code_dir": os.path.join(STUDIO_ROOT, "models", "Wav2Lip") if os.name == "nt" else "/workspace/Wav2Lip",
            "lipsync_wav2lip_checkpoint_path": os.path.join(STUDIO_ROOT, "models", "Wav2Lip", "checkpoints", "wav2lip_gan.pth") if os.name == "nt" else "/workspace/Wav2Lip/checkpoints/wav2lip_gan.pth",
            
            "lipsync_musetalk_python_path": "python",
            "lipsync_musetalk_code_dir": os.path.join(STUDIO_ROOT, "models", "MuseTalk") if os.name == "nt" else "/workspace/MuseTalk",
            "lipsync_musetalk_checkpoint_path": os.path.join(STUDIO_ROOT, "models", "MuseTalk", "models", "musetalk", "musetalk.json") if os.name == "nt" else "/workspace/MuseTalk/models/musetalk/musetalk.json",
            
            "lipsync_synctalk_python_path": "python",
            "lipsync_synctalk_code_dir": os.path.join(STUDIO_ROOT, "models", "SyncTalk") if os.name == "nt" else "/workspace/SyncTalk",
            "lipsync_synctalk_checkpoint_path": os.path.join(STUDIO_ROOT, "models", "SyncTalk", "checkpoints", "synctalk.pth") if os.name == "nt" else "/workspace/SyncTalk/checkpoints/synctalk.pth",
        }
        with self._get_connection() as conn:
            for k, v in default_keys.items():
                row = conn.execute("SELECT 1 FROM settings WHERE key = ?", (k,)).fetchone()
                if not row:
                    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?)", (k, v))
            conn.commit()

    def detect_and_update_engine_status(self):
        import shutil
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        status_suffix = " (GPU active)" if cuda_available else " (CPU fallback)"

        engines = ["wav2lip", "musetalk", "synctalk"]
        for eng in engines:
            python_val = self.get_setting(f"lipsync_{eng}_python_path", "python")
            code_dir = self.get_setting(f"lipsync_{eng}_code_dir", "")
            chk_path = self.get_setting(f"lipsync_{eng}_checkpoint_path", "")

            code_ok = os.path.isdir(code_dir) if code_dir else False
            chk_ok = os.path.isfile(chk_path) if chk_path else False

            if code_ok and chk_ok:
                status = f"Available{status_suffix}"
            else:
                missing = []
                if not code_ok:
                    missing.append("Code Dir missing")
                if not chk_ok:
                    missing.append("Checkpoint missing")
                status = f"Not Installed ({', '.join(missing)})"

            self.set_setting(f"lipsync_{eng}_status", status)

    def _migrate_db(self):
        def ensure_column(conn, table_name, column_name, column_type):
            cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
            if column_name not in cols:
                try:
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                    conn.commit()
                except sqlite3.OperationalError as e:
                    print(f"Migration warning: {e}")

        # Add new columns to characters table if missing
        expected_columns = {
            "biography": "TEXT",
            "personality": "TEXT",
            "prompt_template": "TEXT",
            "wardrobe_notes": "TEXT",
            "expression_notes": "TEXT",
            "voice_notes": "TEXT",
            "dna_hair": "TEXT",
            "dna_eyes": "TEXT",
            "dna_body_type": "TEXT",
            "dna_clothing": "TEXT",
            "dna_ethnicity": "TEXT",
            "dna_description": "TEXT",
            "voice_id": "TEXT"
        }
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(characters)")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            
            for col_name, col_type in expected_columns.items():
                if col_name not in existing_cols:
                    try:
                        conn.execute(f"ALTER TABLE characters ADD COLUMN {col_name} {col_type}")
                        conn.commit()
                    except sqlite3.OperationalError as e:
                        print(f"Migration warning: {e}")
            
            # Check prompt_template in locations table
            cursor.execute("PRAGMA table_info(locations)")
            loc_cols = {row["name"] for row in cursor.fetchall()}
            if "prompt_template" not in loc_cols:
                try:
                    conn.execute("ALTER TABLE locations ADD COLUMN prompt_template TEXT")
                    conn.commit()
                except sqlite3.OperationalError as e:
                    print(f"Migration warning: {e}")

            # Phase 4: Ensure generated_images table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='generated_images'"
            )
            if not cursor.fetchone():
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS generated_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER,
                        scene_id INTEGER,
                        character_id INTEGER,
                        location_id INTEGER,
                        model TEXT NOT NULL,
                        prompt TEXT,
                        negative_prompt TEXT,
                        seed INTEGER,
                        width INTEGER,
                        height INTEGER,
                        steps INTEGER,
                        guidance_scale REAL,
                        file_path TEXT NOT NULL,
                        thumbnail_path TEXT,
                        status TEXT DEFAULT 'pending',
                        approved_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
                        FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE SET NULL,
                        FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL,
                        FOREIGN KEY(location_id) REFERENCES locations(id) ON DELETE SET NULL
                    );
                """)
                conn.commit()

            # Phase 5: Ensure generated_videos table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='generated_videos'"
            )
            if not cursor.fetchone():
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS generated_videos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER,
                        scene_id INTEGER,
                        character_id INTEGER,
                        location_id INTEGER,
                        pipeline TEXT NOT NULL,
                        preset TEXT,
                        prompt TEXT,
                        negative_prompt TEXT,
                        seed INTEGER,
                        width INTEGER,
                        height INTEGER,
                        fps INTEGER,
                        num_frames INTEGER,
                        steps INTEGER,
                        guidance_scale REAL,
                        lora_path TEXT,
                        lora_weight REAL,
                        reference_image_path TEXT,
                        file_path TEXT NOT NULL,
                        thumbnail_path TEXT,
                        duration_seconds REAL,
                        status TEXT DEFAULT 'pending',
                        approved_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
                        FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE SET NULL,
                        FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL,
                        FOREIGN KEY(location_id) REFERENCES locations(id) ON DELETE SET NULL
                    );
                """)
                conn.commit()
            
            # Phase 6: Ensure model_family column exists
            cursor.execute("PRAGMA table_info(generated_videos)")
            vid_cols = {row["name"] for row in cursor.fetchall()}
            if "model_family" not in vid_cols:
                try:
                    conn.execute("ALTER TABLE generated_videos ADD COLUMN model_family TEXT DEFAULT 'LTX-Video'")
                    conn.commit()
                except sqlite3.OperationalError as e:
                    print(f"Migration warning: {e}")

            # Asset lineage and shot-driven workflow columns
            for table_name, columns in {
                "generated_images": {
                    "episode_id": "INTEGER",
                    "shot_id": "INTEGER",
                    "lora_name": "TEXT",
                    "references_json": "TEXT",
                    "model_name": "TEXT",
                },
                "generated_videos": {
                    "episode_id": "INTEGER",
                    "shot_id": "INTEGER",
                    "dialogue_line_id": "INTEGER",
                    "lora_name": "TEXT",
                    "references_json": "TEXT",
                    "model_name": "TEXT",
                    "performance_json": "TEXT",
                    "camera_preset": "TEXT",
                    "movement_preset": "TEXT",
                    "lens_preset": "TEXT",
                },
                "render_queue": {
                    "episode_id": "INTEGER",
                    "shot_id": "INTEGER",
                    "dialogue_line_id": "INTEGER",
                    "model_name": "TEXT",
                    "lora_name": "TEXT",
                    "references_json": "TEXT",
                    "performance_json": "TEXT",
                    "camera_preset": "TEXT",
                    "movement_preset": "TEXT",
                    "lens_preset": "TEXT",
                    "continuity_notes": "TEXT",
                },
                "audio_assets": {
                    "project_id": "INTEGER",
                    "episode_id": "INTEGER",
                    "scene_id": "INTEGER",
                    "shot_id": "INTEGER",
                    "character_id": "INTEGER",
                    "voice_id": "TEXT",
                    "emotion": "TEXT",
                    "intensity": "INTEGER",
                    "speaking_style": "TEXT",
                },
                "audio_queue": {
                    "shot_id": "INTEGER",
                    "dialogue_line_id": "INTEGER",
                    "emotion": "TEXT",
                    "intensity": "INTEGER DEFAULT 5",
                    "expression": "TEXT",
                    "speaking_style": "TEXT",
                },
                "dialogue_lines": {
                    "shot_id": "INTEGER",
                    "emotion": "TEXT",
                    "intensity": "INTEGER DEFAULT 5",
                    "expression": "TEXT",
                    "speaking_style": "TEXT",
                    "performance_notes": "TEXT",
                },
                "shots": {
                    "title": "TEXT",
                    "goal": "TEXT",
                    "camera_language": "TEXT",
                    "lens_preset": "TEXT",
                    "movement_preset": "TEXT",
                    "prompt": "TEXT",
                    "negative_prompt": "TEXT",
                    "references_json": "TEXT",
                    "status": "TEXT DEFAULT 'pending'",
                    "last_consistency_score": "REAL",
                    "override_consistency_warning": "INTEGER DEFAULT 0",
                },
                "character_memory": {
                    "best_references_json": "TEXT",
                    "approved_assets_json": "TEXT",
                    "last_known_wardrobe": "TEXT",
                    "last_known_location_id": "INTEGER",
                },
            }.items():
                for col_name, col_type in columns.items():
                    ensure_column(conn, table_name, col_name, col_type)

            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS lip_sync_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    episode_id INTEGER,
                    scene_id INTEGER,
                    shot_id INTEGER,
                    character_id INTEGER,
                    dialogue_line_id INTEGER,
                    audio_asset_id INTEGER,
                    source_video_id INTEGER,
                    source_audio_path TEXT,
                    source_video_path TEXT,
                    engine TEXT NOT NULL DEFAULT 'Wav2Lip',
                    engine_config_json TEXT,
                    status TEXT DEFAULT 'queued',
                    error_message TEXT,
                    output_video_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS lipsync_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_job_id INTEGER,
                    engine TEXT NOT NULL,
                    input_audio_path TEXT,
                    input_video_path TEXT,
                    output_video_path TEXT,
                    status TEXT DEFAULT 'pending',
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS character_performance_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id INTEGER UNIQUE NOT NULL,
                    voice_id TEXT,
                    default_emotion TEXT DEFAULT 'neutral',
                    default_intensity INTEGER DEFAULT 5,
                    default_expression TEXT DEFAULT 'neutral',
                    speaking_style TEXT DEFAULT 'natural',
                    notes TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS consistency_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    episode_id INTEGER,
                    scene_id INTEGER,
                    shot_id INTEGER,
                    character_id INTEGER,
                    face_drift REAL DEFAULT 0.0,
                    hair_drift REAL DEFAULT 0.0,
                    age_drift REAL DEFAULT 0.0,
                    wardrobe_drift REAL DEFAULT 0.0,
                    location_drift REAL DEFAULT 0.0,
                    reference_strength REAL DEFAULT 1.0,
                    consistency_score REAL DEFAULT 1.0,
                    warning_level TEXT DEFAULT 'ok',
                    warnings_json TEXT,
                    suggestions_json TEXT,
                    override_enabled INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS camera_presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    camera_language TEXT NOT NULL,
                    lens_preset TEXT,
                    movement_preset TEXT,
                    framing_notes TEXT,
                    prompt_addition TEXT,
                    negative_prompt_addition TEXT
                );

                CREATE TABLE IF NOT EXISTS episode_assemblies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    episode_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT DEFAULT 'draft',
                    settings_json TEXT,
                    preview_json TEXT,
                    export_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(project_id, episode_id, name)
                );

                CREATE TABLE IF NOT EXISTS episode_assembly_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    assembly_id INTEGER NOT NULL,
                    item_type TEXT NOT NULL,
                    scene_id INTEGER,
                    shot_id INTEGER,
                    source_video_id INTEGER,
                    source_audio_asset_id INTEGER,
                    subtitle_text TEXT,
                    music_path TEXT,
                    sequence_order INTEGER NOT NULL,
                    duration_seconds REAL DEFAULT 0.0,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_shots_scene ON shots(scene_id, shot_number);
                CREATE INDEX IF NOT EXISTS idx_render_queue_status ON render_queue(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_audio_queue_status ON audio_queue(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_lipsync_queue_status ON lip_sync_queue(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_consistency_reports_scope ON consistency_reports(project_id, episode_id, scene_id, shot_id, character_id);
                """
            )
            conn.commit()

            preset_rows = [
                (
                    "Close Up Prime",
                    "Close Up",
                    "85mm portrait lens",
                    "Locked Camera",
                    "Intimate portrait framing",
                    "cinematic close-up, facial detail, shallow depth of field",
                    "warped face, distant framing",
                ),
                (
                    "Wide Dolly",
                    "Wide Shot",
                    "24mm anamorphic lens",
                    "Dolly In",
                    "Environmental wide framing",
                    "wide cinematic composition, environmental storytelling, controlled dolly move",
                    "flat staging, cropped subject",
                ),
                (
                    "Over Shoulder Dialogue",
                    "Over Shoulder",
                    "50mm spherical lens",
                    "Locked Camera",
                    "Dialogue coverage with screen direction",
                    "over-shoulder framing, conversational blocking, eyeline continuity",
                    "front-facing monologue framing",
                ),
            ]
            for row in preset_rows:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO camera_presets (
                        name, camera_language, lens_preset, movement_preset,
                        framing_notes, prompt_addition, negative_prompt_addition
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )

            for character in conn.execute("SELECT id, voice_id, expression_notes, voice_notes FROM characters").fetchall():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO character_performance_profiles (
                        character_id, voice_id, default_expression, speaking_style, notes
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        character["id"],
                        character["voice_id"],
                        "neutral",
                        "natural",
                        character["voice_notes"] or character["expression_notes"] or "",
                    ),
                )
            conn.commit()

    # ==========================================
    # PROJECTS
    # ==========================================
    def add_project(self, name, description=""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO projects (name, description) VALUES (?, ?)",
                    (name, description)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def delete_project(self, project_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()

    def get_project(self, project_id):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            return dict(row) if row else None

    def get_project_by_name(self, name):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def list_projects(self):
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    # ==========================================
    # EPISODES
    # ==========================================
    def add_episode(self, project_id, episode_number, title="", description=""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO episodes (project_id, episode_number, title, description)
                       VALUES (?, ?, ?, ?)""",
                    (project_id, episode_number, title, description)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Retrieve existing ID
                row = conn.execute(
                    "SELECT id FROM episodes WHERE project_id = ? AND episode_number = ?",
                    (project_id, episode_number)
                ).fetchone()
                return row["id"] if row else None

    def delete_episode(self, episode_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            conn.commit()

    def list_episodes(self, project_id):
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE project_id = ? ORDER BY episode_number ASC",
                (project_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_episode_by_details(self, project_id, episode_number):
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE project_id = ? AND episode_number = ?",
                (project_id, episode_number)
            ).fetchone()
            return dict(row) if row else None

    # ==========================================
    # SCENES
    # ==========================================
    def add_scene(self, episode_id, scene_number, title="", description="", character_id=None, location_id=None, shot_type=None, prompt="", video_prompt=""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO scenes (episode_id, scene_number, title, description, character_id, location_id, shot_type, prompt, video_prompt)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (episode_id, scene_number, title, description, character_id, location_id, shot_type, prompt, video_prompt)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Update existing scene
                conn.execute(
                    """UPDATE scenes 
                       SET title = ?, description = ?, character_id = ?, location_id = ?, shot_type = ?, prompt = ?, video_prompt = ?
                       WHERE episode_id = ? AND scene_number = ?""",
                    (title, description, character_id, location_id, shot_type, prompt, video_prompt, episode_id, scene_number)
                )
                conn.commit()
                row = conn.execute(
                    "SELECT id FROM scenes WHERE episode_id = ? AND scene_number = ?",
                    (episode_id, scene_number)
                ).fetchone()
                return row["id"] if row else None

    def delete_scene(self, scene_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM scenes WHERE id = ?", (scene_id,))
            conn.commit()

    def get_scene(self, scene_id):
        with self._get_connection() as conn:
            row = conn.execute(
                """SELECT s.*, e.episode_number, p.name as project_name, p.id as project_id
                   FROM scenes s
                   JOIN episodes e ON s.episode_id = e.id
                   JOIN projects p ON e.project_id = p.id
                   WHERE s.id = ?""",
                (scene_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_scenes(self, episode_id):
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT s.*, c.name as character_name, l.name as location_name 
                   FROM scenes s
                   LEFT JOIN characters c ON s.character_id = c.id
                   LEFT JOIN locations l ON s.location_id = l.id
                   WHERE s.episode_id = ? 
                   ORDER BY s.scene_number ASC""",
                (episode_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    # ==========================================
    # CHARACTERS
    # ==========================================
    def add_character(self, name, age, gender, description, notes, tags, folder_path,
                      biography="", personality="", prompt_template="", wardrobe_notes="",
                      expression_notes="", voice_notes="", dna_hair="", dna_eyes="",
                      dna_body_type="", dna_clothing="", dna_ethnicity="", dna_description=""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO characters (
                           name, age, gender, description, notes, tags, folder_path,
                           biography, personality, prompt_template, wardrobe_notes,
                           expression_notes, voice_notes, dna_hair, dna_eyes,
                           dna_body_type, dna_clothing, dna_ethnicity, dna_description
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (name, age, gender, description, notes, tags, folder_path,
                     biography, personality, prompt_template, wardrobe_notes,
                     expression_notes, voice_notes, dna_hair, dna_eyes,
                     dna_body_type, dna_clothing, dna_ethnicity, dna_description)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def update_character(self, char_id, name, age, gender, description, notes, tags,
                         biography="", personality="", prompt_template="", wardrobe_notes="",
                         expression_notes="", voice_notes="", dna_hair="", dna_eyes="",
                         dna_body_type="", dna_clothing="", dna_ethnicity="", dna_description=""):
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE characters 
                   SET name = ?, age = ?, gender = ?, description = ?, notes = ?, tags = ?,
                       biography = ?, personality = ?, prompt_template = ?, wardrobe_notes = ?,
                       expression_notes = ?, voice_notes = ?, dna_hair = ?, dna_eyes = ?,
                       dna_body_type = ?, dna_clothing = ?, dna_ethnicity = ?, dna_description = ?
                   WHERE id = ?""",
                (name, age, gender, description, notes, tags,
                 biography, personality, prompt_template, wardrobe_notes,
                 expression_notes, voice_notes, dna_hair, dna_eyes,
                 dna_body_type, dna_clothing, dna_ethnicity, dna_description, char_id)
            )
            conn.commit()

    # ==========================================
    # CHARACTER RELATIONSHIPS (Phase 2)
    # ==========================================
    def add_relationship(self, char_id, target_id, rel_type, description=""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO character_relationships (character_id, target_character_id, relationship_type, description)
                       VALUES (?, ?, ?, ?)""",
                    (char_id, target_id, rel_type, description)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Update existing
                conn.execute(
                    """UPDATE character_relationships
                       SET relationship_type = ?, description = ?
                       WHERE character_id = ? AND target_character_id = ?""",
                    (rel_type, description, char_id, target_id)
                )
                conn.commit()
                row = conn.execute(
                    "SELECT id FROM character_relationships WHERE character_id = ? AND target_character_id = ?",
                    (char_id, target_id)
                ).fetchone()
                return row["id"] if row else None

    def delete_relationship(self, rel_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM character_relationships WHERE id = ?", (rel_id,))
            conn.commit()

    def list_relationships(self, char_id):
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT r.*, c.name as target_character_name
                   FROM character_relationships r
                   JOIN characters c ON r.target_character_id = c.id
                   WHERE r.character_id = ?""",
                (char_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    # ==========================================
    # SEARCH & FILTERS (Phase 2)
    # ==========================================
    def search_characters(self, search_query=None, gender=None, tag=None, project_id=None, location_id=None):
        query = "SELECT DISTINCT c.* FROM characters c"
        params = []
        joins = []
        conditions = []
        
        if project_id:
            joins.append("JOIN scenes s ON s.character_id = c.id JOIN episodes e ON s.episode_id = e.id")
            conditions.append("e.project_id = ?")
            params.append(project_id)
            
        if location_id:
            if not project_id:
                joins.append("JOIN scenes s ON s.character_id = c.id")
            conditions.append("s.location_id = ?")
            params.append(location_id)
            
        if search_query:
            conditions.append("(c.name LIKE ? OR c.description LIKE ? OR c.notes LIKE ? OR c.biography LIKE ? OR c.personality LIKE ? OR c.tags LIKE ?)")
            q = f"%{search_query}%"
            params.extend([q, q, q, q, q, q])
            
        if gender and gender != "All":
            conditions.append("c.gender = ?")
            params.append(gender)
            
        if tag and tag != "All":
            conditions.append("c.tags LIKE ?")
            params.append(f"%{tag}%")
            
        if joins:
            query += " " + " ".join(joins)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY c.name ASC"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]


    def delete_character(self, char_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM characters WHERE id = ?", (char_id,))
            conn.commit()

    def get_character(self, char_id):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM characters WHERE id = ?", (char_id,)).fetchone()
            return dict(row) if row else None

    def get_character_by_name(self, name):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM characters WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def list_characters(self):
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM characters ORDER BY name ASC").fetchall()
            return [dict(row) for row in rows]

    # ==========================================
    # LOCATIONS
    # ==========================================
    def add_location(self, name, description, tags, notes, folder_path, prompt_template=""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO locations (name, description, tags, notes, folder_path, prompt_template)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (name, description, tags, notes, folder_path, prompt_template)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def update_location(self, loc_id, name, description, tags, notes, prompt_template=""):
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE locations 
                   SET name = ?, description = ?, tags = ?, notes = ?, prompt_template = ?
                   WHERE id = ?""",
                (name, description, tags, notes, prompt_template, loc_id)
            )
            conn.commit()

    # ==========================================
    # SUB-LOCATIONS (ROOMS) (Phase 3)
    # ==========================================
    def add_sub_location(self, location_id, name, description="", prompt_template="", folder_path=""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO sub_locations (location_id, name, description, prompt_template, folder_path)
                       VALUES (?, ?, ?, ?, ?)""",
                    (location_id, name, description, prompt_template, folder_path)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Update existing sub-location
                conn.execute(
                    """UPDATE sub_locations
                       SET description = ?, prompt_template = ?
                       WHERE location_id = ? AND name = ?""",
                    (description, prompt_template, location_id, name)
                )
                conn.commit()
                row = conn.execute(
                    "SELECT id FROM sub_locations WHERE location_id = ? AND name = ?",
                    (location_id, name)
                ).fetchone()
                return row["id"] if row else None

    def delete_sub_location(self, sub_loc_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sub_locations WHERE id = ?", (sub_loc_id,))
            conn.commit()

    def list_sub_locations(self, location_id):
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sub_locations WHERE location_id = ? ORDER BY name ASC",
                (location_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_sub_location(self, sub_loc_id):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sub_locations WHERE id = ?", (sub_loc_id,)).fetchone()
            return dict(row) if row else None

    # ==========================================
    # SEARCH LOCATIONS (Phase 3)
    # ==========================================
    def search_locations(self, search_query=None, tag=None, project_id=None, character_id=None):
        query = "SELECT DISTINCT l.* FROM locations l"
        params = []
        joins = []
        conditions = []
        
        if project_id:
            joins.append("JOIN scenes s ON s.location_id = l.id JOIN episodes e ON s.episode_id = e.id")
            conditions.append("e.project_id = ?")
            params.append(project_id)
            
        if character_id:
            if not project_id:
                joins.append("JOIN scenes s ON s.location_id = l.id")
            conditions.append("s.character_id = ?")
            params.append(character_id)
            
        if search_query:
            conditions.append("(l.name LIKE ? OR l.description LIKE ? OR l.notes LIKE ? OR l.tags LIKE ?)")
            q = f"%{search_query}%"
            params.extend([q, q, q, q])
            
        if tag and tag != "All":
            conditions.append("l.tags LIKE ?")
            params.append(f"%{tag}%")
            
        if joins:
            query += " " + " ".join(joins)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY l.name ASC"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ==========================================
    # TIMELINE EVENTS (Phase 3)
    # ==========================================
    def add_timeline_event(self, project_id, event_order, title, description="", event_date=""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO timeline_events (project_id, event_order, title, description, event_date)
                       VALUES (?, ?, ?, ?, ?)""",
                    (project_id, event_order, title, description, event_date)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Update existing event
                conn.execute(
                    """UPDATE timeline_events
                       SET title = ?, description = ?, event_date = ?
                       WHERE project_id = ? AND event_order = ?""",
                    (title, description, event_date, project_id, event_order)
                )
                conn.commit()
                row = conn.execute(
                    "SELECT id FROM timeline_events WHERE project_id = ? AND event_order = ?",
                    (project_id, event_order)
                ).fetchone()
                return row["id"] if row else None

    def delete_timeline_event(self, event_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM timeline_events WHERE id = ?", (event_id,))
            conn.commit()

    def list_timeline_events(self, project_id):
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM timeline_events WHERE project_id = ? ORDER BY event_order ASC",
                (project_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def update_timeline_event_order(self, event_id, new_order):
        with self._get_connection() as conn:
            conn.execute("UPDATE timeline_events SET event_order = ? WHERE id = ?", (new_order, event_id))
            conn.commit()

    # ==========================================
    # STORY BIBLE NOTES (Phase 3)
    # ==========================================
    def add_story_note(self, project_id, title, content, category):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO story_bible_notes (project_id, title, content, category)
                   VALUES (?, ?, ?, ?)""",
                (project_id, title, content, category)
            )
            conn.commit()
            return cursor.lastrowid

    def update_story_note(self, note_id, title, content, category):
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE story_bible_notes
                   SET title = ?, content = ?, category = ?
                   WHERE id = ?""",
                (title, content, category, note_id)
            )
            conn.commit()

    def delete_story_note(self, note_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM story_bible_notes WHERE id = ?", (note_id,))
            conn.commit()

    def list_story_notes(self, project_id, category=None):
        query = "SELECT * FROM story_bible_notes WHERE project_id = ?"
        params = [project_id]
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY created_at DESC"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]


    def delete_location(self, loc_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM locations WHERE id = ?", (loc_id,))
            conn.commit()

    def get_location(self, loc_id):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM locations WHERE id = ?", (loc_id,)).fetchone()
            return dict(row) if row else None

    def get_location_by_name(self, name):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM locations WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def list_locations(self):
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM locations ORDER BY name ASC").fetchall()
            return [dict(row) for row in rows]

    # ==========================================
    # PRESETS
    # ==========================================
    def add_preset(self, name, width, height, fps, frame_count, model, pipeline=None, camera_motion=None, notes=None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO presets (name, width, height, fps, frame_count, model, pipeline, camera_motion, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (name, width, height, fps, frame_count, model, pipeline, camera_motion, notes)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                conn.execute(
                    """UPDATE presets 
                       SET width = ?, height = ?, fps = ?, frame_count = ?, model = ?, pipeline = ?, camera_motion = ?, notes = ?
                       WHERE name = ?""",
                    (width, height, fps, frame_count, model, pipeline, camera_motion, notes, name)
                )
                conn.commit()
                return True

    def delete_preset(self, name):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM presets WHERE name = ?", (name,))
            conn.commit()

    def get_preset(self, name):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM presets WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def list_presets(self):
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM presets ORDER BY name ASC").fetchall()
            return [dict(row) for row in rows]

    # ==========================================
    # SETTINGS & SESSION MEMORY
    # ==========================================
    def set_setting(self, key, value):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, str(value))
            )
            conn.commit()

    def get_setting(self, key, default=None):
        with self._get_connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def get_all_settings(self):
        try:
            self.detect_and_update_engine_status()
        except Exception:
            pass
        with self._get_connection() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
            return {row["key"]: row["value"] for row in rows}

    # ==========================================
    # GENERATION HISTORY
    # ==========================================
    def add_history(self, scene_id, asset_type, file_path, prompt, seed, status="completed"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO generation_history (scene_id, asset_type, file_path, prompt, seed, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (scene_id, asset_type, file_path, prompt, seed, status)
            )
            conn.commit()
            return cursor.lastrowid

    def list_history(self, scene_id=None, limit=100):
        query = "SELECT h.*, s.scene_number, e.episode_number, p.name as project_name FROM generation_history h LEFT JOIN scenes s ON h.scene_id = s.id LEFT JOIN episodes e ON s.episode_id = e.id LEFT JOIN projects p ON e.project_id = p.id"
        params = []
        if scene_id:
            query += " WHERE h.scene_id = ?"
            params.append(scene_id)
        query += " ORDER BY h.created_at DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def delete_history(self, history_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM generation_history WHERE id = ?", (history_id,))
            conn.commit()

    # ==========================================
    # PHASE 4: MODEL CONFIG SETTINGS
    # ==========================================
    def get_model_config(self):
        """Return a dict of all four Flux model config settings."""
        return {
            "flux_dev_path": self.get_setting("flux_dev_path", ""),
            "flux_kontext_path": self.get_setting("flux_kontext_path", ""),
            "flux_device": self.get_setting("flux_device", "cuda"),
            "flux_dtype": self.get_setting("flux_dtype", "bfloat16"),
        }

    def save_model_config(self, flux_dev_path="", flux_kontext_path="",
                          flux_device="cuda", flux_dtype="bfloat16"):
        """Persist all four Flux model config settings atomically."""
        self.set_setting("flux_dev_path", flux_dev_path)
        self.set_setting("flux_kontext_path", flux_kontext_path)
        self.set_setting("flux_device", flux_device)
        self.set_setting("flux_dtype", flux_dtype)

    # ==========================================
    # PHASE 4: GENERATED IMAGES CRUD
    # ==========================================
    def add_generated_image(
        self,
        file_path,
        model,
        prompt="",
        negative_prompt="",
        seed=None,
        width=None,
        height=None,
        steps=None,
        guidance_scale=None,
        thumbnail_path=None,
        project_id=None,
        scene_id=None,
        character_id=None,
        location_id=None,
    ):
        """Insert a new generated image record and return its id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO generated_images (
                       project_id, scene_id, character_id, location_id,
                       model, prompt, negative_prompt, seed,
                       width, height, steps, guidance_scale,
                       file_path, thumbnail_path, status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (
                    project_id, scene_id, character_id, location_id,
                    model, prompt, negative_prompt, seed,
                    width, height, steps, guidance_scale,
                    file_path, thumbnail_path,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_generated_image(self, image_id):
        """Return a single generated image record as a dict."""
        with self._get_connection() as conn:
            row = conn.execute(
                """SELECT gi.*,
                          p.name  AS project_name,
                          c.name  AS character_name,
                          l.name  AS location_name
                   FROM generated_images gi
                   LEFT JOIN projects   p ON gi.project_id   = p.id
                   LEFT JOIN characters c ON gi.character_id = c.id
                   LEFT JOIN locations  l ON gi.location_id  = l.id
                   WHERE gi.id = ?""",
                (image_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_generated_images(
        self,
        project_id=None,
        character_id=None,
        location_id=None,
        status=None,
        model=None,
        limit=200,
    ):
        """Return filtered list of generated image records."""
        query = """SELECT gi.*,
                          p.name  AS project_name,
                          c.name  AS character_name,
                          l.name  AS location_name
                   FROM generated_images gi
                   LEFT JOIN projects   p ON gi.project_id   = p.id
                   LEFT JOIN characters c ON gi.character_id = c.id
                   LEFT JOIN locations  l ON gi.location_id  = l.id
                   WHERE 1=1"""
        params = []
        if project_id is not None:
            query += " AND gi.project_id = ?"
            params.append(project_id)
        if character_id is not None:
            query += " AND gi.character_id = ?"
            params.append(character_id)
        if location_id is not None:
            query += " AND gi.location_id = ?"
            params.append(location_id)
        if status:
            query += " AND gi.status = ?"
            params.append(status)
        if model:
            query += " AND gi.model = ?"
            params.append(model)
        query += " ORDER BY gi.created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def update_image_status(self, image_id, status):
        """Approve ('approved') or reject ('rejected') a generated image."""
        approved_at = datetime.utcnow().isoformat() if status == "approved" else None
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE generated_images SET status = ?, approved_at = ? WHERE id = ?",
                (status, approved_at, image_id),
            )
            conn.commit()

    def delete_generated_image(self, image_id):
        """Remove a generated image record (does NOT delete the file on disk)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM generated_images WHERE id = ?", (image_id,))
            conn.commit()

    # ==========================================
    # PHASE 5: LTX MODEL CONFIG SETTINGS
    # ==========================================
    def get_ltx_config(self) -> dict:
        """Return all six LTX model config settings."""
        return {
            "ltx_model_path":    self.get_setting("ltx_model_path", ""),
            "ltx_gemma_path":    self.get_setting("ltx_gemma_path", ""),
            "ltx_upscaler_path": self.get_setting("ltx_upscaler_path", ""),
            "ltx_lora_dir":      self.get_setting("ltx_lora_dir", ""),
            "ltx_device":        self.get_setting("ltx_device", "cuda"),
            "ltx_dtype":         self.get_setting("ltx_dtype", "bfloat16"),
        }

    def save_ltx_config(
        self,
        ltx_model_path: str = "",
        ltx_gemma_path: str = "",
        ltx_upscaler_path: str = "",
        ltx_lora_dir: str = "",
        ltx_device: str = "cuda",
        ltx_dtype: str = "bfloat16",
    ):
        """Persist all LTX model config settings atomically."""
        self.set_setting("ltx_model_path",    ltx_model_path)
        self.set_setting("ltx_gemma_path",    ltx_gemma_path)
        self.set_setting("ltx_upscaler_path", ltx_upscaler_path)
        self.set_setting("ltx_lora_dir",      ltx_lora_dir)
        self.set_setting("ltx_device",        ltx_device)
        self.set_setting("ltx_dtype",         ltx_dtype)

    # ==========================================
    # PHASE 6: WAN & HUNYUAN CONFIG SETTINGS
    # ==========================================
    def get_wan_config(self) -> dict:
        return {
            "wan_model_path": self.get_setting("wan_model_path", ""),
            "wan_device":     self.get_setting("wan_device", "cuda"),
            "wan_dtype":      self.get_setting("wan_dtype", "bfloat16"),
        }

    def save_wan_config(self, wan_model_path: str = "", wan_device: str = "cuda", wan_dtype: str = "bfloat16"):
        self.set_setting("wan_model_path", wan_model_path)
        self.set_setting("wan_device", wan_device)
        self.set_setting("wan_dtype", wan_dtype)

    def get_hunyuan_config(self) -> dict:
        return {
            "hunyuan_model_path": self.get_setting("hunyuan_model_path", ""),
            "hunyuan_device":     self.get_setting("hunyuan_device", "cuda"),
            "hunyuan_dtype":      self.get_setting("hunyuan_dtype", "bfloat16"),
        }

    def save_hunyuan_config(self, hunyuan_model_path: str = "", hunyuan_device: str = "cuda", hunyuan_dtype: str = "bfloat16"):
        self.set_setting("hunyuan_model_path", hunyuan_model_path)
        self.set_setting("hunyuan_device", hunyuan_device)
        self.set_setting("hunyuan_dtype", hunyuan_dtype)

    # ==========================================
    # PHASE 5/6: GENERATED VIDEOS CRUD
    # ==========================================
    def add_generated_video(
        self,
        file_path: str,
        pipeline: str,
        model_family: str = "LTX-Video",
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = -1,
        width: int = None,
        height: int = None,
        fps: int = 24,
        num_frames: int = 97,
        steps: int = 20,
        guidance_scale: float = 3.5,
        lora_path: str = "",
        lora_weight: float = 1.0,
        reference_image_path: str = "",
        thumbnail_path: str = None,
        duration_seconds: float = None,
        preset: str = "",
        project_id: int = None,
        episode_id: int = None,
        scene_id: int = None,
        shot_id: int = None,
        dialogue_line_id: int = None,
        character_id: int = None,
        location_id: int = None,
        model_name: str = "",
        lora_name: str = "",
        references_json: str = "[]",
        performance_json: str = "{}",
        camera_preset: str = "",
        movement_preset: str = "",
        lens_preset: str = "",
    ) -> int:
        """Insert a new generated video record and return its id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            columns = [
                "project_id", "episode_id", "scene_id", "shot_id", "dialogue_line_id", "character_id", "location_id",
                "model_family", "pipeline", "preset", "prompt", "negative_prompt", "seed",
                "width", "height", "fps", "num_frames", "steps", "guidance_scale",
                "lora_path", "lora_name", "lora_weight", "reference_image_path", "references_json",
                "model_name", "performance_json", "camera_preset", "movement_preset", "lens_preset",
                "file_path", "thumbnail_path", "duration_seconds", "status",
            ]
            values = [
                project_id, episode_id, scene_id, shot_id, dialogue_line_id, character_id, location_id,
                model_family, pipeline, preset, prompt, negative_prompt, seed,
                width, height, fps, num_frames, steps, guidance_scale,
                lora_path or "", lora_name or "", lora_weight, reference_image_path or "", references_json or "[]",
                model_name or "", performance_json or "{}", camera_preset or "", movement_preset or "", lens_preset or "",
                file_path, thumbnail_path, duration_seconds, "pending",
            ]
            placeholders = ", ".join(["?"] * len(values))
            cursor.execute(
                f"INSERT INTO generated_videos ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
            return cursor.lastrowid

    def get_generated_video(self, video_id: int) -> dict | None:
        """Return a single generated video record with joined names."""
        with self._get_connection() as conn:
            row = conn.execute(
                """SELECT gv.*,
                          p.name  AS project_name,
                          c.name  AS character_name,
                          l.name  AS location_name
                   FROM generated_videos gv
                   LEFT JOIN projects   p ON gv.project_id   = p.id
                   LEFT JOIN characters c ON gv.character_id = c.id
                   LEFT JOIN locations  l ON gv.location_id  = l.id
                   WHERE gv.id = ?""",
                (video_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_generated_videos(
        self,
        project_id: int = None,
        character_id: int = None,
        location_id: int = None,
        status: str = None,
        pipeline: str = None,
        model_family: str = None,
        limit: int = 200,
    ) -> list:
        """Return a filtered list of generated video records."""
        query = """SELECT gv.*,
                          p.name  AS project_name,
                          c.name  AS character_name,
                          l.name  AS location_name
                   FROM generated_videos gv
                   LEFT JOIN projects   p ON gv.project_id   = p.id
                   LEFT JOIN characters c ON gv.character_id = c.id
                   LEFT JOIN locations  l ON gv.location_id  = l.id
                   WHERE 1=1"""
        params = []
        if project_id is not None:
            query += " AND gv.project_id = ?"
            params.append(project_id)
        if character_id is not None:
            query += " AND gv.character_id = ?"
            params.append(character_id)
        if location_id is not None:
            query += " AND gv.location_id = ?"
            params.append(location_id)
        if status:
            query += " AND gv.status = ?"
            params.append(status)
        if pipeline:
            query += " AND gv.pipeline = ?"
            params.append(pipeline)
        if model_family:
            query += " AND gv.model_family = ?"
            params.append(model_family)
        query += " ORDER BY gv.created_at DESC LIMIT ?"
        params.append(limit)
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def update_video_status(self, video_id: int, status: str):
        """Approve ('approved') or reject ('rejected') a generated video."""
        approved_at = datetime.utcnow().isoformat() if status == "approved" else None
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE generated_videos SET status = ?, approved_at = ? WHERE id = ?",
                (status, approved_at, video_id),
            )
            conn.commit()

    def delete_generated_video(self, video_id: int):
        """Remove a generated video record (does NOT delete the file on disk)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM generated_videos WHERE id = ?", (video_id,))
            conn.commit()

    # ==========================================
    # PHASE 7: SHOT TEMPLATES
    # ==========================================
    def list_shot_templates(self) -> list:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM shot_templates ORDER BY id ASC").fetchall()
            return [dict(row) for row in rows]

    def get_shot_template_by_name(self, name: str) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM shot_templates WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    # ==========================================
    # PHASE 7: RENDER QUEUE
    # ==========================================
    def add_to_render_queue(
        self,
        project_id: int = None,
        episode_id: int = None,
        scene_id: int = None,
        shot_id: int = None,
        dialogue_line_id: int = None,
        character_id: int = None,
        location_id: int = None,
        shot_type_id: int = None,
        model_family: str = "LTX-Video",
        pipeline: str = "distilled",
        preset: str = "draft",
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = -1,
        width: int = 768,
        height: int = 512,
        fps: int = 24,
        num_frames: int = 65,
        steps: int = 20,
        guidance_scale: float = 3.0,
        lora_path: str = "",
        lora_weight: float = 1.0,
        reference_image_path: str = "",
        model_name: str = "",
        lora_name: str = "",
        references_json: str = "[]",
        performance_json: str = "{}",
        camera_preset: str = "",
        movement_preset: str = "",
        lens_preset: str = "",
        continuity_notes: str = "",
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            columns = [
                "project_id", "episode_id", "scene_id", "shot_id", "dialogue_line_id", "character_id", "location_id", "shot_type_id",
                "model_family", "pipeline", "preset", "prompt", "negative_prompt", "seed",
                "width", "height", "fps", "num_frames", "steps", "guidance_scale",
                "lora_path", "lora_name", "lora_weight", "reference_image_path", "references_json",
                "model_name", "performance_json", "camera_preset", "movement_preset", "lens_preset",
                "continuity_notes", "status",
            ]
            values = [
                project_id, episode_id, scene_id, shot_id, dialogue_line_id, character_id, location_id, shot_type_id,
                model_family, pipeline, preset, prompt, negative_prompt, seed,
                width, height, fps, num_frames, steps, guidance_scale,
                lora_path, lora_name, lora_weight, reference_image_path, references_json or "[]",
                model_name, performance_json or "{}", camera_preset, movement_preset, lens_preset,
                continuity_notes, "queued",
            ]
            placeholders = ", ".join(["?"] * len(values))
            cursor.execute(
                f"INSERT INTO render_queue ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
            return cursor.lastrowid

    def list_render_jobs(self, status: str = None, limit: int = 200) -> list:
        query = """SELECT rq.*, 
                          p.name as project_name, 
                          c.name as character_name, 
                          l.name as location_name,
                          st.name as shot_type_name
                   FROM render_queue rq
                   LEFT JOIN projects p ON rq.project_id = p.id
                   LEFT JOIN characters c ON rq.character_id = c.id
                   LEFT JOIN locations l ON rq.location_id = l.id
                   LEFT JOIN shot_templates st ON rq.shot_type_id = st.id
                   WHERE 1=1"""
        params = []
        if status:
            query += " AND rq.status = ?"
            params.append(status)
        query += " ORDER BY rq.created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_render_job(self, job_id: int) -> dict | None:
        query = """SELECT rq.*, 
                          p.name as project_name, 
                          c.name as character_name, 
                          l.name as location_name,
                          st.name as shot_type_name
                   FROM render_queue rq
                   LEFT JOIN projects p ON rq.project_id = p.id
                   LEFT JOIN characters c ON rq.character_id = c.id
                   LEFT JOIN locations l ON rq.location_id = l.id
                   LEFT JOIN shot_templates st ON rq.shot_type_id = st.id
                   WHERE rq.id = ?"""
        with self._get_connection() as conn:
            row = conn.execute(query, (job_id,)).fetchone()
            return dict(row) if row else None

    def update_render_job_status(self, job_id: int, status: str, error_message: str = None, video_id: int = None):
        with self._get_connection() as conn:
            if status == "running":
                conn.execute("UPDATE render_queue SET status = ?, started_at = CURRENT_TIMESTAMP WHERE id = ?", (status, job_id))
            elif status in ("completed", "failed", "cancelled"):
                conn.execute(
                    "UPDATE render_queue SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?, video_id = ? WHERE id = ?",
                    (status, error_message, video_id, job_id)
                )
            else:
                conn.execute("UPDATE render_queue SET status = ? WHERE id = ?", (status, job_id))
            conn.commit()

    def get_next_queued_job(self) -> dict | None:
        """Atomically find the oldest queued job and mark it running, or return None."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute("SELECT id FROM render_queue WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1").fetchone()
            if row:
                job_id = row["id"]
                cursor.execute("UPDATE render_queue SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
                conn.commit()
                return self.get_render_job(job_id)
            return None

    # ==========================================
    # PHASE 8: VOICE READY
    # ==========================================
    def add_dialogue_line(
        self,
        scene_id: int,
        character_id: int,
        sequence_order: int,
        text: str,
        translation_text: str = "",
        shot_id: int = None,
        emotion: str = "neutral",
        intensity: int = 5,
        expression: str = "neutral",
        speaking_style: str = "natural",
        performance_notes: str = "",
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO dialogue_lines (
                    scene_id, character_id, sequence_order, text, translation_text, shot_id,
                    emotion, intensity, expression, speaking_style, performance_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    character_id,
                    sequence_order,
                    text,
                    translation_text,
                    shot_id,
                    emotion,
                    intensity,
                    expression,
                    speaking_style,
                    performance_notes,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def list_dialogue_lines(self, scene_id: int) -> list:
        query = """
            SELECT d.*, c.name as character_name
            FROM dialogue_lines d
            JOIN characters c ON d.character_id = c.id
            WHERE d.scene_id = ?
            ORDER BY d.sequence_order ASC
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, (scene_id,)).fetchall()
            return [dict(row) for row in rows]

    def add_audio_asset(
        self,
        dialogue_line_id: int,
        file_path: str,
        duration: float = 0.0,
        project_id: int = None,
        episode_id: int = None,
        scene_id: int = None,
        shot_id: int = None,
        character_id: int = None,
        voice_id: str = "",
        emotion: str = "",
        intensity: int = None,
        speaking_style: str = "",
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audio_assets (
                    dialogue_line_id, file_path, duration, project_id, episode_id, scene_id,
                    shot_id, character_id, voice_id, emotion, intensity, speaking_style
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dialogue_line_id,
                    file_path,
                    duration,
                    project_id,
                    episode_id,
                    scene_id,
                    shot_id,
                    character_id,
                    voice_id,
                    emotion,
                    intensity,
                    speaking_style,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    # ==========================================
    # AUDIO QUEUE
    # ==========================================
    def add_to_audio_queue(
        self,
        project_id: int,
        episode_id: int,
        scene_id: int,
        character_id: int,
        dialogue_text: str,
        voice_id: str,
        speed: float = 1.0,
        shot_id: int = None,
        dialogue_line_id: int = None,
        emotion: str = "neutral",
        intensity: int = 5,
        expression: str = "neutral",
        speaking_style: str = "natural",
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audio_queue (
                    project_id, episode_id, scene_id, shot_id, character_id, dialogue_line_id,
                    dialogue_text, voice_id, speed, emotion, intensity, expression, speaking_style
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    episode_id,
                    scene_id,
                    shot_id,
                    character_id,
                    dialogue_line_id,
                    dialogue_text,
                    voice_id,
                    speed,
                    emotion,
                    intensity,
                    expression,
                    speaking_style,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_next_queued_audio_job(self) -> dict | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute("SELECT id FROM audio_queue WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1").fetchone()
            if row:
                job_id = row["id"]
                cursor.execute("UPDATE audio_queue SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
                conn.commit()
                return self.get_audio_job(job_id)
            return None

    def get_audio_job(self, job_id: int) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM audio_queue WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def update_audio_job_status(self, job_id: int, status: str, file_path: str = None, error_message: str = None):
        with self._get_connection() as conn:
            if status == "completed":
                conn.execute("UPDATE audio_queue SET status = ?, completed_at = CURRENT_TIMESTAMP, file_path = ? WHERE id = ?", (status, file_path, job_id))
            elif status == "failed":
                conn.execute("UPDATE audio_queue SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ? WHERE id = ?", (status, error_message, job_id))
            else:
                conn.execute("UPDATE audio_queue SET status = ? WHERE id = ?", (status, job_id))
            conn.commit()

    def list_audio_jobs(self, limit: int = 50) -> list:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT a.*, c.name as character_name
                FROM audio_queue a
                LEFT JOIN characters c ON a.character_id = c.id
                ORDER BY a.created_at DESC LIMIT ?
                """,
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    # ==========================================
    # PHASE 10+: PRODUCTION INTELLIGENCE LAYER
    # ==========================================
    def add_shot(
        self,
        scene_id: int,
        shot_number: int,
        shot_type: str = "",
        description: str = "",
        camera_direction: str = "",
        camera_motion: str = "",
        character_focus: int = None,
        duration: float = 3.0,
        title: str = "",
        goal: str = "",
        camera_language: str = "",
        lens_preset: str = "",
        movement_preset: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        references_json: str = "[]",
        status: str = "pending",
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM shots WHERE scene_id = ? AND shot_number = ?", (scene_id, shot_number))
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    """
                    UPDATE shots
                    SET shot_type = ?, description = ?, camera_direction = ?, camera_motion = ?, character_focus = ?,
                        duration = ?, title = ?, goal = ?, camera_language = ?, lens_preset = ?, movement_preset = ?,
                        prompt = ?, negative_prompt = ?, references_json = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        shot_type,
                        description,
                        camera_direction,
                        camera_motion,
                        character_focus,
                        duration,
                        title,
                        goal,
                        camera_language,
                        lens_preset,
                        movement_preset,
                        prompt,
                        negative_prompt,
                        references_json or "[]",
                        status,
                        existing["id"],
                    ),
                )
                conn.commit()
                return existing["id"]
            cursor.execute(
                """
                INSERT INTO shots (
                    scene_id, shot_number, shot_type, description, camera_direction, camera_motion,
                    character_focus, duration, title, goal, camera_language, lens_preset,
                    movement_preset, prompt, negative_prompt, references_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    shot_number,
                    shot_type,
                    description,
                    camera_direction,
                    camera_motion,
                    character_focus,
                    duration,
                    title,
                    goal,
                    camera_language,
                    lens_preset,
                    movement_preset,
                    prompt,
                    negative_prompt,
                    references_json or "[]",
                    status,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def list_shots(self, scene_id: int) -> list:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT s.*, c.name AS character_name
                FROM shots s
                LEFT JOIN characters c ON s.character_focus = c.id
                WHERE s.scene_id = ?
                ORDER BY s.shot_number ASC
                """,
                (scene_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_shot(self, shot_id: int) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT s.*, c.name AS character_name
                FROM shots s
                LEFT JOIN characters c ON s.character_focus = c.id
                WHERE s.id = ?
                """,
                (shot_id,),
            ).fetchone()
            return dict(row) if row else None

    def delete_shot(self, shot_id: int):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM shots WHERE id = ?", (shot_id,))
            conn.commit()

    def set_continuity_state(self, scene_id: int, location_id: int = None, wardrobe_tag: str = "", weather: str = "", time_of_day: str = "", lighting: str = "", props: str = "", character_state: str = "", scene_state: str = ""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE continuity_state SET location_id=?, wardrobe_tag=?, weather=?, time_of_day=?, lighting=?, props=?, character_state=?, scene_state=? WHERE scene_id=?""",
                (location_id, wardrobe_tag, weather, time_of_day, lighting, props, character_state, scene_state, scene_id)
            )
            if cursor.rowcount == 0:
                cursor.execute(
                    """INSERT INTO continuity_state (scene_id, location_id, wardrobe_tag, weather, time_of_day, lighting, props, character_state, scene_state)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (scene_id, location_id, wardrobe_tag, weather, time_of_day, lighting, props, character_state, scene_state)
                )
            conn.commit()

    def get_continuity_state(self, scene_id: int) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM continuity_state WHERE scene_id = ?", (scene_id,)).fetchone()
            return dict(row) if row else None

    def get_character_memory(self, character_id: int) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM character_memory WHERE character_id = ?", (character_id,)).fetchone()
            return dict(row) if row else None

    def upsert_character_memory(
        self,
        character_id: int,
        default_expression: str = "",
        visual_style: str = "",
        reference_strength: float = 1.0,
        best_references: list | None = None,
        approved_assets: list | None = None,
        last_known_wardrobe: str = "",
        last_known_location_id: int = None,
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM character_memory WHERE character_id = ?", (character_id,))
            existing = cursor.fetchone()
            values = (
                default_expression,
                visual_style,
                reference_strength,
                json.dumps(best_references or []),
                json.dumps(approved_assets or []),
                last_known_wardrobe,
                last_known_location_id,
                character_id,
            )
            if existing:
                cursor.execute(
                    """
                    UPDATE character_memory
                    SET default_expression = ?, visual_style = ?, reference_strength = ?,
                        best_references_json = ?, approved_assets_json = ?, last_known_wardrobe = ?,
                        last_known_location_id = ?
                    WHERE character_id = ?
                    """,
                    values,
                )
                conn.commit()
                return existing["id"]
            cursor.execute(
                """
                INSERT INTO character_memory (
                    character_id, default_expression, visual_style, reference_strength,
                    best_references_json, approved_assets_json, last_known_wardrobe, last_known_location_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character_id,
                    default_expression,
                    visual_style,
                    reference_strength,
                    json.dumps(best_references or []),
                    json.dumps(approved_assets or []),
                    last_known_wardrobe,
                    last_known_location_id,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def add_approved_asset(self, character_id: int, asset_type: str, file_path: str, is_primary_reference: bool = False, wardrobe_tag: str = ""):
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO approved_character_assets (character_id, asset_type, file_path, is_primary_reference, wardrobe_tag)
                   VALUES (?, ?, ?, ?, ?)""",
                (character_id, asset_type, file_path, 1 if is_primary_reference else 0, wardrobe_tag)
            )
            conn.commit()

    def list_approved_assets(self, character_id: int = None) -> list:
        query = "SELECT * FROM approved_character_assets WHERE 1=1"
        params = []
        if character_id is not None:
            query += " AND character_id = ?"
            params.append(character_id)
        query += " ORDER BY is_primary_reference DESC, created_at DESC"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def add_reference_graph_edge(self, target_video_id: int, source_type: str, source_path: str):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO asset_reference_graph (target_video_id, source_type, source_path) VALUES (?, ?, ?)",
                (target_video_id, source_type, source_path),
            )
            conn.commit()

    def list_reference_graph(self, target_video_id: int = None) -> list:
        query = "SELECT * FROM asset_reference_graph"
        params = []
        if target_video_id is not None:
            query += " WHERE target_video_id = ?"
            params.append(target_video_id)
        query += " ORDER BY created_at DESC"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def upsert_character_performance_profile(
        self,
        character_id: int,
        voice_id: str = "",
        default_emotion: str = "neutral",
        default_intensity: int = 5,
        default_expression: str = "neutral",
        speaking_style: str = "natural",
        notes: str = "",
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM character_performance_profiles WHERE character_id = ?", (character_id,))
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    """
                    UPDATE character_performance_profiles
                    SET voice_id = ?, default_emotion = ?, default_intensity = ?, default_expression = ?,
                        speaking_style = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE character_id = ?
                    """,
                    (voice_id, default_emotion, default_intensity, default_expression, speaking_style, notes, character_id),
                )
                conn.commit()
                return existing["id"]
            cursor.execute(
                """
                INSERT INTO character_performance_profiles (
                    character_id, voice_id, default_emotion, default_intensity,
                    default_expression, speaking_style, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (character_id, voice_id, default_emotion, default_intensity, default_expression, speaking_style, notes),
            )
            conn.commit()
            return cursor.lastrowid

    def get_character_performance_profile(self, character_id: int) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM character_performance_profiles WHERE character_id = ?", (character_id,)).fetchone()
            return dict(row) if row else None

    def save_consistency_report(
        self,
        project_id: int = None,
        episode_id: int = None,
        scene_id: int = None,
        shot_id: int = None,
        character_id: int = None,
        face_drift: float = 0.0,
        hair_drift: float = 0.0,
        age_drift: float = 0.0,
        wardrobe_drift: float = 0.0,
        location_drift: float = 0.0,
        reference_strength: float = 1.0,
        consistency_score: float = 1.0,
        warning_level: str = "ok",
        warnings: list | None = None,
        suggestions: list | None = None,
        override_enabled: bool = False,
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO consistency_reports (
                    project_id, episode_id, scene_id, shot_id, character_id,
                    face_drift, hair_drift, age_drift, wardrobe_drift, location_drift,
                    reference_strength, consistency_score, warning_level, warnings_json,
                    suggestions_json, override_enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    episode_id,
                    scene_id,
                    shot_id,
                    character_id,
                    face_drift,
                    hair_drift,
                    age_drift,
                    wardrobe_drift,
                    location_drift,
                    reference_strength,
                    consistency_score,
                    warning_level,
                    json.dumps(warnings or []),
                    json.dumps(suggestions or []),
                    1 if override_enabled else 0,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_latest_consistency_report(self, shot_id: int = None, character_id: int = None, scene_id: int = None) -> dict | None:
        query = "SELECT * FROM consistency_reports WHERE 1=1"
        params = []
        if shot_id is not None:
            query += " AND shot_id = ?"
            params.append(shot_id)
        if character_id is not None:
            query += " AND character_id = ?"
            params.append(character_id)
        if scene_id is not None:
            query += " AND scene_id = ?"
            params.append(scene_id)
        query += " ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else None

    def list_camera_presets(self) -> list:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM camera_presets ORDER BY name ASC").fetchall()
            return [dict(row) for row in rows]

    def get_camera_preset(self, name: str) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM camera_presets WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def add_to_lip_sync_queue(
        self,
        project_id: int = None,
        episode_id: int = None,
        scene_id: int = None,
        shot_id: int = None,
        character_id: int = None,
        dialogue_line_id: int = None,
        audio_asset_id: int = None,
        source_video_id: int = None,
        source_audio_path: str = "",
        source_video_path: str = "",
        engine: str = "Wav2Lip",
        engine_config: dict | None = None,
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO lip_sync_queue (
                    project_id, episode_id, scene_id, shot_id, character_id, dialogue_line_id,
                    audio_asset_id, source_video_id, source_audio_path, source_video_path,
                    engine, engine_config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    episode_id,
                    scene_id,
                    shot_id,
                    character_id,
                    dialogue_line_id,
                    audio_asset_id,
                    source_video_id,
                    source_audio_path,
                    source_video_path,
                    engine,
                    json.dumps(engine_config or {}),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_next_queued_lipsync_job(self) -> dict | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute("SELECT id FROM lip_sync_queue WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1").fetchone()
            if row:
                job_id = row["id"]
                cursor.execute("UPDATE lip_sync_queue SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
                conn.commit()
                return self.get_lipsync_job(job_id)
            return None

    def get_lipsync_job(self, job_id: int) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM lip_sync_queue WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def update_lipsync_job_status(self, job_id: int, status: str, output_video_path: str = None, error_message: str = None):
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE lip_sync_queue
                SET status = ?, completed_at = CASE WHEN ? IN ('completed', 'failed') THEN CURRENT_TIMESTAMP ELSE completed_at END,
                    output_video_path = COALESCE(?, output_video_path), error_message = ?
                WHERE id = ?
                """,
                (status, status, output_video_path, error_message, job_id),
            )
            conn.commit()

    def add_lipsync_result(self, queue_job_id: int, engine: str, input_audio_path: str, input_video_path: str, output_video_path: str, status: str = "completed", metadata: dict | None = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO lipsync_jobs (
                    queue_job_id, engine, input_audio_path, input_video_path, output_video_path, status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (queue_job_id, engine, input_audio_path, input_video_path, output_video_path, status, json.dumps(metadata or {})),
            )
            conn.commit()
            return cursor.lastrowid

    def list_lipsync_jobs(self, limit: int = 50) -> list:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM lip_sync_queue ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(row) for row in rows]

    def upsert_episode_assembly(self, project_id: int, episode_id: int, name: str = "Latest Cut", status: str = "draft", settings: dict | None = None, preview: dict | None = None, export_path: str = "") -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM episode_assemblies WHERE project_id = ? AND episode_id = ? AND name = ?", (project_id, episode_id, name))
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    """
                    UPDATE episode_assemblies
                    SET status = ?, settings_json = ?, preview_json = ?, export_path = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (status, json.dumps(settings or {}), json.dumps(preview or {}), export_path, existing["id"]),
                )
                conn.commit()
                return existing["id"]
            cursor.execute(
                """
                INSERT INTO episode_assemblies (project_id, episode_id, name, status, settings_json, preview_json, export_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (project_id, episode_id, name, status, json.dumps(settings or {}), json.dumps(preview or {}), export_path),
            )
            conn.commit()
            return cursor.lastrowid

    def replace_episode_assembly_items(self, assembly_id: int, items: list[dict]):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM episode_assembly_items WHERE assembly_id = ?", (assembly_id,))
            for item in items:
                conn.execute(
                    """
                    INSERT INTO episode_assembly_items (
                        assembly_id, item_type, scene_id, shot_id, source_video_id, source_audio_asset_id,
                        subtitle_text, music_path, sequence_order, duration_seconds, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assembly_id,
                        item.get("item_type", "shot"),
                        item.get("scene_id"),
                        item.get("shot_id"),
                        item.get("source_video_id"),
                        item.get("source_audio_asset_id"),
                        item.get("subtitle_text"),
                        item.get("music_path"),
                        item.get("sequence_order", 1),
                        item.get("duration_seconds", 0.0),
                        json.dumps(item.get("metadata", {})),
                    ),
                )
            conn.commit()

    def get_episode_assembly(self, assembly_id: int) -> dict | None:
        with self._get_connection() as conn:
            assembly = conn.execute("SELECT * FROM episode_assemblies WHERE id = ?", (assembly_id,)).fetchone()
            if not assembly:
                return None
            items = conn.execute("SELECT * FROM episode_assembly_items WHERE assembly_id = ? ORDER BY sequence_order ASC, id ASC", (assembly_id,)).fetchall()
            payload = dict(assembly)
            payload["items"] = [dict(row) for row in items]
            return payload

    def list_episode_assemblies(self, project_id: int = None, episode_id: int = None) -> list:
        query = "SELECT * FROM episode_assemblies WHERE 1=1"
        params = []
        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)
        if episode_id is not None:
            query += " AND episode_id = ?"
            params.append(episode_id)
        query += " ORDER BY updated_at DESC, id DESC"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
