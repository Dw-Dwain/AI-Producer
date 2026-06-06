-- SQLite Schema for Phase 1 AI Drama Production Studio

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    episode_number INTEGER NOT NULL,
    title TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
    UNIQUE(project_id, episode_number)
);

CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL,
    scene_number INTEGER NOT NULL,
    title TEXT,
    description TEXT,
    character_id INTEGER,
    location_id INTEGER,
    shot_type TEXT,
    prompt TEXT,
    video_prompt TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL,
    FOREIGN KEY(location_id) REFERENCES locations(id) ON DELETE SET NULL,
    UNIQUE(episode_id, scene_number)
);

CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    age INTEGER,
    gender TEXT,
    description TEXT,
    notes TEXT,
    tags TEXT,
    folder_path TEXT NOT NULL,
    biography TEXT,
    personality TEXT,
    prompt_template TEXT,
    wardrobe_notes TEXT,
    expression_notes TEXT,
    voice_notes TEXT,
    dna_hair TEXT,
    dna_eyes TEXT,
    dna_body_type TEXT,
    dna_clothing TEXT,
    dna_ethnicity TEXT,
    dna_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS character_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id INTEGER NOT NULL,
    target_character_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,
    description TEXT,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE,
    FOREIGN KEY(target_character_id) REFERENCES characters(id) ON DELETE CASCADE,
    UNIQUE(character_id, target_character_id)
);


CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    tags TEXT,
    notes TEXT,
    folder_path TEXT NOT NULL,
    prompt_template TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sub_locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    prompt_template TEXT,
    folder_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(location_id) REFERENCES locations(id) ON DELETE CASCADE,
    UNIQUE(location_id, name)
);

CREATE TABLE IF NOT EXISTS timeline_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    event_order INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    event_date TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
    UNIQUE(project_id, event_order)
);

CREATE TABLE IF NOT EXISTS story_bible_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    category TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fps INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    model TEXT NOT NULL,
    pipeline TEXT,
    camera_motion TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id INTEGER,
    asset_type TEXT NOT NULL, -- 'image' or 'video'
    file_path TEXT NOT NULL,
    prompt TEXT,
    seed INTEGER,
    status TEXT NOT NULL DEFAULT 'completed', -- 'completed', 'failed', 'pending'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE SET NULL
);

-- Phase 4: Generated Images (Flux image generation pipeline)
CREATE TABLE IF NOT EXISTS generated_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    scene_id INTEGER,
    character_id INTEGER,
    location_id INTEGER,
    model TEXT NOT NULL,           -- 'flux_dev' | 'flux_kontext'
    prompt TEXT,
    negative_prompt TEXT,
    seed INTEGER,
    width INTEGER,
    height INTEGER,
    steps INTEGER,
    guidance_scale REAL,
    file_path TEXT NOT NULL,
    thumbnail_path TEXT,
    status TEXT DEFAULT 'pending', -- 'pending' | 'approved' | 'rejected'
    approved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE SET NULL,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL,
    FOREIGN KEY(location_id) REFERENCES locations(id) ON DELETE SET NULL
);

-- Phase 5: Generated Videos (LTX-2 video generation pipeline)
CREATE TABLE IF NOT EXISTS generated_videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    scene_id INTEGER,
    character_id INTEGER,
    location_id INTEGER,
    model_family TEXT DEFAULT 'LTX-Video', -- 'LTX-Video' | 'Wan 2.2' | 'Hunyuan Video'
    pipeline TEXT NOT NULL,          -- 'distilled' | 'two_stage' | 'two_stage_hq' | 'text2video' | 'image2video'
    preset TEXT,                     -- 'draft' | 'production'
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
    status TEXT DEFAULT 'pending',   -- 'pending' | 'approved' | 'rejected'
    approved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE SET NULL,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL,
    FOREIGN KEY(location_id) REFERENCES locations(id) ON DELETE SET NULL
);

-- Pre-populate default presets
INSERT OR IGNORE INTO presets (name, width, height, fps, frame_count, model, pipeline, camera_motion, notes) VALUES
('TikTok/Reels vertical (Fast LTX)', 720, 1280, 24, 96, 'LTX-2', 'Distilled', 'Static camera, high detail character focus', 'Optimized for mobile vertical streams'),
('YouTube Shorts (WAN 2.2 High)', 1080, 1920, 30, 120, 'Wan 2.2', 'I2V-14B', 'Slow tracking shot forward', 'Full HD cinematic quality preset'),
('Hunyuan Cinematic Vertical', 720, 1280, 24, 96, 'Hunyuan Video', 'Standard', 'Pan right, depth of field blur', 'Emotional closeup optimized preset');

-- Phase 7: Shot Templates and Render Queue
CREATE TABLE IF NOT EXISTS shot_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    prompt_addition TEXT,
    negative_prompt_addition TEXT
);

CREATE TABLE IF NOT EXISTS render_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    scene_id INTEGER,
    character_id INTEGER,
    location_id INTEGER,
    shot_type_id INTEGER,
    model_family TEXT,
    pipeline TEXT,
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
    status TEXT DEFAULT 'queued', -- 'queued', 'running', 'completed', 'failed', 'cancelled'
    progress REAL DEFAULT 0.0,
    error_message TEXT,
    video_id INTEGER, -- FK to generated_videos once completed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE SET NULL,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL,
    FOREIGN KEY(location_id) REFERENCES locations(id) ON DELETE SET NULL,
    FOREIGN KEY(shot_type_id) REFERENCES shot_templates(id) ON DELETE SET NULL
);

INSERT OR IGNORE INTO shot_templates (name, description, prompt_addition, negative_prompt_addition) VALUES
('Closeup', 'Tight shot focusing on face/emotions', 'extreme closeup, tight framing, detailed face, deep depth of field, blurred background', ''),
('Medium Shot', 'Waist up, good for dialogue', 'medium shot, waist up framing, subject centered', ''),
('Wide Shot', 'Establishes location and character placement', 'wide establishing shot, full body, cinematic environment, grand scale', ''),
('Walking', 'Character moving towards or parallel to camera', 'tracking shot, character walking, steady camera movement', ''),
('Conversation', 'Over the shoulder or two-shot for dialogue', 'two-shot, over the shoulder, characters conversing, dynamic interaction', ''),
('Dining Table', 'Characters seated around a table', 'seated at dining table, food props, eye level camera', ''),
('Phone Call', 'Character talking on phone', 'holding phone to ear, speaking on phone, intimate lighting', ''),
('Window Reflection', 'Looking out a window with reflection', 'looking through glass, window reflection, moody atmospheric lighting', '');

-- Phase 8: Voice Ready Architecture
CREATE TABLE IF NOT EXISTS dialogue_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id INTEGER NOT NULL,
    character_id INTEGER NOT NULL,
    sequence_order INTEGER NOT NULL,
    text TEXT NOT NULL,
    translation_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE CASCADE,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audio_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dialogue_line_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    duration REAL,
    status TEXT DEFAULT 'completed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dialogue_line_id) REFERENCES dialogue_lines(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audio_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    episode_id INTEGER,
    scene_id INTEGER,
    character_id INTEGER,
    dialogue_text TEXT NOT NULL,
    voice_id TEXT NOT NULL,
    speed REAL DEFAULT 1.0,
    status TEXT DEFAULT 'queued',
    file_path TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
    FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE SET NULL,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE SET NULL,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL
);

-- ==========================================
-- PHASE 10: INTELLIGENCE LAYER
-- ==========================================

CREATE TABLE IF NOT EXISTS shots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id INTEGER NOT NULL,
    shot_number INTEGER NOT NULL,
    shot_type TEXT,
    description TEXT,
    camera_direction TEXT,
    camera_motion TEXT,
    character_focus INTEGER,
    duration REAL DEFAULT 3.0,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE CASCADE,
    FOREIGN KEY(character_focus) REFERENCES characters(id) ON DELETE SET NULL,
    UNIQUE(scene_id, shot_number)
);

CREATE TABLE IF NOT EXISTS character_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id INTEGER UNIQUE NOT NULL,
    primary_wardrobe_id INTEGER,
    default_expression TEXT,
    visual_style TEXT,
    reference_strength REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS approved_character_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id INTEGER NOT NULL,
    asset_type TEXT NOT NULL, 
    file_path TEXT NOT NULL,
    is_primary_reference BOOLEAN DEFAULT 0,
    wardrobe_tag TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS continuity_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id INTEGER UNIQUE NOT NULL,
    location_id INTEGER,
    wardrobe_tag TEXT,
    weather TEXT,
    time_of_day TEXT,
    lighting TEXT,
    props TEXT,
    character_state TEXT,
    scene_state TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE CASCADE,
    FOREIGN KEY(location_id) REFERENCES locations(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS asset_reference_graph (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_video_id INTEGER NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(target_video_id) REFERENCES generated_videos(id) ON DELETE CASCADE
);

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
    completed_at TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
    FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE SET NULL,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE SET NULL,
    FOREIGN KEY(shot_id) REFERENCES shots(id) ON DELETE SET NULL,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL,
    FOREIGN KEY(dialogue_line_id) REFERENCES dialogue_lines(id) ON DELETE SET NULL,
    FOREIGN KEY(audio_asset_id) REFERENCES audio_assets(id) ON DELETE SET NULL,
    FOREIGN KEY(source_video_id) REFERENCES generated_videos(id) ON DELETE SET NULL
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(queue_job_id) REFERENCES lip_sync_queue(id) ON DELETE CASCADE
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
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL,
    FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE SET NULL,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE SET NULL,
    FOREIGN KEY(shot_id) REFERENCES shots(id) ON DELETE SET NULL,
    FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE SET NULL
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
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(assembly_id) REFERENCES episode_assemblies(id) ON DELETE CASCADE,
    FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE SET NULL,
    FOREIGN KEY(shot_id) REFERENCES shots(id) ON DELETE SET NULL,
    FOREIGN KEY(source_video_id) REFERENCES generated_videos(id) ON DELETE SET NULL,
    FOREIGN KEY(source_audio_asset_id) REFERENCES audio_assets(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_shots_scene ON shots(scene_id, shot_number);
CREATE INDEX IF NOT EXISTS idx_render_queue_status ON render_queue(status, created_at);
CREATE INDEX IF NOT EXISTS idx_audio_queue_status ON audio_queue(status, created_at);
CREATE INDEX IF NOT EXISTS idx_lipsync_queue_status ON lip_sync_queue(status, created_at);
CREATE INDEX IF NOT EXISTS idx_consistency_reports_scope ON consistency_reports(project_id, episode_id, scene_id, shot_id, character_id);
