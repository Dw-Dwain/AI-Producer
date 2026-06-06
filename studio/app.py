import os
import sys
import logging
import gradio as gr
from pathlib import Path

# Add project root to sys path
STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(STUDIO_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Initialize Path Abstractions and Centralized Logging
from studio.deployment.paths import STUDIO_ROOT, DB_PATH, LOG_DIR, ensure_directories
ensure_directories()

from studio.deployment.logging_setup import setup_centralized_logging
setup_centralized_logging(LOG_DIR)

logger = logging.getLogger("studio.app")
logger.info("==================================================")
logger.info("Initializing AI Drama Production Studio - Production Pipeline Expansion")
logger.info("==================================================")

from studio.database.db_manager import DatabaseManager
from studio.ui.theme import get_custom_theme
from studio.ui.workspace_tab import create_workspace_tab
from studio.ui.project_tab import create_project_tab
from studio.ui.character_tab import create_character_tab
from studio.ui.location_tab import create_location_tab
from studio.ui.story_bible_tab import create_story_bible_tab
from studio.ui.scene_tab import create_scene_tab
from studio.ui.preset_tab import create_preset_tab
from studio.ui.settings_tab import create_settings_tab
from studio.ui.timeline_tab import create_timeline_tab
from studio.ui.dashboard_tab import create_dashboard_tab
from studio.generation.worker import start_render_worker
from studio.deployment.diagnostics import run_system_startup_checks

db = DatabaseManager()

# Run Startup Checks
startup_report = run_system_startup_checks(STUDIO_ROOT, DB_PATH, STUDIO_ROOT / "models")
if not startup_report["ready"]:
    logger.error("Startup validation failed with critical errors:")
    for err in startup_report["critical_errors"]:
        logger.error(err)
else:
    logger.info("Startup validation checks completed successfully. System ready.")

start_render_worker(db, str(STUDIO_ROOT))
theme, css = get_custom_theme()


with gr.Blocks(theme=theme, css=css, title="AI Drama Production Studio") as demo:
    gr.HTML("<h1 class='studio-title'>AI DRAMA PRODUCTION STUDIO</h1>")
    gr.HTML("<p class='studio-subtitle'>Shot-driven production intelligence, lip sync, performance, consistency, cinematography, and episode assembly</p>")

    # Critical startup warning banner
    if not startup_report["ready"]:
        with gr.Row(variant="stop"):
            with gr.Column():
                gr.Markdown("### 🚨 Critical Startup Validation Failures Detected!")
                for err in startup_report["critical_errors"]:
                    gr.Markdown(f"- {err}")
                gr.Markdown("*Please review the Settings tab config, database table integrity, or console logs for more details.*")

    with gr.Tabs():
        workspace_outputs = create_workspace_tab(db)
        with gr.TabItem("Pre-Production"):
            with gr.Tabs():
                project_outputs = create_project_tab(db)
                character_outputs = create_character_tab(db)
                location_outputs = create_location_tab(db)
                story_bible_outputs = create_story_bible_tab(db)
                scene_outputs = create_scene_tab(db)
        with gr.TabItem("Production Intelligence"):
            with gr.Tabs():
                timeline_outputs = create_timeline_tab(db)
                dashboard_outputs = create_dashboard_tab(db)
        with gr.TabItem("Infrastructure"):
            with gr.Tabs():
                preset_outputs = create_preset_tab(db)
                settings_outputs = create_settings_tab(db)

    ws_proj, ws_ep, ws_sc, ws_char, ws_loc, ws_shot = workspace_outputs
    active_proj_dropdown, proj_delete_selector, ep_delete_selector = project_outputs
    char_selector, filter_gender, filter_tag, filter_project, filter_location, rel_target = character_outputs
    loc_selector, loc_filter_tag, loc_filter_project, loc_filter_char = location_outputs
    sb_project_selector = story_bible_outputs[0]
    scene_proj_selector, scene_ep_selector, scene_sc_selector, casting_char, casting_loc = scene_outputs
    timeline_proj_selector, timeline_ep_selector = timeline_outputs
    dashboard_char_selector = dashboard_outputs
    preset_selector = preset_outputs[0]

    def restore_session_on_load():
        projs = db.list_projects()
        proj_names = [p["name"] for p in projs]
        chars = db.list_characters()
        char_names = [c["name"] for c in chars]
        locs = db.list_locations()
        loc_names = [l["name"] for l in locs]
        presets = db.list_presets()
        preset_names = [p["name"] for p in presets]

        genders = sorted({c["gender"] for c in chars if c["gender"]})
        all_tags = sorted({tag.strip() for c in chars for tag in (c["tags"] or "").split(",") if tag.strip()})
        loc_tags = sorted({tag.strip() for l in locs for tag in (l["tags"] or "").split(",") if tag.strip()})

        sess_proj = db.get_setting("session_project", "")
        sess_ep = db.get_setting("session_episode", "")
        sess_sc = db.get_setting("session_scene", "")

        proj_val = sess_proj if sess_proj in proj_names else (proj_names[0] if proj_names else None)
        ep_choices = []
        ep_val = None
        sc_choices = []
        sc_val = None
        if proj_val:
            project = db.get_project_by_name(proj_val)
            episodes = db.list_episodes(project["id"]) if project else []
            ep_choices = [f"Ep {ep['episode_number']}: {ep['title'] or 'Untitled'}" for ep in episodes]
            ep_val = sess_ep if sess_ep in ep_choices else (ep_choices[0] if ep_choices else None)
            if ep_val:
                ep_num = int(ep_val.split(":")[0].replace("Ep ", "").strip())
                episode = db.get_episode_by_details(project["id"], ep_num)
                scenes = db.list_scenes(episode["id"]) if episode else []
                sc_choices = [f"Scene {scene['scene_number']}: {scene['title'] or 'Untitled'}" for scene in scenes]
                sc_val = sess_sc if sess_sc in sc_choices else (sc_choices[0] if sc_choices else None)

        return (
            gr.update(choices=proj_names, value=proj_val),
            gr.update(choices=ep_choices, value=ep_val),
            gr.update(choices=sc_choices, value=sc_val),
            gr.update(choices=char_names, value=char_names[0] if char_names else None),
            gr.update(choices=loc_names, value=loc_names[0] if loc_names else None),
            gr.update(choices=[], value=None),
            gr.update(choices=proj_names, value=proj_val),
            gr.update(choices=proj_names, value=proj_val),
            gr.update(choices=ep_choices, value=ep_val),
            gr.update(choices=char_names, value=char_names[0] if char_names else None),
            gr.update(choices=["All"] + genders, value="All"),
            gr.update(choices=["All"] + all_tags, value="All"),
            gr.update(choices=["All"] + proj_names, value="All"),
            gr.update(choices=["All"] + loc_names, value="All"),
            gr.update(choices=char_names, value=None),
            gr.update(choices=loc_names, value=loc_names[0] if loc_names else None),
            gr.update(choices=["All"] + loc_tags, value="All"),
            gr.update(choices=["All"] + proj_names, value="All"),
            gr.update(choices=["All"] + char_names, value="All"),
            gr.update(choices=proj_names, value=proj_val),
            gr.update(choices=proj_names, value=proj_val),
            gr.update(choices=ep_choices, value=ep_val),
            gr.update(choices=sc_choices, value=sc_val),
            gr.update(choices=["None"] + char_names, value="None"),
            gr.update(choices=["None"] + loc_names, value="None"),
            gr.update(choices=proj_names, value=proj_val),
            gr.update(choices=ep_choices, value=ep_val),
            gr.update(choices=char_names, value=char_names[0] if char_names else None),
            gr.update(choices=preset_names, value=preset_names[0] if preset_names else None),
        )

    demo.load(
        fn=restore_session_on_load,
        inputs=None,
        outputs=[
            ws_proj, ws_ep, ws_sc, ws_char, ws_loc, ws_shot,
            active_proj_dropdown, proj_delete_selector, ep_delete_selector,
            char_selector, filter_gender, filter_tag, filter_project, filter_location, rel_target,
            loc_selector, loc_filter_tag, loc_filter_project, loc_filter_char,
            sb_project_selector,
            scene_proj_selector, scene_ep_selector, scene_sc_selector, casting_char, casting_loc,
            timeline_proj_selector, timeline_ep_selector, dashboard_char_selector,
            preset_selector,
        ],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start AI Drama Production Studio")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # Directories are already ensured via paths.ensure_directories()
    # Read port from environment variable fallback to argument
    port_val = int(os.getenv("PORT", args.port))

    demo.launch(server_name="0.0.0.0", server_port=port_val, share=args.share)
