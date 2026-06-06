# AI Drama Production Studio - Phase 1: Core Studio Foundation

Welcome to the **AI Drama Production Studio** (Phase 1). This project establishes the foundation for managing projects, episodes, characters, locations, and scenes in a premium, responsive Gradio interface.

## Project Structure

```
/studio
 ├── database/
 │    ├── __init__.py
 │    ├── db_manager.py       # SQLite connection manager and CRUD queries
 │    └── schema.sql          # DB schema definition SQL
 ├── ui/
 │    ├── __init__.py
 │    ├── theme.py            # Premium dark theme and styling
 │    ├── dashboard_tab.py    # Main status dashboard & quick-start panel
 │    ├── project_tab.py      # Project, episode, scene hierarchy builder
 │    ├── character_tab.py    # Character directory and image portfolio viewer
 │    ├── location_tab.py     # Shot location catalog and reference manager
 │    ├── scene_tab.py        # Scene editor with casting & prompt scripting
 │    ├── preset_tab.py       # Render presets panel
 │    └── settings_tab.py     # System session configurations
 ├── app.py                   # Main Gradio application running the web server
 ├── requirements.txt         # Minimal Python requirements
 ├── start.sh                 # Startup script for RunPod instances
 └── README.md                # Deployment and installation guide
```

## Setup & Running Locally

1. **Install Dependencies**:
   Ensure you have Python 3.9+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   Start the Gradio application locally:
   ```bash
   python app.py
   ```
   Or explicitly choose a port:
   ```bash
   python app.py --port 7860
   ```

3. **Access the Studio**:
   Open your browser and navigate to `http://localhost:7860`.

## Key Features in Phase 1

- **Session Context Auto-Restore**: Select a project, episode, and scene context on the Dashboard and save it. It will auto-save to SQLite settings and reload exactly where you left off.
- **Glassmorphic Dark Mode Theme**: Premium theme featuring custom typography, neon purple/indigo accent colors, translucent containers, and hover effects.
- **Asset Portfolios**: Create characters and locations; they get allocated physical subfolders on disk (`/studio/characters/{name}` and `/studio/locations/{name}`) where references and photos are copied and shown in dynamic galleries.
- **Scene Prompt Scripting**: Draft Flux prompts and camera motion scripts per scene, with cast lookups.
- **Presets Library**: Save/update rendering configurations for TikTok, YouTube Shorts, or custom aspect sizes.
