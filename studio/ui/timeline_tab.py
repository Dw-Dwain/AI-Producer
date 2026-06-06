"""
Episode Assembly Engine UI.
"""
import json
import gradio as gr
from studio.database.db_manager import DatabaseManager
from studio.intelligence.episode_assembly import EpisodeAssemblyEngine


def create_timeline_tab(db: DatabaseManager):
    with gr.TabItem("Episode Assembly"):
        gr.Markdown("### Timeline, Sequencer, and Export Preview")
        with gr.Row():
            proj_sel = gr.Dropdown(label="Project", choices=[], interactive=True)
            ep_sel = gr.Dropdown(label="Episode", choices=[], interactive=True)
            assembly_name = gr.Textbox(label="Assembly Name", value="Latest Cut")
        btn_refresh = gr.Button("Build Assembly Preview", variant="primary")
        timeline_df = gr.Dataframe(
            headers=["Order", "Scene", "Shot", "Duration", "Video"],
            datatype=["number", "str", "str", "number", "str"],
            interactive=False,
        )
        timeline_gallery = gr.Gallery(label="Sequenced Shots", columns=4, object_fit="cover", height=420)
        assembly_summary = gr.Markdown("*Assembly preview will appear here.*")

        def load_episodes(proj_name):
            if not proj_name:
                return gr.update(choices=[], value=None)
            project = db.get_project_by_name(proj_name)
            episodes = db.list_episodes(project["id"]) if project else []
            labels = [f"Ep {ep['episode_number']}: {ep['title'] or 'Untitled'}" for ep in episodes]
            return gr.update(choices=labels, value=labels[0] if labels else None)

        def update_timeline(proj_name, ep_label, name):
            if not proj_name or not ep_label:
                return [], [], "Select a project and episode."
            project = db.get_project_by_name(proj_name)
            ep_num = int(ep_label.split(":")[0].replace("Ep ", "").strip())
            episode = db.get_episode_by_details(project["id"], ep_num) if project else None
            if not project or not episode:
                return [], [], "Episode not found."
            payload = EpisodeAssemblyEngine(db).build_episode_preview(project["id"], episode["id"], name=name or "Latest Cut")
            rows = []
            gallery = []
            for item in payload.get("items", []):
                metadata = item.get("metadata_json") or "{}"
                try:
                    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
                except Exception:
                    metadata = {}
                rows.append([
                    item.get("sequence_order"),
                    metadata.get("scene_label", ""),
                    metadata.get("shot_label", ""),
                    item.get("duration_seconds", 0.0),
                    metadata.get("video_path", ""),
                ])
                if metadata.get("video_path"):
                    gallery.append((metadata["video_path"], f"{metadata.get('scene_label', '')} {metadata.get('shot_label', '')}".strip()))
            preview = payload.get("preview", {})
            summary = f"**Scenes:** {preview.get('scene_count', 0)}  \n**Shots:** {preview.get('shot_count', 0)}  \n**Duration:** {preview.get('duration_seconds', 0)} sec  \n**Layers:** audio, subtitles, music, export"
            return rows, gallery, summary

        proj_sel.change(load_episodes, [proj_sel], [ep_sel])
        btn_refresh.click(update_timeline, [proj_sel, ep_sel, assembly_name], [timeline_df, timeline_gallery, assembly_summary])
        return proj_sel, ep_sel
