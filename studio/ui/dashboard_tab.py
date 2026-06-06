"""
Production dashboard for consistency, reference graph, and asset metrics.
"""
import json
import gradio as gr
from studio.database.db_manager import DatabaseManager


def create_dashboard_tab(db: DatabaseManager):
    with gr.TabItem("Production Dashboard"):
        gr.Markdown("### Production Metrics")
        metrics_scenes = gr.Number(label="Scenes", value=0, interactive=False)
        metrics_shots = gr.Number(label="Shots", value=0, interactive=False)
        metrics_videos = gr.Number(label="Approved Videos", value=0, interactive=False)

        gr.Markdown("### Reference Graph and Consistency")
        search_char = gr.Dropdown(label="Character", choices=[], interactive=True)
        btn_refresh = gr.Button("Refresh Dashboard", variant="primary")
        consistency_df = gr.Dataframe(
            headers=["Shot ID", "Character", "Score", "Level", "Warnings"],
            datatype=["number", "str", "number", "str", "str"],
            interactive=False,
        )
        search_gallery = gr.Gallery(label="Approved Assets", columns=4, object_fit="cover", height=340)

        def do_refresh(char_name):
            characters = db.list_characters()
            char_choices = [char["name"] for char in characters]
            char_obj = db.get_character_by_name(char_name) if char_name else None
            assets = db.list_approved_assets(char_obj["id"]) if char_obj else db.list_approved_assets()
            reports = []
            approved_videos = db.list_generated_videos(status="approved", limit=500)
            for video in approved_videos:
                report = db.get_latest_consistency_report(shot_id=video.get("shot_id"), character_id=video.get("character_id"), scene_id=video.get("scene_id"))
                if report:
                    reports.append([
                        report.get("shot_id"),
                        char_name or video.get("character_name", ""),
                        report.get("consistency_score", 0.0),
                        report.get("warning_level", ""),
                        ", ".join(json.loads(report.get("warnings_json") or "[]")),
                    ])
            gallery = [(asset["file_path"], asset.get("asset_type", "asset")) for asset in assets]
            return (
                gr.update(choices=char_choices, value=char_name if char_name in char_choices else None),
                len({video.get("scene_id") for video in approved_videos if video.get("scene_id")}),
                sum(len(db.list_shots(scene["id"])) for project in db.list_projects() for episode in db.list_episodes(project["id"]) for scene in db.list_scenes(episode["id"])),
                len(approved_videos),
                reports,
                gallery,
            )

        btn_refresh.click(do_refresh, [search_char], [search_char, metrics_scenes, metrics_shots, metrics_videos, consistency_df, search_gallery])
        return search_char
