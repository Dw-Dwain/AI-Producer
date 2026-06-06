"""
Creates a persistent episode assembly preview from approved shots, audio, subtitles,
and future export layers.
"""
import json
from studio.database.db_manager import DatabaseManager


class EpisodeAssemblyEngine:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def build_episode_preview(self, project_id: int, episode_id: int, name: str = "Latest Cut") -> dict:
        scenes = self.db.list_scenes(episode_id)
        approved_videos = self.db.list_generated_videos(project_id=project_id, status="approved", limit=1000)

        items = []
        sequence_order = 1
        total_duration = 0.0
        for scene in scenes:
            shots = self.db.list_shots(scene["id"])
            for shot in shots:
                video = next((v for v in approved_videos if v.get("shot_id") == shot["id"]), None)
                subtitles = []
                for line in self.db.list_dialogue_lines(scene["id"]):
                    if line.get("shot_id") in (None, shot["id"]):
                        subtitles.append(f"{line['character_name']}: {line['text']}")
                duration = float(video.get("duration_seconds") or shot.get("duration") or 0.0) if video else float(shot.get("duration") or 0.0)
                items.append(
                    {
                        "item_type": "shot",
                        "scene_id": scene["id"],
                        "shot_id": shot["id"],
                        "source_video_id": video.get("id") if video else None,
                        "subtitle_text": "\n".join(subtitles),
                        "sequence_order": sequence_order,
                        "duration_seconds": duration,
                        "metadata": {
                            "scene_label": f"Scene {scene['scene_number']}",
                            "shot_label": f"Shot {shot['shot_number']}",
                            "video_path": video.get("file_path") if video else "",
                        },
                    }
                )
                sequence_order += 1
                total_duration += duration

        preview = {
            "scene_count": len(scenes),
            "shot_count": len(items),
            "duration_seconds": round(total_duration, 2),
            "subtitle_layer": True,
            "music_layer": True,
            "audio_layer": True,
            "future_integrations": ["voice cloning", "translation", "auto subtitles", "auto editing"],
        }
        assembly_id = self.db.upsert_episode_assembly(project_id, episode_id, name=name, preview=preview)
        self.db.replace_episode_assembly_items(assembly_id, items)
        payload = self.db.get_episode_assembly(assembly_id) or {}
        payload["preview"] = preview
        return payload
