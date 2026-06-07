"""
Episode assembly: builds metadata preview and exports final video via ffmpeg.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
from studio.database.db_manager import DatabaseManager

logger = logging.getLogger("studio.assembly")


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
        }
        assembly_id = self.db.upsert_episode_assembly(project_id, episode_id, name=name, preview=preview)
        self.db.replace_episode_assembly_items(assembly_id, items)
        payload = self.db.get_episode_assembly(assembly_id) or {}
        payload["preview"] = preview
        return payload

    def export_episode_video(
        self,
        project_id: int,
        episode_id: int,
        output_path: str,
        burn_subtitles: bool = False,
        assembly_name: str = "Latest Cut",
    ) -> dict:
        """
        Concatenates all approved shot videos for an episode into a single MP4.
        Uses ffmpeg concat demuxer — no re-encode (stream copy) for speed.
        Set burn_subtitles=True to hard-burn subtitle text via ffmpeg drawtext.
        Returns {"success": bool, "output_path": str, "error": str}.
        """
        if not shutil.which("ffmpeg"):
            return {"success": False, "error": "ffmpeg not found on PATH. Install it with: apt-get install ffmpeg"}

        # Ensure the assembly preview is up to date
        preview_payload = self.build_episode_preview(project_id, episode_id, name=assembly_name)
        assembly_id = preview_payload.get("id")
        if not assembly_id:
            return {"success": False, "error": "Failed to create/find episode assembly record."}

        assembly_record = self.db.get_episode_assembly(assembly_id)
        items = (assembly_record or {}).get("items", [])

        # Collect video paths in sequence order
        clip_entries = []
        for item in sorted(items, key=lambda x: x.get("sequence_order", 0)):
            if item.get("item_type") != "shot":
                continue
            try:
                meta = json.loads(item.get("metadata_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                meta = {}
            vp = meta.get("video_path", "")
            subtitle = (item.get("subtitle_text") or "").strip()
            if vp and os.path.isfile(vp):
                clip_entries.append({"path": vp, "subtitle": subtitle, "duration": item.get("duration_seconds", 0)})
            else:
                logger.warning(f"Assembly item skipped — video not found: {vp!r}")

        if not clip_entries:
            return {"success": False, "error": "No approved video clips found for this episode. Generate and approve shots first."}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if burn_subtitles:
            return self._export_with_subtitles(clip_entries, output_path)
        else:
            return self._export_concat(clip_entries, output_path)

    def _export_concat(self, clip_entries: list, output_path: str) -> dict:
        """Fast stream-copy concat — no subtitle burn, no re-encode."""
        concat_file = output_path + "_concat_list.txt"
        try:
            with open(concat_file, "w", encoding="utf-8") as f:
                for entry in clip_entries:
                    # ffmpeg concat demuxer requires forward-slash paths even on Windows
                    safe_path = entry["path"].replace("\\", "/")
                    f.write(f"file '{safe_path}'\n")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path,
            ]
            logger.info(f"Running ffmpeg concat: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return {"success": False, "error": f"ffmpeg concat failed:\n{result.stderr}"}
            return {"success": True, "output_path": output_path, "clip_count": len(clip_entries)}
        except Exception as exc:
            logger.error(f"Episode export failed: {exc}", exc_info=True)
            return {"success": False, "error": str(exc)}
        finally:
            try:
                os.remove(concat_file)
            except OSError:
                pass

    def _export_with_subtitles(self, clip_entries: list, output_path: str) -> dict:
        """
        Concatenates clips with subtitle text burned in via ffmpeg drawtext.
        Each clip is re-encoded separately with its subtitle, then the results are concatenated.
        """
        tmp_dir = tempfile.mkdtemp(prefix="studio_assembly_")
        try:
            burnt_clips = []
            for i, entry in enumerate(clip_entries):
                clip_out = os.path.join(tmp_dir, f"clip_{i:04d}.mp4")
                subtitle_text = entry["subtitle"].replace("'", "\\'").replace(":", "\\:")
                if subtitle_text:
                    vf = (
                        f"drawtext=text='{subtitle_text}'"
                        f":fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5"
                        f":x=(w-text_w)/2:y=h-th-20"
                    )
                    cmd = [
                        "ffmpeg", "-y", "-i", entry["path"],
                        "-vf", vf, "-c:v", "libx264", "-crf", "18",
                        "-c:a", "copy", clip_out,
                    ]
                else:
                    cmd = [
                        "ffmpeg", "-y", "-i", entry["path"],
                        "-c:v", "libx264", "-crf", "18",
                        "-c:a", "copy", clip_out,
                    ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"Subtitle burn failed for clip {i}: {result.stderr}")
                    burnt_clips.append({"path": entry["path"], "subtitle": ""})
                else:
                    burnt_clips.append({"path": clip_out, "subtitle": ""})

            # Concat the burnt clips
            return self._export_concat(burnt_clips, output_path)
        except Exception as exc:
            logger.error(f"Subtitle export failed: {exc}", exc_info=True)
            return {"success": False, "error": str(exc)}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
