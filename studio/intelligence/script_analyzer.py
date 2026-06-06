"""
Rule-based script-aware scene builder.

Generates a persistent shot list and dialogue lines from scene script text
without requiring an LLM dependency.
"""
import json
import re
from studio.database.db_manager import DatabaseManager
from studio.intelligence.cinematography_engine import CinematographyEngine


_CAMERA_HINTS = {
    "whisper": "Extreme Close Up",
    "argument": "Close Up",
    "walk": "Tracking Shot",
    "run": "Tracking Shot",
    "reveal": "Crane Shot",
    "look": "Over Shoulder",
    "phone": "Close Up",
}


class ScriptAnalyzer:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.camera_engine = CinematographyEngine(db)

    def analyze_scene_to_shots(self, scene_id: int, dialogue_text: str, scene_description: str) -> list[dict]:
        scene = self.db.get_scene(scene_id)
        if not scene:
            return []

        lines = self._parse_dialogue(dialogue_text)
        scene_keywords = f"{scene_description or ''} {dialogue_text or ''}".lower()
        camera_language = self._guess_camera_language(scene_keywords)

        establishing_shot = self._save_shot(
            scene_id=scene_id,
            shot_number=1,
            shot_type="Wide Shot",
            title="Establishing Beat",
            description=scene_description or "Establish the location, geography, and emotional temperature of the scene.",
            goal="Orient the audience to the scene.",
            camera_language="Wide Shot",
            movement_preset="Locked Camera",
            prompt=scene.get("video_prompt") or scene_description or "",
            character_focus=scene.get("character_id"),
        )
        results = [establishing_shot]

        for index, line in enumerate(lines, start=2):
            char_obj = self.db.get_character_by_name(line["character"])
            line_character_id = char_obj["id"] if char_obj else scene.get("character_id")
            shot = self._save_shot(
                scene_id=scene_id,
                shot_number=index,
                shot_type=camera_language,
                title=f"Dialogue Beat {index - 1}",
                description=f"{line['character']} delivers: {line['text']}",
                goal=f"Cover the performance beat for {line['character']}.",
                camera_language=camera_language,
                movement_preset="Locked Camera" if camera_language in ("Close Up", "Over Shoulder") else "Dolly In",
                prompt=f"{scene_description or ''} {line['character']} saying '{line['text']}'".strip(),
                character_focus=line_character_id,
            )
            if line_character_id:
                self.db.add_dialogue_line(
                    scene_id=scene_id,
                    character_id=line_character_id,
                    shot_id=shot["id"],
                    sequence_order=index - 1,
                    text=line["text"],
                    emotion=line["emotion"],
                    intensity=line["intensity"],
                    expression=line["emotion"],
                    speaking_style=line["speaking_style"],
                )
            results.append(shot)

        if not lines and scene_description:
            results.append(
                self._save_shot(
                    scene_id=scene_id,
                    shot_number=2,
                    shot_type=camera_language,
                    title="Action Beat",
                    description=scene_description,
                    goal="Cover the scene action.",
                    camera_language=camera_language,
                    movement_preset="Dolly In" if camera_language == "Wide Shot" else "Locked Camera",
                    prompt=scene_description,
                    character_focus=scene.get("character_id"),
                )
            )
        return results

    def _save_shot(self, **kwargs) -> dict:
        shot_id = self.db.add_shot(**kwargs)
        shot = self.db.get_shot(shot_id)
        camera_plan = self.camera_engine.build_camera_plan(shot)
        self.db.add_shot(
            scene_id=shot["scene_id"],
            shot_number=shot["shot_number"],
            shot_type=shot.get("shot_type") or camera_plan["camera_language"],
            description=shot.get("description") or "",
            camera_direction=shot.get("camera_direction") or shot.get("shot_type") or camera_plan["camera_language"],
            camera_motion=camera_plan["movement_preset"],
            character_focus=shot.get("character_focus"),
            duration=shot.get("duration") or 3.0,
            title=shot.get("title") or "",
            goal=shot.get("goal") or "",
            camera_language=camera_plan["camera_language"],
            lens_preset=camera_plan["lens_preset"],
            movement_preset=camera_plan["movement_preset"],
            prompt=f"{shot.get('prompt') or shot.get('description') or ''}, {camera_plan['prompt_addition']}".strip(", "),
            negative_prompt=camera_plan["negative_prompt_addition"],
            references_json=shot.get("references_json") or "[]",
            status=shot.get("status") or "planned",
        )
        return self.db.get_shot(shot_id)

    def _parse_dialogue(self, dialogue_text: str) -> list[dict]:
        results = []
        for raw_line in (dialogue_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^([^:]+):\s*(.*)$", line)
            if not match:
                continue
            character = match.group(1).strip()
            text = match.group(2).strip()
            lowered = text.lower()
            emotion = "neutral"
            intensity = 5
            if any(word in lowered for word in ("angry", "furious", "shout")):
                emotion = "angry"
                intensity = 8
            elif any(word in lowered for word in ("sad", "cry", "tears")):
                emotion = "sad"
                intensity = 7
            elif any(word in lowered for word in ("whisper", "quiet")):
                emotion = "tense"
                intensity = 4
            speaking_style = "measured" if "..." in text else "natural"
            results.append(
                {
                    "character": character,
                    "text": text,
                    "emotion": emotion,
                    "intensity": intensity,
                    "speaking_style": speaking_style,
                }
            )
        return results

    def _guess_camera_language(self, scene_keywords: str) -> str:
        for keyword, camera in _CAMERA_HINTS.items():
            if keyword in scene_keywords:
                return camera
        return "Medium Shot"
