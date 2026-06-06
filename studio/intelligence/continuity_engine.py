"""
Continuity and reference enrichment for shot-driven generation.
"""
import json
from studio.database.db_manager import DatabaseManager
from studio.intelligence.performance_engine import PerformanceEngine
from studio.intelligence.cinematography_engine import CinematographyEngine


class ContinuityEngine:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.performance_engine = PerformanceEngine(db)
        self.camera_engine = CinematographyEngine(db)

    def build_generation_context(self, scene_id: int, character_id: int, base_prompt: str, shot: dict | None = None, dialogue_line: dict | None = None) -> dict:
        enriched_prompt = base_prompt or ""
        reference_images = []
        reference_bundle = []
        continuity_notes = []

        if shot:
            camera_plan = self.camera_engine.build_camera_plan(shot)
            enriched_prompt = ", ".join(bit for bit in [enriched_prompt, camera_plan["prompt_addition"]] if bit)
        else:
            camera_plan = {
                "camera_language": "",
                "lens_preset": "",
                "movement_preset": "",
                "prompt_addition": "",
                "negative_prompt_addition": "",
            }

        if scene_id:
            cont = self.db.get_continuity_state(scene_id)
            if cont:
                for key in ("time_of_day", "weather", "lighting", "wardrobe_tag", "scene_state", "character_state"):
                    if cont.get(key):
                        continuity_notes.append(f"{key.replace('_', ' ')}: {cont[key]}")

        if character_id:
            memory = self.db.get_character_memory(character_id)
            approved_assets = self.db.list_approved_assets(character_id)
            if memory:
                if memory.get("visual_style"):
                    continuity_notes.append(f"character style: {memory['visual_style']}")
                if memory.get("last_known_wardrobe"):
                    continuity_notes.append(f"last wardrobe: {memory['last_known_wardrobe']}")
                if memory.get("best_references_json"):
                    for ref in json.loads(memory["best_references_json"]):
                        reference_bundle.append({"type": "memory", "path": ref})
                        reference_images.append(ref)
            for asset in approved_assets[:3]:
                reference_bundle.append({"type": asset["asset_type"], "path": asset["file_path"]})
                reference_images.append(asset["file_path"])

        performance = {}
        if dialogue_line:
            performance = self.performance_engine.build_dialogue_payload(dialogue_line, character_id)
            if performance["video_prompt_addition"]:
                continuity_notes.append(performance["video_prompt_addition"])

        if continuity_notes:
            enriched_prompt = f"{enriched_prompt} | Continuity: {', '.join(continuity_notes)}"

        return {
            "enriched_prompt": enriched_prompt.strip(),
            "reference_images": reference_images[:5],
            "references_json": json.dumps(reference_bundle[:5]),
            "continuity_notes": " | ".join(continuity_notes),
            "performance": performance,
            "camera": camera_plan,
        }
