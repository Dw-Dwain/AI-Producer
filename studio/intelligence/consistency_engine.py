"""
Character consistency scoring built from recent approved assets, continuity, and
character memory. The heuristics are deliberately transparent and overrideable.
"""
import json
from studio.database.db_manager import DatabaseManager


class ConsistencyEngine:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def evaluate_shot(self, project_id: int, episode_id: int, scene: dict, shot: dict, character_id: int = None, allow_override: bool = False) -> dict:
        memory = self.db.get_character_memory(character_id) if character_id else None
        continuity = self.db.get_continuity_state(scene["id"]) if scene else None
        approved_assets = self.db.list_approved_assets(character_id) if character_id else []

        face_drift = 0.05 if approved_assets else 0.35
        hair_drift = 0.08 if memory and memory.get("visual_style") else 0.28
        age_drift = 0.06 if memory else 0.22
        wardrobe_drift = 0.10 if continuity and continuity.get("wardrobe_tag") else 0.30
        location_drift = 0.08 if continuity and continuity.get("location_id") else 0.25
        reference_strength = float(memory.get("reference_strength", 1.0)) if memory else 0.4

        drift_total = face_drift + hair_drift + age_drift + wardrobe_drift + location_drift
        score = max(0.0, min(1.0, reference_strength - (drift_total / 5.0) + 0.35))

        warnings = []
        suggestions = []
        if not approved_assets:
            warnings.append("No approved character reference assets linked.")
        if continuity and not continuity.get("wardrobe_tag"):
            warnings.append("Wardrobe continuity is not set for this scene.")
        if continuity and not continuity.get("location_id"):
            warnings.append("Location continuity is not pinned for this scene.")
        if score < 0.75:
            suggestions.extend(asset["file_path"] for asset in approved_assets[:3])
        if memory and memory.get("best_references_json"):
            suggestions.extend(json.loads(memory["best_references_json"]))

        warning_level = "ok"
        if score < 0.85 or warnings:
            warning_level = "warn"
        if score < 0.65:
            warning_level = "critical"

        report = {
            "face_drift": round(face_drift, 3),
            "hair_drift": round(hair_drift, 3),
            "age_drift": round(age_drift, 3),
            "wardrobe_drift": round(wardrobe_drift, 3),
            "location_drift": round(location_drift, 3),
            "reference_strength": round(reference_strength, 3),
            "consistency_score": round(score, 3),
            "warning_level": warning_level,
            "warnings": warnings,
            "suggestions": suggestions[:5],
            "override_enabled": allow_override,
            "project_id": project_id,
            "episode_id": episode_id,
            "scene_id": scene["id"] if scene else None,
            "shot_id": shot["id"] if shot else None,
            "character_id": character_id,
        }
        self.db.save_consistency_report(**report)
        return report
