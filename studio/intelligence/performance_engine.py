"""
Builds performance-aware voice, expression, and prompt instructions from
character defaults plus dialogue-specific overrides.
"""
import json
from studio.database.db_manager import DatabaseManager


class PerformanceEngine:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def build_dialogue_payload(self, dialogue_line: dict, character_id: int = None) -> dict:
        profile = self.db.get_character_performance_profile(character_id or dialogue_line.get("character_id")) or {}
        emotion = dialogue_line.get("emotion") or profile.get("default_emotion") or "neutral"
        intensity = int(dialogue_line.get("intensity") or profile.get("default_intensity") or 5)
        expression = dialogue_line.get("expression") or profile.get("default_expression") or emotion
        speaking_style = dialogue_line.get("speaking_style") or profile.get("speaking_style") or "natural"
        voice_id = profile.get("voice_id") or "af_heart"

        prompt_bits = [
            f"emotion: {emotion}",
            f"intensity: {intensity}/10",
            f"facial expression: {expression}",
            f"speaking style: {speaking_style}",
        ]
        if dialogue_line.get("text"):
            prompt_bits.append(f"spoken dialogue delivery: {dialogue_line['text']}")

        payload = {
            "voice_id": voice_id,
            "emotion": emotion,
            "intensity": intensity,
            "expression": expression,
            "speaking_style": speaking_style,
            "tts_instruction": f"{emotion} delivery at intensity {intensity}/10 with {speaking_style} pacing",
            "lip_sync_instruction": f"Match mouth articulation and facial tension for {emotion} at intensity {intensity}/10",
            "expression_instruction": f"Favor {expression} micro-expressions while speaking",
            "video_prompt_addition": ", ".join(prompt_bits),
        }
        payload["performance_json"] = json.dumps(payload)
        return payload
