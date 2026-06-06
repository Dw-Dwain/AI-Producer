"""
Maps supported camera language into reusable preset-driven prompt language.
"""
from studio.database.db_manager import DatabaseManager


SUPPORTED_CAMERA_LANGUAGE = [
    "Close Up",
    "Extreme Close Up",
    "Medium Shot",
    "Wide Shot",
    "Over Shoulder",
    "Dolly In",
    "Dolly Out",
    "Crane Shot",
    "Tracking Shot",
    "Handheld",
    "Locked Camera",
]


class CinematographyEngine:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def build_camera_plan(self, shot: dict) -> dict:
        preset = None
        if shot.get("camera_direction"):
            preset = self.db.get_camera_preset(shot["camera_direction"])
        if not preset and shot.get("camera_language"):
            preset = self.db.get_camera_preset(shot["camera_language"])

        camera_language = (
            shot.get("camera_language")
            or shot.get("shot_type")
            or (preset.get("camera_language") if preset else "Medium Shot")
        )
        lens_preset = shot.get("lens_preset") or (preset.get("lens_preset") if preset else "50mm cinematic lens")
        movement_preset = shot.get("movement_preset") or shot.get("camera_motion") or (preset.get("movement_preset") if preset else "Locked Camera")

        prompt_bits = [camera_language, lens_preset, movement_preset]
        if preset and preset.get("prompt_addition"):
            prompt_bits.append(preset["prompt_addition"])

        return {
            "camera_language": camera_language,
            "lens_preset": lens_preset,
            "movement_preset": movement_preset,
            "prompt_addition": ", ".join(bit for bit in prompt_bits if bit),
            "negative_prompt_addition": preset.get("negative_prompt_addition", "") if preset else "",
            "framing_notes": preset.get("framing_notes", "") if preset else "",
        }
