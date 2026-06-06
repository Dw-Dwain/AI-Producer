from studio.database.db_manager import DatabaseManager

def test_phase8_db():
    db = DatabaseManager()
    
    # 1. We need a character and a scene.
    char_id = db.add_character("Test Actor", 30, "Male", "Test", "Notes", "Tags", "/tmp/actor")
    if not char_id:
        char_id = db.get_character_by_name("Test Actor")["id"]

    proj_id = db.add_project("Test Project")
    if not proj_id:
        proj_id = db.get_project_by_name("Test Project")["id"]

    ep_id = db.add_episode(proj_id, 1, "Pilot")
    scene_id = db.add_scene(ep_id, 1, "Test Scene")
    
    dl_id = db.add_dialogue_line(scene_id, char_id, 1, "Hello world", "Hola mundo")
    assert dl_id is not None, "Failed to add dialogue line."
    
    lines = db.list_dialogue_lines(scene_id)
    assert len(lines) > 0, "No dialogue lines returned."
    
    audio_id = db.add_audio_asset(dl_id, "/tmp/audio.wav", 2.5)
    assert audio_id is not None, "Failed to add audio asset."
    
    print("Phase 8 DB tests passed.")

if __name__ == "__main__":
    test_phase8_db()
