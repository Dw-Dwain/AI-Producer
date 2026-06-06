import os
import time
import logging
import threading
from studio.database.db_manager import DatabaseManager
from studio.generation import worker, ltx_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_e2e")

def test_full_pipeline():
    logger.info("Initializing E2E Test...")
    studio_root = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(studio_root, "test_e2e.db")
    
    # Clean up old db
    if os.path.exists(db_path):
        os.remove(db_path)
        
    db = DatabaseManager(db_path)

    # Need custom helper for the test
    def execute_custom_query(self, query, params):
        with self._get_connection() as conn:
            conn.execute(query, params)
            conn.commit()
    DatabaseManager.execute_custom_query = execute_custom_query

    # 1. Create Pre-Production Assets
    logger.info("Creating Pre-Production Assets...")
    proj_id = db.add_project("Bible Project", "Testing E2E")
    if proj_id is None:
        proj = db.get_project_by_name("Bible Project")
        proj_id = proj["id"]

    ep_id = db.add_episode(proj_id, 1, "Pilot", "The first episode")
    char_id = db.add_character("E2E Actor", 30, "Male", "Test", "Notes", "Tags", "/tmp/actor")
    loc_id = db.add_location("E2E Set", "A test set", "indoor", "notes", "/tmp/loc")
    sc_id = db.add_scene(ep_id, 1, "Opening", "Actor walks in", char_id, loc_id, "Close Up")

    # PHASE 10 additions
    from studio.intelligence.script_analyzer import ScriptAnalyzer
    from studio.intelligence.continuity_engine import ContinuityEngine
    
    logger.info("Running Script Analyzer...")
    db.set_continuity_state(sc_id, loc_id, wardrobe_tag="red_jacket", weather="raining")
    db.add_approved_asset(char_id, "image", "/tmp/actor/primary.png", True)
    db.execute_custom_query("INSERT INTO character_memory (character_id, visual_style) VALUES (?, ?)", (char_id, "cinematic lighting"))
    
    analyzer = ScriptAnalyzer(db)
    dialogue = "E2E Actor: We need to test the pipeline.\nE2E Actor: It's critical."
    analyzer.analyze_scene_to_shots(sc_id, dialogue, "Actor stands in rain.")
    
    shots = db.list_shots(sc_id)
    assert len(shots) == 3, f"Expected 3 shots, got {len(shots)}"
    logger.info("Script Analyzer generated 3 shots successfully!")
    
    ce = ContinuityEngine(db)
    ctx = ce.build_generation_context(sc_id, char_id, shots[0]["description"])
    
    logger.info(f"Enriched prompt: {ctx['enriched_prompt']}")
    assert "raining" in ctx["enriched_prompt"], "Continuity not applied!"
    assert "/tmp/actor/primary.png" in ctx["reference_images"], "Character memory not applied!"
    
    # 2. Queue Render Jobs using the Continuity output
    logger.info("Queuing Jobs...")
    vid_job_id = db.add_to_render_queue(
        project_id=proj_id, scene_id=sc_id, character_id=char_id, location_id=loc_id,
        model_family="LTX-Video", pipeline="distilled", preset="draft", prompt=ctx["enriched_prompt"],
        negative_prompt="blurry", seed=12345, width=768, height=512, fps=24, num_frames=65,
        steps=20, guidance_scale=3.0, reference_image_path=ctx["reference_images"][0]
    )

    aud_job_id = db.add_to_audio_queue(
        project_id=proj_id, episode_id=ep_id, scene_id=sc_id, character_id=char_id,
        dialogue_text="Hello world, this is an end to end test.", voice_id="am_adam"
    )

    # Mock LTX generate_video function directly
    ltx_manager.generate_video = lambda *args, **kwargs: (["mock_frame"], 12345, None)
    ltx_manager._cached_state = {"pipeline": "mock", "device": "cpu"}

    # 3. Start Worker
    logger.info("Starting Worker Thread...")
    worker.start_render_worker(db, studio_root)

    # 4. Wait for processing
    timeout = 60
    start_time = time.time()
    
    vid_completed = False
    aud_completed = False

    while time.time() - start_time < timeout:
        vid_job = db.get_render_job(vid_job_id)
        aud_job = db.get_audio_job(aud_job_id)

        if vid_job["status"] == "completed" and not vid_completed:
            logger.info("✅ Video Job Completed!")
            vid_completed = True
        elif vid_job["status"] == "failed":
            logger.error(f"❌ Video Job Failed: {vid_job.get('error_message')}")
            break

        if aud_job["status"] == "completed" and not aud_completed:
            logger.info("✅ Audio Job Completed!")
            aud_completed = True
        elif aud_job["status"] == "failed":
            logger.error(f"❌ Audio Job Failed: {aud_job.get('error_message')}")
            break

        if vid_completed and aud_completed:
            break

        time.sleep(2)

    # Assertions
    assert vid_completed, "Video job did not complete successfully."
    assert aud_completed, "Audio job did not complete successfully."
    
    logger.info("🎉 End-to-End Pipeline Test Passed Successfully!")

    # Cleanup
    try:
        os.remove("test_e2e.db")
    except:
        pass

if __name__ == "__main__":
    test_full_pipeline()
