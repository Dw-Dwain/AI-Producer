import os
import sys
import shutil
import tempfile
import json
from studio.database.db_manager import DatabaseManager
from studio.generation.lipsync_manager import LipSyncManager

def test_lipsync_pipeline():
    print("=== STARTING LIP SYNC PIPELINE TESTS ===")
    
    # 1. Setup a test database
    temp_db_path = os.path.join(tempfile.gettempdir(), "test_lipsync.db")
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)
        
    db = DatabaseManager(db_path=temp_db_path)
    print("Database manager initialized with test database.")

    # 2. Check defaults are populated
    settings = db.get_all_settings()
    assert "lipsync_wav2lip_python_path" in settings, "Wav2Lip python path setting missing"
    assert "lipsync_wav2lip_code_dir" in settings, "Wav2Lip code dir setting missing"
    assert "lipsync_wav2lip_checkpoint_path" in settings, "Wav2Lip checkpoint setting missing"
    assert "lipsync_wav2lip_status" in settings, "Wav2Lip status setting missing"
    
    print("Default settings key population validated.")
    print(f"Detected Wav2Lip Status: {settings['lipsync_wav2lip_status']}")

    # 3. Create a temporary project directory to act as studio root
    temp_studio_root = tempfile.mkdtemp()
    print(f"Created temporary studio root: {temp_studio_root}")

    try:
        # Create fake source video and audio
        fake_video_dir = os.path.join(temp_studio_root, "output", "videos", "test_proj", "test_char")
        os.makedirs(fake_video_dir, exist_ok=True)
        
        fake_video_path = os.path.join(fake_video_dir, "source_vid.mp4")
        with open(fake_video_path, "w") as f:
            f.write("Fake Video Content")
            
        fake_audio_path = os.path.join(temp_studio_root, "audio.wav")
        with open(fake_audio_path, "w") as f:
            f.write("Fake Audio Content")

        # Create a mock Wav2Lip python engine script in a fake engine folder
        mock_engine_dir = os.path.join(temp_studio_root, "mock_wav2lip")
        os.makedirs(mock_engine_dir, exist_ok=True)
        
        mock_inference_script = os.path.join(mock_engine_dir, "inference.py")
        with open(mock_inference_script, "w") as f:
            f.write("""
import argparse
import shutil
import sys

print("Mock Wav2Lip inference tool started!")
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint")
parser.add_argument("--face")
parser.add_argument("--audio")
parser.add_argument("--outfile")
parser.add_argument("--custom_flag", action="store_true")
parser.add_argument("--custom_val")

args = parser.parse_args()
print(f"Arguments parsed: checkpoint={args.checkpoint}, face={args.face}, audio={args.audio}, outfile={args.outfile}")
print(f"Custom flag={args.custom_flag}, Custom val={args.custom_val}")

if not args.checkpoint or not args.face or not args.audio or not args.outfile:
    sys.exit(1)

# Simulate generating output by copying source video
shutil.copy2(args.face, args.outfile)
print("Mock Lip Sync output generated successfully!")
""")

        mock_checkpoint_path = os.path.join(mock_engine_dir, "wav2lip_gan.pth")
        with open(mock_checkpoint_path, "w") as f:
            f.write("Fake Model Weights")

        # 4. Inject mock engine configuration into settings
        db.set_setting("lipsync_wav2lip_python_path", sys.executable)
        db.set_setting("lipsync_wav2lip_code_dir", mock_engine_dir)
        db.set_setting("lipsync_wav2lip_checkpoint_path", mock_checkpoint_path)

        # Trigger settings refresh & assert engine is detected as Available
        refreshed_settings = db.get_all_settings()
        assert "Available" in refreshed_settings["lipsync_wav2lip_status"], f"Expected engine to be Available, got {refreshed_settings['lipsync_wav2lip_status']}"
        print("Engine detection and availability reporting verified successfully.")

        # 5. Populate dummy records for scene & source video to satisfy foreign keys
        with db._get_connection() as conn:
            conn.execute("INSERT INTO projects (id, name) VALUES (1, 'Test Project')")
            conn.execute("INSERT INTO episodes (id, project_id, episode_number, title) VALUES (1, 1, 1, 'Test Episode')")
            conn.execute("INSERT INTO scenes (id, episode_id, scene_number, title) VALUES (1, 1, 1, 'Test Scene')")
            conn.execute("INSERT INTO characters (id, name, folder_path) VALUES (1, 'Test Character', 'chars/test')")
            conn.commit()

        # Add source video to database asset manager
        source_video_id = db.add_generated_video(
            file_path=fake_video_path,
            pipeline="two_stage",
            model_family="LTX-Video",
            prompt="A character talking",
            width=768,
            height=512,
            fps=24,
            num_frames=48,
            project_id=1,
            episode_id=1,
            scene_id=1,
        )
        db.update_video_status(source_video_id, "approved")

        # 6. Queue a Lip Sync Job
        job_id = db.add_to_lip_sync_queue(
            project_id=1,
            episode_id=1,
            scene_id=1,
            character_id=1,
            source_video_id=source_video_id,
            source_video_path=fake_video_path,
            source_audio_path=fake_audio_path,
            engine="Wav2Lip",
            engine_config={"custom_flag": True, "custom_val": "hello_world"}
        )
        assert job_id is not None, "Failed to queue lipsync job"
        print(f"Queued lip sync job with ID {job_id}")

        # Check job is queued
        job = db.get_lipsync_job(job_id)
        assert job["status"] == "queued", f"Expected job status 'queued', got '{job['status']}'"

        # 7. Worker simulation: get next queued job
        lipsync_job = db.get_next_queued_lipsync_job()
        assert lipsync_job is not None, "Failed to fetch queued job"
        assert lipsync_job["id"] == job_id, "Fetched incorrect job ID"
        assert lipsync_job["status"] == "running", f"Expected claimed job status 'running', got '{lipsync_job['status']}'"
        print("Worker successfully claimed the lip sync job from the queue.")

        # 8. Process using LipSyncManager
        lipsync = LipSyncManager(temp_studio_root, db=db)
        engine_config = json.loads(lipsync_job["engine_config_json"])
        
        output_video_path, metadata = lipsync.process(
            engine=lipsync_job["engine"],
            input_video_path=lipsync_job["source_video_path"],
            input_audio_path=lipsync_job["source_audio_path"],
            engine_config=engine_config,
        )

        assert os.path.exists(output_video_path), "Output video path does not exist"
        assert metadata["execution"] == "runtime_subprocess", "Incorrect execution mode"
        assert metadata["returncode"] == 0, f"Subprocess returned non-zero code {metadata['returncode']}"
        print(f"Lipsync execution completed successfully. Output saved to {output_video_path}")

        # Write results and assert database updates
        db.add_lipsync_result(
            queue_job_id=lipsync_job["id"],
            engine=lipsync_job["engine"],
            input_audio_path=lipsync_job["source_audio_path"],
            input_video_path=lipsync_job["source_video_path"],
            output_video_path=output_video_path,
            metadata=metadata,
        )

        # Register video inside the asset system (generated_videos table)
        new_video_id = db.add_generated_video(
            file_path=output_video_path,
            pipeline="lipsync",
            model_family=lipsync_job["engine"],
            prompt="Lip synced output video",
            width=768,
            height=512,
            fps=24,
            num_frames=48,
            project_id=lipsync_job.get("project_id"),
            episode_id=lipsync_job.get("episode_id"),
            scene_id=lipsync_job.get("scene_id"),
            model_name=lipsync_job["engine"],
        )
        db.update_video_status(new_video_id, "approved")
        db.update_lipsync_job_status(lipsync_job["id"], "completed", output_video_path=output_video_path)

        # 9. Verify results stored in DB
        finished_job = db.get_lipsync_job(job_id)
        assert finished_job["status"] == "completed", f"Expected job status 'completed', got '{finished_job['status']}'"
        assert finished_job["output_video_path"] == output_video_path, "Output video path mismatch in DB"
        
        # Verify the new video is in the approved video list
        approved_videos = db.list_generated_videos(project_id=1, status="approved")
        assert any(v["id"] == new_video_id for v in approved_videos), "New lip synced video asset not found in approved assets"
        
        print("Database status updates and asset registration verified.")
        print("=== ALL PIPELINE TESTS PASSED ===")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_studio_root, ignore_errors=True)
        try:
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)
        except Exception:
            pass

if __name__ == "__main__":
    test_lipsync_pipeline()
