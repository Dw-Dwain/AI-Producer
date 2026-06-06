import os
import sys
from studio.database.db_manager import DatabaseManager

def test_render_queue():
    db = DatabaseManager()
    
    # 1. Test Shot Templates
    shots = db.list_shot_templates()
    assert len(shots) > 0, "Shot templates not populated."
    shot_id = shots[0]["id"]
    print("Shot templates passed.")

    # 2. Add to Render Queue
    job_id = db.add_to_render_queue(
        shot_type_id=shot_id,
        prompt="A testing prompt",
        model_family="LTX-Video",
        pipeline="distilled"
    )
    assert job_id is not None, "Failed to add to render queue."
    
    # 3. Check status is queued
    job = db.get_render_job(job_id)
    assert job["status"] == "queued", "Job status is not queued."
    
    # 4. Get next queued job
    next_job = db.get_next_queued_job()
    assert next_job is not None, "Failed to get next queued job."
    assert next_job["id"] == job_id, "Got wrong job ID."
    assert next_job["status"] == "running", "Job status is not running."
    print("Job claiming passed.")

    # 5. Complete job
    db.update_render_job_status(job_id, "completed", video_id=123)
    completed_job = db.get_render_job(job_id)
    assert completed_job["status"] == "completed", "Failed to update status."
    assert completed_job["video_id"] == 123, "Failed to store video ID."
    print("Job completion passed.")

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_render_queue()
