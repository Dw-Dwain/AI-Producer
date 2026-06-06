import os
import sys
from studio.database.db_manager import DatabaseManager

def test_db():
    db = DatabaseManager()
    
    # test save ltx config
    db.save_ltx_config(ltx_model_path="dummy_path")
    cfg = db.get_ltx_config()
    assert cfg["ltx_model_path"] == "dummy_path", "save_ltx_config failed"
    print("LTX Config test passed")

    # test add video
    vid_id = db.add_generated_video(
        file_path="dummy.mp4",
        pipeline="distilled"
    )
    assert vid_id is not None, "add_generated_video failed"
    
    # get video
    vid = db.get_generated_video(vid_id)
    assert vid["file_path"] == "dummy.mp4", "get_generated_video failed"

    # update status
    db.update_video_status(vid_id, "approved")
    vid = db.get_generated_video(vid_id)
    assert vid["status"] == "approved", "update_video_status failed"

    # list videos
    vids = db.list_generated_videos(status="approved")
    assert any(v["id"] == vid_id for v in vids), "list_generated_videos failed"

    # delete video
    db.delete_generated_video(vid_id)
    vid = db.get_generated_video(vid_id)
    assert vid is None, "delete_generated_video failed"

    print("Video CRUD test passed")

if __name__ == "__main__":
    test_db()
