import os
import sys
from studio.database.db_manager import DatabaseManager

def test_db_phase6():
    db = DatabaseManager()
    
    # Test Wan Config
    db.save_wan_config(wan_model_path="dummy_wan")
    cfg = db.get_wan_config()
    assert cfg["wan_model_path"] == "dummy_wan", "save_wan_config failed"
    print("Wan Config test passed")

    # Test Hunyuan Config
    db.save_hunyuan_config(hunyuan_model_path="dummy_hunyuan")
    cfg = db.get_hunyuan_config()
    assert cfg["hunyuan_model_path"] == "dummy_hunyuan", "save_hunyuan_config failed"
    print("Hunyuan Config test passed")

    # Test Add Video with model_family
    vid_id = db.add_generated_video(
        file_path="dummy_wan.mp4",
        model_family="Wan 2.2",
        pipeline="text2video"
    )
    assert vid_id is not None, "add_generated_video failed"

    # Get Video
    vid = db.get_generated_video(vid_id)
    assert vid["model_family"] == "Wan 2.2", "model_family retrieval failed"
    
    # List Videos with filter
    db.update_video_status(vid_id, "approved")
    vids = db.list_generated_videos(model_family="Wan 2.2")
    assert any(v["id"] == vid_id for v in vids), "list_generated_videos filter failed"

    print("Phase 6 Video CRUD test passed")

if __name__ == "__main__":
    test_db_phase6()
