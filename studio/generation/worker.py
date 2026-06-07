import os
import time
import logging
import threading
import json
from studio.database.db_manager import DatabaseManager
from studio.generation import ltx_manager, wan_manager, hunyuan_manager, video_output_manager, tts_manager
from studio.generation.lipsync_manager import LipSyncManager

logger = logging.getLogger("studio.worker")

# Module-level singleton — avoids reloading the Kokoro model on every audio job
_tts_manager_instance: tts_manager.TTSManager | None = None

def _get_tts_manager(studio_root: str) -> tts_manager.TTSManager:
    global _tts_manager_instance
    if _tts_manager_instance is None:
        _tts_manager_instance = tts_manager.TTSManager(os.path.join(studio_root, "audio_assets"))
    return _tts_manager_instance

def recover_unfinished_jobs(db: DatabaseManager):
    """Scan and recover any render, audio, or lip sync jobs that were left running due to a crash."""
    logger.info("Checking for unfinished or interrupted queue jobs to recover...")
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            
            # Recover render_queue
            cursor.execute("SELECT COUNT(*) FROM render_queue WHERE status = 'running'")
            renders = cursor.fetchone()[0]
            if renders > 0:
                cursor.execute("UPDATE render_queue SET status = 'queued', started_at = NULL WHERE status = 'running'")
                logger.info(f"Recovered {renders} interrupted render queue jobs.")
                
            # Recover audio_queue
            cursor.execute("SELECT COUNT(*) FROM audio_queue WHERE status = 'running'")
            audios = cursor.fetchone()[0]
            if audios > 0:
                cursor.execute("UPDATE audio_queue SET status = 'queued', started_at = NULL WHERE status = 'running'")
                logger.info(f"Recovered {audios} interrupted audio queue jobs.")
                
            # Recover lip_sync_queue
            cursor.execute("SELECT COUNT(*) FROM lip_sync_queue WHERE status = 'running'")
            lipsyncs = cursor.fetchone()[0]
            if lipsyncs > 0:
                cursor.execute("UPDATE lip_sync_queue SET status = 'queued', started_at = NULL WHERE status = 'running'")
                logger.info(f"Recovered {lipsyncs} interrupted lip sync queue jobs.")
                
            conn.commit()
    except Exception as e:
        logger.error(f"Error during queue recovery: {e}")

def render_worker_loop(db: DatabaseManager, studio_root: str):
    logger.info("Render worker started. Polling for queued jobs...")
    
    # Run database queue recovery
    recover_unfinished_jobs(db)
    
    lipsync = LipSyncManager(studio_root, db=db)
    while True:
        try:
            job = db.get_next_queued_job()
            if job:
                logger.info(f"Picked up job {job['id']} for Model Family {job['model_family']}")
                
                # Fetch shot prompt additions if applicable
                shot_prompt_addition = ""
                shot_neg_addition = ""
                if job.get("shot_type_id"):
                    with db._get_connection() as conn:
                        st = conn.execute("SELECT * FROM shot_templates WHERE id = ?", (job["shot_type_id"],)).fetchone()
                        if st:
                            shot_prompt_addition = st["prompt_addition"] or ""
                            shot_neg_addition = st["negative_prompt_addition"] or ""

                final_prompt = f"{job['prompt']}, {shot_prompt_addition}".strip(", ")
                final_neg = f"{job['negative_prompt']}, {shot_neg_addition}".strip(", ")

                family = job["model_family"]
                mgr = ltx_manager if family == "LTX-Video" else (wan_manager if family == "Wan 2.2" else hunyuan_manager)
                
                # Unload other model families to protect VRAM usage
                if family == "LTX-Video":
                    if wan_manager.get_loaded_pipeline_name() is not None:
                        logger.info("Pipeline switch detected... Model family changed from Wan 2.2 to LTX-Video. Unloading Wan 2.2 pipeline.")
                        wan_manager.unload_pipeline()
                    if hunyuan_manager.get_loaded_pipeline_name() is not None:
                        logger.info("Pipeline switch detected... Model family changed from Hunyuan Video to LTX-Video. Unloading Hunyuan Video pipeline.")
                        hunyuan_manager.unload_pipeline()
                elif family == "Wan 2.2":
                    if ltx_manager.get_loaded_pipeline_name() is not None:
                        logger.info("Pipeline switch detected... Model family changed from LTX-Video to Wan 2.2. Unloading LTX-Video pipeline.")
                        ltx_manager.unload_pipeline()
                    if hunyuan_manager.get_loaded_pipeline_name() is not None:
                        logger.info("Pipeline switch detected... Model family changed from Hunyuan Video to Wan 2.2. Unloading Hunyuan Video pipeline.")
                        hunyuan_manager.unload_pipeline()
                elif family == "Hunyuan Video":
                    if ltx_manager.get_loaded_pipeline_name() is not None:
                        logger.info("Pipeline switch detected... Model family changed from LTX-Video to Hunyuan Video. Unloading LTX-Video pipeline.")
                        ltx_manager.unload_pipeline()
                    if wan_manager.get_loaded_pipeline_name() is not None:
                        logger.info("Pipeline switch detected... Model family changed from Wan 2.2 to Hunyuan Video. Unloading Wan 2.2 pipeline.")
                        wan_manager.unload_pipeline()

                loaded_pipe_name = mgr.get_loaded_pipeline_name()
                state = mgr.get_state()
                is_mock = isinstance(state, dict) and state.get("pipeline") == "mock"

                if is_mock:
                    logger.info("Using cached model... (Mock state detected for testing)")
                else:
                    # Check if pipeline name mismatched or if not currently loaded
                    if not state or not loaded_pipe_name or loaded_pipe_name != job["pipeline"]:
                        if state:
                            logger.info(f"Pipeline switch detected... (cached: {loaded_pipe_name}, requested: {job['pipeline']}). Unloading existing pipeline.")
                            mgr.unload_pipeline()
                            state = None
                        
                        logger.info(f"Loading model... for {family} with pipeline {job['pipeline']}.")
                        try:
                            if family == "LTX-Video":
                                config = db.get_ltx_config()
                                ltx_path = config.get("ltx_model_path", "")
                                if not ltx_path:
                                    raise ValueError("LTX Model Path is empty in database configuration.")
                                upscaler_path = config.get("ltx_upscaler_path", "")
                                gemma_path = config.get("ltx_gemma_path", "")
                                device = config.get("ltx_device", "cuda")
                                dtype = config.get("ltx_dtype", "bfloat16")
                                lora_path = job.get("lora_path") or ""
                                if lora_path == "None":
                                    lora_path = ""
                                lora_wt = 1.0
                                if job.get("lora_weight") is not None:
                                    try:
                                        lora_wt = float(job["lora_weight"])
                                    except ValueError:
                                        pass

                                state, msg = mgr.load_pipeline(
                                    pipeline_name=job["pipeline"],
                                    ltx_path=ltx_path,
                                    upscaler_path=upscaler_path,
                                    gemma_path=gemma_path,
                                    device=device,
                                    dtype_str=dtype,
                                    lora_path=lora_path,
                                    lora_weight=lora_wt
                                )
                            elif family == "Wan 2.2":
                                config = db.get_wan_config()
                                wan_path = config.get("wan_model_path", "")
                                if not wan_path:
                                    raise ValueError("Wan Model Path is empty in database configuration.")
                                device = config.get("wan_device", "cuda")
                                dtype = config.get("wan_dtype", "bfloat16")

                                state, msg = mgr.load_pipeline(
                                    pipeline_name=job["pipeline"],
                                    wan_path=wan_path,
                                    device=device,
                                    dtype_str=dtype
                                )
                            elif family == "Hunyuan Video":
                                config = db.get_hunyuan_config()
                                hun_path = config.get("hunyuan_model_path", "")
                                if not hun_path:
                                    raise ValueError("Hunyuan Model Path is empty in database configuration.")
                                device = config.get("hunyuan_device", "cuda")
                                dtype = config.get("hunyuan_dtype", "bfloat16")

                                state, msg = mgr.load_pipeline(
                                    pipeline_name=job["pipeline"],
                                    hunyuan_path=hun_path,
                                    device=device,
                                    dtype_str=dtype
                                )
                            else:
                                raise ValueError(f"Unknown model family: {family}")

                            if not state:
                                raise Exception(f"Model load failed... Manager returned empty state. Details: {msg}")
                            
                            logger.info(f"Model loaded successfully... Details: {msg}")
                        except Exception as exc:
                            logger.error(f"Model load failed... Error: {exc}", exc_info=True)
                            db.update_render_job_status(job["id"], "failed", error_message=f"Model load failed: {exc}")
                            continue
                    else:
                        logger.info(f"Using cached model... (cached: {loaded_pipe_name})")
                
                frames, actual_seed, err = mgr.generate_video(
                    state=state, prompt=final_prompt, negative_prompt=final_neg, 
                    width=job["width"], height=job["height"],
                    fps=job["fps"], num_frames=job["num_frames"], steps=job["steps"], guidance_scale=job["guidance_scale"],
                    seed=job["seed"], pipeline_name=job["pipeline"], reference_image=job["reference_image_path"]
                )

                if err or not frames:
                    logger.error(f"Generation failed for job {job['id']}: {err}")
                    db.update_render_job_status(job["id"], "failed", error_message=err or "Unknown generation error.")
                    continue

                # Save video
                out_dir = video_output_manager.get_video_output_dir(studio_root, job.get("project_name") or "", job.get("character_name") or "")
                meta = video_output_manager.build_video_metadata(
                    final_prompt, final_neg, actual_seed, job["pipeline"], job["preset"], 
                    job["width"], job["height"], job["fps"], job["num_frames"], job["steps"], job["guidance_scale"],
                    job["lora_path"], job["lora_weight"], "", job.get("character_name") or "", job.get("location_name") or "", job.get("project_name") or ""
                )
                meta["model_family"] = family
                mp4_path, thumb_path, duration = video_output_manager.save_video(frames, job["fps"], out_dir, meta)
                
                # Add to generated_videos (awaiting approval)
                vid_id = db.add_generated_video(
                    file_path=mp4_path, pipeline=job["pipeline"], model_family=family, prompt=final_prompt, negative_prompt=final_neg,
                    seed=actual_seed, width=job["width"], height=job["height"], fps=job["fps"], num_frames=job["num_frames"], steps=job["steps"], guidance_scale=job["guidance_scale"],
                    lora_path=job["lora_path"], lora_name=job.get("lora_name") or "", lora_weight=job["lora_weight"], thumbnail_path=thumb_path, duration_seconds=duration, preset=job["preset"],
                    project_id=job["project_id"], episode_id=job.get("episode_id"), scene_id=job.get("scene_id"), shot_id=job.get("shot_id"),
                    dialogue_line_id=job.get("dialogue_line_id"), character_id=job["character_id"], location_id=job["location_id"],
                    model_name=job.get("model_name") or family, references_json=job.get("references_json") or "[]",
                    performance_json=job.get("performance_json") or "{}", camera_preset=job.get("camera_preset") or "",
                    movement_preset=job.get("movement_preset") or "", lens_preset=job.get("lens_preset") or ""
                )
                try:
                    for ref in json.loads(job.get("references_json") or "[]"):
                        if isinstance(ref, dict) and ref.get("path"):
                            db.add_reference_graph_edge(vid_id, ref.get("type", "reference"), ref["path"])
                        elif isinstance(ref, str):
                            db.add_reference_graph_edge(vid_id, "reference", ref)
                except json.JSONDecodeError:
                    pass
                if job.get("reference_image_path"):
                    db.add_reference_graph_edge(vid_id, "primary_reference", job["reference_image_path"])
                
                db.update_render_job_status(job["id"], "completed", video_id=vid_id)
                logger.info(f"Job {job['id']} completed successfully! Video ID: {vid_id}")
                
            # Check for audio jobs
            audio_job = db.get_next_queued_audio_job()
            if audio_job:
                logger.info(f"Picked up audio job {audio_job['id']}")
                try:
                    tts = _get_tts_manager(studio_root)
                    audio_path = tts.generate_voice(
                        text=audio_job["dialogue_text"], 
                        voice_id=audio_job["voice_id"], 
                        speed=audio_job["speed"]
                    )
                    dialogue_line_id = audio_job.get("dialogue_line_id")
                    if not dialogue_line_id and audio_job.get("scene_id") and audio_job.get("character_id"):
                        existing_lines = db.list_dialogue_lines(audio_job["scene_id"])
                        dialogue_line_id = db.add_dialogue_line(
                            scene_id=audio_job["scene_id"],
                            character_id=audio_job["character_id"],
                            shot_id=audio_job.get("shot_id"),
                            sequence_order=len(existing_lines) + 1,
                            text=audio_job["dialogue_text"],
                            emotion=audio_job.get("emotion") or "neutral",
                            intensity=audio_job.get("intensity") or 5,
                            expression=audio_job.get("expression") or "neutral",
                            speaking_style=audio_job.get("speaking_style") or "natural",
                        )
                    db.add_audio_asset(
                        dialogue_line_id=dialogue_line_id,
                        file_path=audio_path,
                        project_id=audio_job.get("project_id"),
                        episode_id=audio_job.get("episode_id"),
                        scene_id=audio_job.get("scene_id"),
                        shot_id=audio_job.get("shot_id"),
                        character_id=audio_job.get("character_id"),
                        voice_id=audio_job.get("voice_id") or "",
                        emotion=audio_job.get("emotion") or "",
                        intensity=audio_job.get("intensity"),
                        speaking_style=audio_job.get("speaking_style") or "",
                    )
                    db.update_audio_job_status(audio_job["id"], "completed", file_path=audio_path)
                    logger.info(f"Audio job {audio_job['id']} completed: {audio_path}")
                except Exception as e:
                    logger.error(f"Audio generation failed: {e}")
                    db.update_audio_job_status(audio_job["id"], "failed", error_message=str(e))

            lipsync_job = db.get_next_queued_lipsync_job()
            if lipsync_job:
                logger.info(f"Picked up lip sync job {lipsync_job['id']} using {lipsync_job['engine']}")
                try:
                    engine_config = {}
                    if lipsync_job.get("engine_config_json"):
                        try:
                            engine_config = json.loads(lipsync_job["engine_config_json"])
                        except Exception:
                            pass

                    output_video_path, metadata = lipsync.process(
                        engine=lipsync_job["engine"],
                        input_video_path=lipsync_job["source_video_path"],
                        input_audio_path=lipsync_job["source_audio_path"],
                        engine_config=engine_config,
                    )
                    db.add_lipsync_result(
                        queue_job_id=lipsync_job["id"],
                        engine=lipsync_job["engine"],
                        input_audio_path=lipsync_job["source_audio_path"],
                        input_video_path=lipsync_job["source_video_path"],
                        output_video_path=output_video_path,
                        metadata=metadata,
                    )

                    # Save output to existing asset system (generated_videos table)
                    source_video = db.get_generated_video(lipsync_job["source_video_id"]) if lipsync_job.get("source_video_id") else None
                    
                    thumbnail_path = None
                    try:
                        import imageio
                        from PIL import Image
                        reader = imageio.get_reader(output_video_path)
                        first_frame = reader.get_data(0)
                        reader.close()
                        
                        img = Image.fromarray(first_frame)
                        img.thumbnail((320, 320))
                        
                        out_dir = os.path.dirname(output_video_path)
                        base_name = os.path.splitext(os.path.basename(output_video_path))[0]
                        thumbnail_path = os.path.join(out_dir, f"{base_name}_thumb.png")
                        img.save(thumbnail_path, format="PNG")
                    except Exception as thumb_err:
                        logger.warning(f"Could not generate thumbnail for lipsynced video: {thumb_err}")
                    
                    duration_seconds = source_video.get("duration_seconds") if source_video else None
                    if not duration_seconds:
                        try:
                            import imageio
                            reader = imageio.get_reader(output_video_path)
                            meta_reader = reader.get_meta_data()
                            duration_seconds = meta_reader.get("duration", 3.0)
                            reader.close()
                        except Exception:
                            duration_seconds = 3.0

                    new_video_id = db.add_generated_video(
                        file_path=output_video_path,
                        pipeline="lipsync",
                        model_family=lipsync_job["engine"],
                        prompt=source_video.get("prompt") if source_video else "Lip synced output video",
                        negative_prompt=source_video.get("negative_prompt") if source_video else "",
                        seed=source_video.get("seed", -1) if source_video else -1,
                        width=source_video.get("width") if source_video else None,
                        height=source_video.get("height") if source_video else None,
                        fps=source_video.get("fps", 24) if source_video else 24,
                        num_frames=source_video.get("num_frames", 97) if source_video else 97,
                        steps=source_video.get("steps", 20) if source_video else 20,
                        guidance_scale=source_video.get("guidance_scale", 3.5) if source_video else 3.5,
                        lora_path=source_video.get("lora_path") if source_video else "",
                        lora_name=source_video.get("lora_name") if source_video else "",
                        lora_weight=source_video.get("lora_weight", 1.0) if source_video else 1.0,
                        reference_image_path=source_video.get("reference_image_path") if source_video else "",
                        thumbnail_path=thumbnail_path,
                        duration_seconds=duration_seconds,
                        preset=source_video.get("preset") if source_video else "",
                        project_id=lipsync_job.get("project_id"),
                        episode_id=lipsync_job.get("episode_id"),
                        scene_id=lipsync_job.get("scene_id"),
                        shot_id=lipsync_job.get("shot_id"),
                        dialogue_line_id=lipsync_job.get("dialogue_line_id"),
                        character_id=lipsync_job.get("character_id"),
                        location_id=source_video.get("location_id") if source_video else None,
                        model_name=lipsync_job["engine"],
                        camera_preset=source_video.get("camera_preset") if source_video else "",
                        movement_preset=source_video.get("movement_preset") if source_video else "",
                        lens_preset=source_video.get("lens_preset") if source_video else "",
                    )
                    db.update_video_status(new_video_id, "approved")

                    db.update_lipsync_job_status(lipsync_job["id"], "completed", output_video_path=output_video_path)
                except Exception as e:
                    logger.error(f"Lip sync failed: {e}")
                    db.update_lipsync_job_status(lipsync_job["id"], "failed", error_message=str(e))

            if not job and not audio_job and not lipsync_job:
                time.sleep(3) # Polling interval
        except Exception as e:
            logger.error(f"Worker loop encountered an error: {e}", exc_info=True)
            time.sleep(5)

def start_render_worker(db: DatabaseManager, studio_root: str):
    t = threading.Thread(target=render_worker_loop, args=(db, studio_root), name="RenderWorker", daemon=True)
    t.start()
