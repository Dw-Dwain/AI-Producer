"""
Primary production workspace driven by Project -> Episode -> Scene -> Shot.
"""
import random
import gradio as gr
from studio.database.db_manager import DatabaseManager
from studio.intelligence.script_analyzer import ScriptAnalyzer
from studio.intelligence.continuity_engine import ContinuityEngine
from studio.intelligence.consistency_engine import ConsistencyEngine
from studio.intelligence.performance_engine import PerformanceEngine


_PRESETS = {
    "draft": {"width": 768, "height": 512, "fps": 24, "num_frames": 65, "steps": 20, "guidance_scale": 3.0, "pipeline": "distilled"},
    "production": {"width": 1024, "height": 576, "fps": 24, "num_frames": 97, "steps": 40, "guidance_scale": 3.5, "pipeline": "two_stage"},
    "cinema": {"width": 1280, "height": 720, "fps": 24, "num_frames": 129, "steps": 50, "guidance_scale": 4.0, "pipeline": "text2video"},
}


def create_workspace_tab(db: DatabaseManager):
    with gr.TabItem("Studio Workspace"):
        active_scene_id = gr.State(None)
        active_shot_id = gr.State(None)
        active_episode_id = gr.State(None)

        with gr.Row():
            with gr.Column(scale=2, variant="panel"):
                gr.Markdown("### Story Context")
                with gr.Row():
                    ws_proj = gr.Dropdown(label="Project", choices=[], interactive=True)
                    ws_ep = gr.Dropdown(label="Episode", choices=[], interactive=True)
                    ws_sc = gr.Dropdown(label="Scene", choices=[], interactive=True)

                ws_scene_desc = gr.Markdown("*Select a scene to inspect its plan.*")
                ws_script = gr.Textbox(label="Scene Script", lines=6, placeholder="Aki: I trusted you.\nRen: Then trust me one more time.")
                btn_analyze_script = gr.Button("Analyze Script to Shot List", variant="secondary")

                gr.Markdown("### Shot Plan")
                ws_shot = gr.Dropdown(label="Shot", choices=[], interactive=True)
                ws_shot_list_df = gr.Dataframe(
                    headers=["Shot #", "Type", "Character", "Goal", "Status"],
                    datatype=["number", "str", "str", "str", "str"],
                    interactive=False,
                )
                btn_generate_scene = gr.Button("Generate Entire Scene", variant="primary")

                gr.Markdown("### Scene Continuity")
                continuity_box = gr.Markdown("*Continuity notes will appear here.*")
                consistency_box = gr.Markdown("*Consistency score will appear here.*")

                with gr.Row():
                    ws_char = gr.Dropdown(label="Character Override", choices=[], interactive=True)
                    ws_loc = gr.Dropdown(label="Location Override", choices=[], interactive=True)

                ws_dialogue = gr.Textbox(label="Dialogue Line", lines=2, placeholder="Optional dialogue for direct voice generation")
                ws_voice_id = gr.Textbox(label="Voice ID", value="af_heart")

            with gr.Column(scale=3, variant="panel"):
                gr.Markdown("### Generation Canvas")
                ws_video_player = gr.Video(label="Selected / Completed Video", interactive=False, height=380)
                ws_audio_player = gr.Audio(label="Generated Audio", interactive=False)

                with gr.Row():
                    ws_family = gr.Radio(label="Model Family", choices=["LTX-Video", "Wan 2.2", "Hunyuan Video"], value="LTX-Video")
                    ws_preset = gr.Radio(label="Quality Preset", choices=["draft", "production", "cinema", "custom"], value="draft")

                with gr.Row():
                    ws_pipeline = gr.Dropdown(label="Pipeline", choices=["distilled", "two_stage", "two_stage_hq", "text2video", "image2video"], value="distilled")
                    ws_seed = gr.Number(label="Seed", value=-1, precision=0)
                    ws_lipsync_engine = gr.Dropdown(label="Lip Sync Engine", choices=["Wav2Lip", "SyncTalk", "MuseTalk"], value="Wav2Lip")

                with gr.Row():
                    ws_w = gr.Slider(label="Width", minimum=256, maximum=1920, step=32, value=768)
                    ws_h = gr.Slider(label="Height", minimum=256, maximum=1920, step=32, value=512)
                with gr.Row():
                    ws_fps = gr.Slider(label="FPS", minimum=8, maximum=30, step=1, value=24)
                    ws_nf = gr.Slider(label="Frames", minimum=9, maximum=257, step=8, value=65)
                with gr.Row():
                    ws_steps = gr.Slider(label="Steps", minimum=1, maximum=60, step=1, value=20)
                    ws_cfg = gr.Slider(label="CFG", minimum=1.0, maximum=10.0, step=0.5, value=3.0)
                    ws_lora_wt = gr.Slider(label="LoRA Weight", minimum=0.0, maximum=1.5, step=0.05, value=1.0)

                ws_prompt = gr.Textbox(label="Shot Prompt", lines=5, placeholder="Auto-filled from shot plan and continuity.")
                ws_neg = gr.Textbox(label="Negative Prompt", lines=2)
                with gr.Row():
                    btn_queue = gr.Button("Queue Selected Shot", variant="primary")
                    btn_queue_audio = gr.Button("Queue Voice", variant="secondary")
                    btn_queue_lipsync = gr.Button("Queue Lip Sync", variant="secondary")
                ws_action_msg = gr.Markdown("")

            with gr.Column(scale=2, variant="panel"):
                with gr.Tabs():
                    with gr.TabItem("Video Queue"):
                        btn_q_refresh = gr.Button("Refresh")
                        ws_queue_df = gr.Dataframe(headers=["ID", "Status", "Character", "Shot", "Model"], datatype=["number", "str", "str", "str", "str"], interactive=False)
                        ws_q_sel_id = gr.Number(value=-1, visible=False)
                    with gr.TabItem("Audio Queue"):
                        btn_a_refresh = gr.Button("Refresh")
                        ws_audio_queue_df = gr.Dataframe(headers=["ID", "Status", "Character", "Text"], datatype=["number", "str", "str", "str"], interactive=False)
                        ws_a_sel_id = gr.Number(value=-1, visible=False)
                    with gr.TabItem("Lip Sync Queue"):
                        btn_l_refresh = gr.Button("Refresh")
                        ws_lipsync_df = gr.Dataframe(headers=["ID", "Status", "Engine", "Shot"], datatype=["number", "str", "str", "str"], interactive=False)

                gr.Markdown("### Approved Scene Assets")
                ws_scene_gallery = gr.Gallery(label="Approved Shots", columns=2, object_fit="cover", height=280)

    def _scene_lookup(proj_name, ep_val, sc_val):
        if not proj_name or not ep_val or not sc_val:
            return None, None, None
        project = db.get_project_by_name(proj_name)
        if not project:
            return None, None, None
        ep_num = int(ep_val.split(":")[0].replace("Ep ", "").strip())
        sc_num = int(sc_val.split(":")[0].replace("Scene ", "").strip())
        episode = db.get_episode_by_details(project["id"], ep_num)
        if not episode:
            return project, None, None
        scene = next((s for s in db.list_scenes(episode["id"]) if s["scene_number"] == sc_num), None)
        return project, episode, scene

    def _shot_label(shot):
        return f"Shot {shot['shot_number']}: {shot.get('title') or shot.get('description') or shot.get('shot_type') or 'Untitled'}"

    def _refresh_queue_tables():
        video_rows = [
            [j["id"], j.get("status", ""), j.get("character_name", ""), j.get("shot_id") or "—", j.get("model_family", "")]
            for j in db.list_render_jobs(limit=25)
        ]
        audio_rows = [
            [j["id"], j.get("status", ""), j.get("character_name", ""), j.get("dialogue_text", "")]
            for j in db.list_audio_jobs(limit=25)
        ]
        lipsync_rows = [
            [j["id"], j.get("status", ""), j.get("engine", ""), j.get("shot_id") or "—"]
            for j in db.list_lipsync_jobs(limit=25)
        ]
        return video_rows, audio_rows, lipsync_rows

    def on_family_change(family):
        return gr.update(choices=["distilled", "two_stage", "two_stage_hq"], value="distilled") if family == "LTX-Video" else gr.update(choices=["text2video", "image2video"], value="text2video")

    def on_preset_change(preset_val, pipe_val):
        if preset_val == "custom":
            return (gr.update(),) * 7
        preset = _PRESETS.get(preset_val, _PRESETS["draft"])
        pipeline = preset["pipeline"] if pipe_val in ["distilled", "two_stage", "two_stage_hq"] else "text2video"
        return (
            gr.update(value=preset["width"]),
            gr.update(value=preset["height"]),
            gr.update(value=preset["fps"]),
            gr.update(value=preset["num_frames"]),
            gr.update(value=preset["steps"]),
            gr.update(value=preset["guidance_scale"]),
            gr.update(value=pipeline),
        )

    def on_proj_change(proj_name):
        if not proj_name:
            return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
        project = db.get_project_by_name(proj_name)
        episodes = db.list_episodes(project["id"]) if project else []
        labels = [f"Ep {ep['episode_number']}: {ep['title'] or 'Untitled'}" for ep in episodes]
        return gr.update(choices=labels, value=labels[0] if labels else None), gr.update(choices=[], value=None)

    def on_ep_change(proj_name, ep_val):
        if not proj_name or not ep_val:
            return gr.update(choices=[], value=None)
        project = db.get_project_by_name(proj_name)
        ep_num = int(ep_val.split(":")[0].replace("Ep ", "").strip())
        episode = db.get_episode_by_details(project["id"], ep_num) if project else None
        scenes = db.list_scenes(episode["id"]) if episode else []
        labels = [f"Scene {scene['scene_number']}: {scene['title'] or 'Untitled'}" for scene in scenes]
        return gr.update(choices=labels, value=labels[0] if labels else None)

    def on_scene_change(proj_name, ep_val, sc_val):
        project, episode, scene = _scene_lookup(proj_name, ep_val, sc_val)
        if not scene:
            return None, None, "*Select a scene.*", "", gr.update(choices=[], value=None), [], gr.update(value=None), gr.update(value=None), [], "*", "*"
        shot_rows = db.list_shots(scene["id"])
        shot_labels = [_shot_label(shot) for shot in shot_rows]
        dialogue_lines = db.list_dialogue_lines(scene["id"])
        script_text = "\n".join(f"{line['character_name']}: {line['text']}" for line in dialogue_lines)
        approved = [v for v in db.list_generated_videos(project_id=project["id"], status="approved", limit=1000) if v.get("scene_id") == scene["id"]]
        gallery = [(v.get("thumbnail_path") or v.get("file_path"), f"Shot {v.get('shot_id') or v['id']}") for v in approved]
        shot_df = [[shot["shot_number"], shot.get("shot_type", ""), shot.get("character_name", ""), shot.get("goal", ""), shot.get("status", "")] for shot in shot_rows]
        continuity = db.get_continuity_state(scene["id"]) or {}
        continuity_md = "\n".join([
            f"- Time: {continuity.get('time_of_day') or 'unset'}",
            f"- Weather: {continuity.get('weather') or 'unset'}",
            f"- Wardrobe: {continuity.get('wardrobe_tag') or 'unset'}",
            f"- Lighting: {continuity.get('lighting') or 'unset'}",
        ])
        return (
            episode["id"],
            scene["id"],
            f"**Scene {scene['scene_number']}**\n\n{scene.get('description') or '*No description.*'}",
            script_text,
            gr.update(choices=shot_labels, value=shot_labels[0] if shot_labels else None),
            shot_df,
            gr.update(value=scene.get("character_name")),
            gr.update(value=scene.get("location_name")),
            gallery,
            continuity_md,
            "*Select a shot to inspect consistency.*",
        )

    def on_shot_change(proj_name, ep_val, sc_val, shot_label):
        project, episode, scene = _scene_lookup(proj_name, ep_val, sc_val)
        if not scene or not shot_label:
            return None, "", "", "*", gr.update(value=""), gr.update(value="")
        shot_number = int(shot_label.split(":")[0].replace("Shot ", "").strip())
        shot = next((row for row in db.list_shots(scene["id"]) if row["shot_number"] == shot_number), None)
        if not shot:
            return None, "", "", "*", gr.update(value=""), gr.update(value="")
        continuity = ContinuityEngine(db)
        context = continuity.build_generation_context(scene["id"], shot.get("character_focus"), shot.get("prompt") or shot.get("description") or "", shot=shot)
        report = ConsistencyEngine(db).evaluate_shot(project["id"], episode["id"], scene, shot, shot.get("character_focus"))
        report_md = f"**Consistency Score:** {report['consistency_score']:.2f}  \n**Warnings:** {', '.join(report['warnings']) if report['warnings'] else 'None'}"
        return shot["id"], context["enriched_prompt"], shot.get("negative_prompt") or "", report_md, gr.update(value=shot.get("character_name")), gr.update(value=scene.get("location_name"))

    def on_analyze_script(proj_name, ep_val, sc_val, script_text):
        _, _, scene = _scene_lookup(proj_name, ep_val, sc_val)
        if not scene:
            return gr.update(), gr.update(), "Select a scene first."
        analyzer = ScriptAnalyzer(db)
        shots = analyzer.analyze_scene_to_shots(scene["id"], script_text, scene.get("description") or "")
        shot_labels = [_shot_label(shot) for shot in shots]
        shot_df = [[shot["shot_number"], shot.get("shot_type", ""), shot.get("character_name", ""), shot.get("goal", ""), shot.get("status", "")] for shot in db.list_shots(scene["id"])]
        return gr.update(choices=shot_labels, value=shot_labels[0] if shot_labels else None), gr.update(value=shot_df), f"Queued plan for {len(shots)} shots."

    def on_generate_scene(proj_name, ep_val, sc_val, family, preset, pipe, w, h, fps, nf, steps, cfg):
        project, episode, scene = _scene_lookup(proj_name, ep_val, sc_val)
        if not scene:
            return "Select a scene first."
        continuity = ContinuityEngine(db)
        performance = PerformanceEngine(db)
        shots = db.list_shots(scene["id"])
        if not shots:
            return "No shots found for this scene."
        render_count = 0
        audio_count = 0
        for shot in shots:
            dialogue_line = next((line for line in db.list_dialogue_lines(scene["id"]) if line.get("shot_id") == shot["id"]), None)
            context = continuity.build_generation_context(scene["id"], shot.get("character_focus"), shot.get("prompt") or shot.get("description") or "", shot=shot, dialogue_line=dialogue_line)
            performance_payload = context["performance"] or (performance.build_dialogue_payload(dialogue_line, shot.get("character_focus")) if dialogue_line else {})
            actual_seed = random.randint(0, 2**32 - 1)
            db.add_to_render_queue(
                project_id=project["id"],
                episode_id=episode["id"],
                scene_id=scene["id"],
                shot_id=shot["id"],
                dialogue_line_id=dialogue_line.get("id") if dialogue_line else None,
                character_id=shot.get("character_focus") or scene.get("character_id"),
                location_id=scene.get("location_id"),
                model_family=family,
                model_name=family,
                pipeline=pipe,
                preset=preset,
                prompt=context["enriched_prompt"],
                negative_prompt=shot.get("negative_prompt") or context["camera"].get("negative_prompt_addition", ""),
                seed=actual_seed,
                width=int(w),
                height=int(h),
                fps=int(fps),
                num_frames=int(nf),
                steps=int(steps),
                guidance_scale=float(cfg),
                reference_image_path=context["reference_images"][0] if context["reference_images"] else "",
                references_json=context["references_json"],
                performance_json=performance_payload.get("performance_json", "{}"),
                camera_preset=context["camera"].get("camera_language", ""),
                movement_preset=context["camera"].get("movement_preset", ""),
                lens_preset=context["camera"].get("lens_preset", ""),
                continuity_notes=context["continuity_notes"],
            )
            render_count += 1
            if dialogue_line:
                db.add_to_audio_queue(
                    project_id=project["id"],
                    episode_id=episode["id"],
                    scene_id=scene["id"],
                    shot_id=shot["id"],
                    character_id=dialogue_line["character_id"],
                    dialogue_line_id=dialogue_line["id"],
                    dialogue_text=dialogue_line["text"],
                    voice_id=performance_payload.get("voice_id", "af_heart"),
                    emotion=performance_payload.get("emotion", "neutral"),
                    intensity=performance_payload.get("intensity", 5),
                    expression=performance_payload.get("expression", "neutral"),
                    speaking_style=performance_payload.get("speaking_style", "natural"),
                )
                audio_count += 1
        return f"Queued {render_count} render jobs and {audio_count} audio jobs."

    def on_queue_selected_shot(proj_name, ep_val, sc_val, shot_label, char_name, loc_name, family, preset, pipe, w, h, fps, nf, steps, cfg, seed, lora_wt, prompt, neg):
        project, episode, scene = _scene_lookup(proj_name, ep_val, sc_val)
        if not scene or not shot_label:
            return "Select a scene and shot first.", _refresh_queue_tables()[0]
        shot_number = int(shot_label.split(":")[0].replace("Shot ", "").strip())
        shot = next((row for row in db.list_shots(scene["id"]) if row["shot_number"] == shot_number), None)
        if not shot:
            return "Shot not found.", _refresh_queue_tables()[0]
        character = db.get_character_by_name(char_name) if char_name else None
        location = db.get_location_by_name(loc_name) if loc_name else None
        actual_seed = int(seed) if int(seed) != -1 else random.randint(0, 2**32 - 1)
        db.add_to_render_queue(
            project_id=project["id"],
            episode_id=episode["id"],
            scene_id=scene["id"],
            shot_id=shot["id"],
            character_id=character["id"] if character else shot.get("character_focus"),
            location_id=location["id"] if location else scene.get("location_id"),
            model_family=family,
            model_name=family,
            pipeline=pipe,
            preset=preset,
            prompt=prompt,
            negative_prompt=neg,
            seed=actual_seed,
            width=int(w),
            height=int(h),
            fps=int(fps),
            num_frames=int(nf),
            steps=int(steps),
            guidance_scale=float(cfg),
            lora_weight=float(lora_wt),
            camera_preset=shot.get("camera_language") or shot.get("shot_type") or "",
            movement_preset=shot.get("movement_preset") or "",
            lens_preset=shot.get("lens_preset") or "",
            references_json=shot.get("references_json") or "[]",
        )
        return "Shot queued.", _refresh_queue_tables()[0]

    def on_queue_audio(proj_name, ep_val, sc_val, shot_label, char_name, dialogue, voice_id):
        project, episode, scene = _scene_lookup(proj_name, ep_val, sc_val)
        if not scene or not char_name or not dialogue:
            return "Select a scene, character, and dialogue line.", _refresh_queue_tables()[1]
        shot_id = None
        if shot_label:
            shot_number = int(shot_label.split(":")[0].replace("Shot ", "").strip())
            shot = next((row for row in db.list_shots(scene["id"]) if row["shot_number"] == shot_number), None)
            shot_id = shot["id"] if shot else None
        character = db.get_character_by_name(char_name)
        db.add_to_audio_queue(project["id"], episode["id"], scene["id"], character["id"], dialogue, voice_id, shot_id=shot_id)
        return "Voice job queued.", _refresh_queue_tables()[1]

    def on_queue_lipsync(proj_name, ep_val, sc_val, shot_label, engine):
        project, episode, scene = _scene_lookup(proj_name, ep_val, sc_val)
        if not scene or not shot_label:
            return "Select a scene and shot.", _refresh_queue_tables()[2]
        shot_number = int(shot_label.split(":")[0].replace("Shot ", "").strip())
        shot = next((row for row in db.list_shots(scene["id"]) if row["shot_number"] == shot_number), None)
        if not shot:
            return "Shot not found.", _refresh_queue_tables()[2]
        video = next((v for v in db.list_generated_videos(project_id=project["id"], status="approved", limit=1000) if v.get("shot_id") == shot["id"]), None)
        if not video:
            return "Approve a generated shot video before lip sync.", _refresh_queue_tables()[2]
        audio_job = next((job for job in db.list_audio_jobs(limit=200) if job.get("shot_id") == shot["id"] and job.get("file_path")), None)
        if not audio_job:
            return "Generate audio for this shot before lip sync.", _refresh_queue_tables()[2]
        db.add_to_lip_sync_queue(
            project_id=project["id"],
            episode_id=episode["id"],
            scene_id=scene["id"],
            shot_id=shot["id"],
            character_id=shot.get("character_focus"),
            source_video_id=video["id"],
            source_video_path=video["file_path"],
            source_audio_path=audio_job["file_path"],
            engine=engine,
        )
        return "Lip sync job queued.", _refresh_queue_tables()[2]

    def on_q_select(evt: gr.SelectData, df):
        job_id = int(df.iloc[evt.index[0], 0])
        job = db.get_render_job(job_id)
        if job and job.get("video_id"):
            video = db.get_generated_video(job["video_id"])
            return job_id, video.get("file_path") if video else None
        return job_id, None

    def on_a_select(evt: gr.SelectData, df):
        job_id = int(df.iloc[evt.index[0], 0])
        job = db.get_audio_job(job_id)
        return job_id, job.get("file_path") if job else None

    ws_family.change(on_family_change, [ws_family], [ws_pipeline])
    ws_preset.change(on_preset_change, [ws_preset, ws_pipeline], [ws_w, ws_h, ws_fps, ws_nf, ws_steps, ws_cfg, ws_pipeline])
    ws_proj.change(on_proj_change, [ws_proj], [ws_ep, ws_sc])
    ws_ep.change(on_ep_change, [ws_proj, ws_ep], [ws_sc])
    ws_sc.change(on_scene_change, [ws_proj, ws_ep, ws_sc], [active_episode_id, active_scene_id, ws_scene_desc, ws_script, ws_shot, ws_shot_list_df, ws_char, ws_loc, ws_scene_gallery, continuity_box, consistency_box])
    ws_shot.change(on_shot_change, [ws_proj, ws_ep, ws_sc, ws_shot], [active_shot_id, ws_prompt, ws_neg, consistency_box, ws_char, ws_loc])
    btn_analyze_script.click(on_analyze_script, [ws_proj, ws_ep, ws_sc, ws_script], [ws_shot, ws_shot_list_df, ws_action_msg])
    btn_generate_scene.click(on_generate_scene, [ws_proj, ws_ep, ws_sc, ws_family, ws_preset, ws_pipeline, ws_w, ws_h, ws_fps, ws_nf, ws_steps, ws_cfg], [ws_action_msg])
    btn_queue.click(on_queue_selected_shot, [ws_proj, ws_ep, ws_sc, ws_shot, ws_char, ws_loc, ws_family, ws_preset, ws_pipeline, ws_w, ws_h, ws_fps, ws_nf, ws_steps, ws_cfg, ws_seed, ws_lora_wt, ws_prompt, ws_neg], [ws_action_msg, ws_queue_df])
    btn_queue_audio.click(on_queue_audio, [ws_proj, ws_ep, ws_sc, ws_shot, ws_char, ws_dialogue, ws_voice_id], [ws_action_msg, ws_audio_queue_df])
    btn_queue_lipsync.click(on_queue_lipsync, [ws_proj, ws_ep, ws_sc, ws_shot, ws_lipsync_engine], [ws_action_msg, ws_lipsync_df])
    btn_q_refresh.click(lambda: _refresh_queue_tables()[0], [], [ws_queue_df])
    btn_a_refresh.click(lambda: _refresh_queue_tables()[1], [], [ws_audio_queue_df])
    btn_l_refresh.click(lambda: _refresh_queue_tables()[2], [], [ws_lipsync_df])
    ws_queue_df.select(on_q_select, [ws_queue_df], [ws_q_sel_id, ws_video_player])
    ws_audio_queue_df.select(on_a_select, [ws_audio_queue_df], [ws_a_sel_id, ws_audio_player])

    return ws_proj, ws_ep, ws_sc, ws_char, ws_loc, ws_shot
