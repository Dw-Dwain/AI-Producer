"""
studio/ui/video_tab.py
Phase 6: 🎬 Cinematic Video Generation Tab (LTX, Wan 2.2, Hunyuan).
"""

import os
import logging
import gradio as gr
from studio.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

STUDIO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_current_video_id: int | None = None

_PRESETS = {
    "draft": {
        "width": 768, "height": 512, "fps": 24, "num_frames": 65,
        "steps": 20, "guidance_scale": 3.0, "pipeline": "distilled",
    },
    "production": {
        "width": 1024, "height": 576, "fps": 24, "num_frames": 97,
        "steps": 40, "guidance_scale": 3.5, "pipeline": "two_stage",
    },
    "cinema": {
        "width": 1280, "height": 720, "fps": 24, "num_frames": 129,
        "steps": 50, "guidance_scale": 4.0, "pipeline": "text2video",
    },
}

def create_video_tab(db: DatabaseManager):
    with gr.TabItem("🎬 Video Generation"):
        with gr.Accordion("⚙️ Video Model Manager", open=False):
            gr.Markdown("Point to your **local** model folders or `.safetensors` files. No models are downloaded automatically.")
            with gr.Tabs():
                with gr.TabItem("LTX-Video"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            mm_ltx_path      = gr.Textbox(label="LTX-Video Model Path", placeholder="/path/to/ltx-video-2b", interactive=True)
                            mm_gemma_path    = gr.Textbox(label="Gemma Prompt Enhancer (optional)", placeholder="/path/to/gemma-2b", interactive=True)
                            mm_upscaler_path = gr.Textbox(label="Upscaler Model Path (required for Two-Stage)", placeholder="/path/to/ltx-upscaler", interactive=True)
                            mm_lora_dir      = gr.Textbox(label="LoRA Directory", placeholder="/path/to/loras/", interactive=True)
                        with gr.Column(scale=1):
                            mm_device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda", interactive=True)
                            mm_dtype  = gr.Dropdown(label="dtype", choices=["bfloat16", "float16", "float32"], value="bfloat16", interactive=True)
                    with gr.Row():
                        btn_detect_ltx    = gr.Button("🔍 Detect Models", variant="secondary")
                        btn_save_cfg_ltx  = gr.Button("💾 Save Paths", variant="secondary")
                        btn_load_pipe_ltx = gr.Button("⚡ Load LTX Pipeline", variant="primary")
                        btn_unload_ltx    = gr.Button("🗑️ Unload", variant="stop")
                    mm_status_html_ltx = gr.HTML("<div style='padding:8px;color:#9ca3af;font-size:0.85rem;'>LTX status...</div>")
                
                with gr.TabItem("Wan 2.2"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            mm_wan_path = gr.Textbox(label="Wan 2.2 Model Path", placeholder="/path/to/Wan2.2", interactive=True)
                        with gr.Column(scale=1):
                            mm_wan_device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda", interactive=True)
                            mm_wan_dtype  = gr.Dropdown(label="dtype", choices=["bfloat16", "float16", "float32"], value="bfloat16", interactive=True)
                    with gr.Row():
                        btn_detect_wan    = gr.Button("🔍 Detect Models", variant="secondary")
                        btn_save_cfg_wan  = gr.Button("💾 Save Paths", variant="secondary")
                        btn_load_pipe_wan = gr.Button("⚡ Load Wan Pipeline", variant="primary")
                        btn_unload_wan    = gr.Button("🗑️ Unload", variant="stop")
                    mm_status_html_wan = gr.HTML("<div style='padding:8px;color:#9ca3af;font-size:0.85rem;'>Wan status...</div>")

                with gr.TabItem("Hunyuan Video"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            mm_hunyuan_path = gr.Textbox(label="Hunyuan Model Path", placeholder="/path/to/HunyuanVideo", interactive=True)
                        with gr.Column(scale=1):
                            mm_hunyuan_device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda", interactive=True)
                            mm_hunyuan_dtype  = gr.Dropdown(label="dtype", choices=["bfloat16", "float16", "float32"], value="bfloat16", interactive=True)
                    with gr.Row():
                        btn_detect_hunyuan    = gr.Button("🔍 Detect Models", variant="secondary")
                        btn_save_cfg_hunyuan  = gr.Button("💾 Save Paths", variant="secondary")
                        btn_load_pipe_hunyuan = gr.Button("⚡ Load Hunyuan Pipeline", variant="primary")
                        btn_unload_hunyuan    = gr.Button("🗑️ Unload", variant="stop")
                    mm_status_html_hunyuan = gr.HTML("<div style='padding:8px;color:#9ca3af;font-size:0.85rem;'>Hunyuan status...</div>")

        with gr.Tabs():
            with gr.TabItem("🎬 Generate"):
                with gr.Row():
                    with gr.Column(scale=2, variant="panel"):
                        gr.Markdown("#### 🎭 Context")
                        with gr.Row():
                            vid_proj_selector = gr.Dropdown(label="Project", choices=[], value=None, interactive=True)
                            vid_char_selector = gr.Dropdown(label="Character", choices=[], value=None, interactive=True)
                            vid_loc_selector  = gr.Dropdown(label="Location", choices=[], value=None, interactive=True)

                        gr.Markdown("---")
                        gr.Markdown("#### ⚙️ Pipeline & Preset")
                        with gr.Row():
                            vid_model_family = gr.Radio(label="Model Family", choices=["LTX-Video", "Wan 2.2", "Hunyuan Video"], value="LTX-Video", interactive=True)
                        with gr.Row():
                            vid_pipeline = gr.Radio(label="Pipeline", choices=["distilled", "two_stage", "two_stage_hq"], value="distilled", interactive=True)
                            vid_preset = gr.Radio(label="Quick Preset", choices=["draft", "production", "cinema", "custom"], value="draft", interactive=True)

                        gr.Markdown("---")
                        gr.Markdown("#### ✍️ Prompt")
                        vid_prompt = gr.Textbox(label="Video Prompt", lines=4, placeholder="Describe the scene...", interactive=True)
                        with gr.Row():
                            btn_autofill   = gr.Button("🧬 Fill from Character DNA", variant="secondary")
                            btn_scene_fill = gr.Button("🎬 Load Scene Prompt", variant="secondary")
                        vid_neg_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="blurry, distorted...", interactive=True)

                        gr.Markdown("---")
                        gr.Markdown("#### 🖼️ Reference Image (optional)")
                        vid_ref_image = gr.Image(label="Reference / Starting Frame", type="pil", interactive=True, height=200)

                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("#### 🎛️ Parameters")
                        vid_width  = gr.Slider(label="Width", minimum=256, maximum=1920, step=32, value=768, interactive=True)
                        vid_height = gr.Slider(label="Height", minimum=256, maximum=1920, step=32, value=512, interactive=True)
                        vid_fps    = gr.Slider(label="FPS", minimum=8, maximum=30, step=1, value=24, interactive=True)
                        vid_frames = gr.Slider(label="Frames", minimum=9, maximum=257, step=8, value=65, interactive=True)
                        vid_steps  = gr.Slider(label="Steps", minimum=1, maximum=60, step=1, value=20, interactive=True)
                        vid_guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.5, value=3.0, interactive=True)

                        gr.Markdown("---")
                        gr.Markdown("#### 🎨 LoRA")
                        vid_lora_selector = gr.Dropdown(label="LoRA File", choices=["None"], value="None", interactive=True)
                        vid_lora_weight = gr.Slider(label="LoRA Weight", minimum=0.0, maximum=1.5, step=0.05, value=1.0, interactive=True)
                        btn_refresh_loras = gr.Button("🔄 Refresh LoRA List", variant="secondary")

                        gr.Markdown("---")
                        vid_seed = gr.Number(label="Seed (−1 = random)", value=-1, precision=0, interactive=True)
                        btn_generate = gr.Button("🎬 Generate Video", variant="primary")
                        vid_status   = gr.Markdown("Ready.")

                gr.Markdown("---")
                gr.Markdown("#### 🎥 Preview")
                with gr.Row():
                    vid_preview   = gr.Video(label="Generated Video", interactive=False, autoplay=True)
                    vid_thumb_out = gr.Image(label="First Frame", type="filepath", interactive=False, height=300)

                vid_meta_md = gr.Markdown("*Generate a video to see metadata here.*")
                with gr.Row():
                    btn_approve  = gr.Button("✅ Approve", variant="primary")
                    btn_reject   = gr.Button("❌ Reject", variant="stop")
                    btn_reroll   = gr.Button("🔄 Re-roll Seed", variant="secondary")
                    btn_save_lib = gr.Button("💾 Save to Library", variant="secondary")
                vid_action_msg = gr.Markdown("")

            with gr.TabItem("⏳ Pending Review"):
                gr.Markdown("#### Videos awaiting approval")
                with gr.Row():
                    btn_refresh_pending = gr.Button("🔄 Refresh", variant="secondary")
                    pending_count_md    = gr.Markdown("*Click Refresh to load.*")

                with gr.Row():
                    with gr.Column(scale=2):
                        pending_df = gr.Dataframe(
                            headers=["ID", "Family", "Pipeline", "Character", "Project", "Frames", "FPS", "Created"],
                            datatype=["number", "str", "str", "str", "str", "number", "number", "str"],
                            row_count=10, interactive=False,
                        )
                    with gr.Column(scale=1):
                        pending_vid_player = gr.Video(label="Selected Video", interactive=False, autoplay=False)
                        pending_meta_md    = gr.Markdown("*Select a row to preview.*")
                        pending_sel_id     = gr.Number(label="Selected ID", value=-1, visible=False)

                with gr.Row():
                    btn_pend_approve = gr.Button("✅ Approve", variant="primary")
                    btn_pend_reject  = gr.Button("❌ Reject", variant="stop")
                    btn_pend_delete  = gr.Button("🗑️ Delete", variant="secondary")
                pending_action_msg = gr.Markdown("")

            with gr.TabItem("📼 Video Library"):
                gr.Markdown("#### Approved video library")
                with gr.Row():
                    lib_filter_proj     = gr.Dropdown(label="Project", choices=["All"], value="All", interactive=True)
                    lib_filter_char     = gr.Dropdown(label="Character", choices=["All"], value="All", interactive=True)
                    lib_filter_family   = gr.Dropdown(label="Family", choices=["All", "LTX-Video", "Wan 2.2", "Hunyuan Video"], value="All", interactive=True)
                    lib_filter_pipeline = gr.Dropdown(label="Pipeline", choices=["All", "distilled", "two_stage", "two_stage_hq", "text2video", "image2video"], value="All", interactive=True)
                    btn_lib_refresh = gr.Button("🔄 Refresh", variant="secondary")

                lib_gallery = gr.Gallery(label="Approved Videos (thumbnails)", columns=4, object_fit="cover", height=400)

                gr.Markdown("---")
                gr.Markdown("#### 📋 Details & Playback")
                with gr.Row():
                    lib_vid_player = gr.Video(label="Selected Video", interactive=False, autoplay=False, scale=2)
                    lib_detail_md  = gr.Markdown("*Click a thumbnail above.*", scale=1)

                lib_sel_id = gr.Number(label="Selected Library ID", value=-1, visible=False)
                with gr.Row():
                    lib_export_path = gr.Textbox(label="Export destination path", placeholder="C:/exports/", interactive=True, scale=3)
                    btn_lib_export  = gr.Button("📤 Export", variant="secondary", scale=1)
                    btn_lib_remove  = gr.Button("🗑️ Remove from Library", variant="stop", scale=1)
                lib_action_msg = gr.Markdown("")

    def _status_html(detect_result: dict, loaded_name: str | None) -> str:
        rows = []
        for key, info in detect_result.items():
            dot = "🟢" if info.get("found") else "🔴"
            p = info.get("path", "") or "—"
            n = len(info.get("files", []))
            badge = ""
            if loaded_name and (key in ["ltx", "wan", "hunyuan"]):
                badge = f"<span style='background:#22c55e;color:#000;padding:1px 7px;border-radius:4px;font-size:0.72rem;margin-left:6px;'>{loaded_name.upper()} LOADED</span>"
            rows.append(f"<div style='padding:5px 12px;margin:3px 0;border-radius:8px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'>{dot} <b>{info.get('label', key)}</b>{badge} <code style='font-size:0.78rem;color:#a5b4fc;'>{p}</code><span style='color:#6b7280;font-size:0.78rem;margin-left:8px;'>({n} files)</span></div>")
        return "".join(rows)

    def _video_row(vid: dict) -> list:
        return [vid["id"], vid.get("model_family", "LTX-Video"), vid.get("pipeline", ""), vid.get("character_name") or "—", vid.get("project_name") or "—", vid.get("num_frames", 0), vid.get("fps", 0), (vid.get("created_at") or "")[:16]]

    def _meta_md(vid: dict) -> str:
        if not vid: return "*No video selected.*"
        dur = vid.get("duration_seconds")
        dur_str = f"{dur:.1f}s" if dur else "?"
        lines = [
            f"**ID:** {vid['id']}  ·  **Family:** `{vid.get('model_family','LTX-Video')}`  ·  **Pipeline:** `{vid.get('pipeline','?')}`  ·  **Preset:** `{vid.get('preset') or '—'}`",
            f"**Status:** {vid.get('status','?')}",
            f"**Seed:** `{vid.get('seed','?')}`  ·  **Res:** {vid.get('width','?')}×{vid.get('height','?')}  ·  **FPS:** {vid.get('fps','?')}  ·  **Frames:** {vid.get('num_frames','?')}  ·  **Duration:** {dur_str}",
            f"**Steps:** {vid.get('steps','?')}  ·  **CFG:** {vid.get('guidance_scale','?')}",
            f"**Project:** {vid.get('project_name') or '—'}  ·  **Character:** {vid.get('character_name') or '—'}  ·  **Location:** {vid.get('location_name') or '—'}"
        ]
        if vid.get("lora_path"): lines.append(f"**LoRA:** `{vid['lora_path']}` @ {vid.get('lora_weight',1.0)}")
        lines.append(f"\n**Prompt:**\n> {vid.get('prompt','') or '—'}")
        if vid.get("negative_prompt"): lines.append(f"\n**Negative:**\n> {vid['negative_prompt']}")
        lines.append(f"\n*Created: {(vid.get('created_at') or '')[:19]} UTC*")
        return "\n\n".join(lines)

    # Model Manager Callbacks
    def on_save_config_ltx(ltx, gemma, upscaler, lora_dir, device, dtype):
        db.save_ltx_config(ltx_model_path=ltx, ltx_gemma_path=gemma, ltx_upscaler_path=upscaler, ltx_lora_dir=lora_dir, ltx_device=device, ltx_dtype=dtype)
        return "<div style='padding:8px;color:#22c55e;'>✅ LTX Paths saved.</div>"

    def on_detect_ltx(ltx, gemma, upscaler, lora_dir):
        from studio.generation import ltx_manager
        result = ltx_manager.detect_models(ltx, gemma, upscaler, lora_dir)
        loaded = ltx_manager.get_loaded_pipeline_name()
        return _status_html(result, loaded)

    def on_load_pipe_ltx(pipe_name, ltx, gemma, upscaler, lora_dir, device, dtype, lora_sel, lora_wt):
        from studio.generation import ltx_manager
        lora_full = os.path.join(lora_dir, lora_sel) if lora_sel and lora_sel != "None" and lora_dir else ""
        _, msg = ltx_manager.load_pipeline(pipe_name, ltx, upscaler, gemma, device, dtype, lora_full, float(lora_wt))
        return on_detect_ltx(ltx, gemma, upscaler, lora_dir) + f"<p style='color:#e5e7eb;'>{msg}</p>"

    def on_save_config_wan(wan_path, device, dtype):
        db.save_wan_config(wan_path, device, dtype)
        return "<div style='padding:8px;color:#22c55e;'>✅ Wan Paths saved.</div>"

    def on_detect_wan(wan_path):
        from studio.generation import wan_manager
        res = wan_manager.detect_models(wan_path)
        return _status_html(res, wan_manager.get_loaded_pipeline_name())

    def on_load_pipe_wan(pipe_name, wan_path, device, dtype):
        from studio.generation import wan_manager
        _, msg = wan_manager.load_pipeline(pipe_name, wan_path, device, dtype)
        return on_detect_wan(wan_path) + f"<p style='color:#e5e7eb;'>{msg}</p>"

    def on_save_config_hunyuan(hun_path, device, dtype):
        db.save_hunyuan_config(hun_path, device, dtype)
        return "<div style='padding:8px;color:#22c55e;'>✅ Hunyuan Paths saved.</div>"

    def on_detect_hunyuan(hun_path):
        from studio.generation import hunyuan_manager
        res = hunyuan_manager.detect_models(hun_path)
        return _status_html(res, hunyuan_manager.get_loaded_pipeline_name())

    def on_load_pipe_hunyuan(pipe_name, hun_path, device, dtype):
        from studio.generation import hunyuan_manager
        _, msg = hunyuan_manager.load_pipeline(pipe_name, hun_path, device, dtype)
        return on_detect_hunyuan(hun_path) + f"<p style='color:#e5e7eb;'>{msg}</p>"

    def on_unload(family):
        if family == "LTX-Video":
            from studio.generation import ltx_manager
            return ltx_manager.unload_pipeline()
        elif family == "Wan 2.2":
            from studio.generation import wan_manager
            return wan_manager.unload_pipeline()
        elif family == "Hunyuan Video":
            from studio.generation import hunyuan_manager
            return hunyuan_manager.unload_pipeline()
        return "Unknown family"

    def on_family_change(family):
        if family == "LTX-Video":
            return gr.update(choices=["distilled", "two_stage", "two_stage_hq"], value="distilled")
        elif family in ("Wan 2.2", "Hunyuan Video"):
            return gr.update(choices=["text2video", "image2video"], value="text2video")
        return gr.update()

    def on_preset_change(preset_val, pipe_val):
        if preset_val == "custom":
            return (gr.update(),)*7
        p = _PRESETS.get(preset_val, _PRESETS["draft"])
        return (gr.update(value=p["width"]), gr.update(value=p["height"]), gr.update(value=p["fps"]),
                gr.update(value=p["num_frames"]), gr.update(value=p["steps"]), gr.update(value=p["guidance_scale"]),
                gr.update(value=p["pipeline"] if pipe_val in ["distilled", "two_stage", "two_stage_hq"] else "text2video"))

    def on_generate(prompt, neg_prompt, family, pipeline_name, preset_val, w, h, fps, nf, steps, cfg, lora_dir, lora_sel, lora_wt, seed, ref_img, proj, char, loc):
        global _current_video_id
        from studio.generation import ltx_manager, wan_manager, hunyuan_manager, video_output_manager
        
        mgr = ltx_manager if family == "LTX-Video" else (wan_manager if family == "Wan 2.2" else hunyuan_manager)
        state = mgr._cached_state
        if not state:
            return None, None, f"❌ **No pipeline loaded for {family}.**", "*No video generated.*"
        
        lora_full = os.path.join(lora_dir, lora_sel) if lora_sel and lora_sel != "None" and lora_dir else ""
        
        frames, actual_seed, err = mgr.generate_video(
            state=state, prompt=prompt, negative_prompt=neg_prompt, width=int(w), height=int(h),
            fps=int(fps), num_frames=int(nf), steps=int(steps), guidance_scale=float(cfg),
            seed=int(seed), pipeline_name=pipeline_name, reference_image=ref_img
        )
        if err or not frames:
            return None, None, err or "❌ Unknown generation error.", "*Generation failed.*"
            
        proj_obj = db.get_project_by_name(proj) if proj else None
        char_obj = db.get_character_by_name(char) if char else None
        loc_obj = db.get_location_by_name(loc) if loc else None
        
        out_dir = video_output_manager.get_video_output_dir(STUDIO_ROOT, proj or "", char or "")
        meta = video_output_manager.build_video_metadata(
            prompt, neg_prompt, actual_seed, pipeline_name, preset_val, int(w), int(h), int(fps), int(nf), int(steps), float(cfg),
            lora_full, float(lora_wt), "", char or "", loc or "", proj or ""
        )
        meta["model_family"] = family
        mp4_path, thumb_path, duration = video_output_manager.save_video(frames, int(fps), out_dir, meta)
        
        vid_id = db.add_generated_video(
            file_path=mp4_path, pipeline=pipeline_name, model_family=family, prompt=prompt, negative_prompt=neg_prompt,
            seed=actual_seed, width=int(w), height=int(h), fps=int(fps), num_frames=int(nf), steps=int(steps), guidance_scale=float(cfg),
            lora_path=lora_full, lora_weight=float(lora_wt), thumbnail_path=thumb_path, duration_seconds=duration, preset=preset_val,
            project_id=proj_obj["id"] if proj_obj else None, character_id=char_obj["id"] if char_obj else None, location_id=loc_obj["id"] if loc_obj else None
        )
        _current_video_id = vid_id
        return mp4_path, thumb_path, f"✅ **Generated!** Seed: `{actual_seed}`", _meta_md(db.get_generated_video(vid_id))

    def on_reroll(*args):
        l = list(args); l[14] = -1 # seed is index 14
        return on_generate(*l)

    def on_approve_current():
        if _current_video_id is None: return "⚠️ No video."
        db.update_video_status(_current_video_id, "approved")
        return f"✅ Video **#{_current_video_id}** approved."
    def on_reject_current():
        if _current_video_id is None: return "⚠️ No video."
        db.update_video_status(_current_video_id, "rejected")
        return f"❌ Video **#{_current_video_id}** rejected."
    
    def on_refresh_pending():
        vids = db.list_generated_videos(status="pending")
        return [[v["id"], v.get("model_family"), v.get("pipeline"), v.get("character_name"), v.get("project_name"), v.get("num_frames"), v.get("fps"), v.get("created_at")] for v in vids], f"**{len(vids)} pending.**"
    def on_select_pending(evt: gr.SelectData, df):
        vid = db.get_generated_video(int(df.iloc[evt.index[0], 0]))
        return vid.get("file_path") if vid else None, _meta_md(vid), vid["id"]
    def on_pend_approve(sel): db.update_video_status(int(sel), "approved"); return "✅ Approved"
    def on_pend_reject(sel): db.update_video_status(int(sel), "rejected"); return "❌ Rejected"
    def on_pend_delete(sel): db.delete_generated_video(int(sel)); return "🗑️ Deleted"

    _lib_records = []
    def on_lib_refresh(proj_f, char_f, fam_f, pipe_f):
        nonlocal _lib_records
        proj_id = db.get_project_by_name(proj_f)["id"] if proj_f and proj_f != "All" else None
        char_id = db.get_character_by_name(char_f)["id"] if char_f and char_f != "All" else None
        _lib_records = db.list_generated_videos(project_id=proj_id, character_id=char_id, status="approved", pipeline=pipe_f if pipe_f != "All" else None, model_family=fam_f if fam_f != "All" else None)
        items = [(v.get("thumbnail_path") or v.get("file_path"), f"{v.get('pipeline')} | seed:{v.get('seed')}") for v in _lib_records]
        return [i for i in items if i[0]], f"**{len(_lib_records)} videos.**", None, "*Select a thumbnail.*"
    def on_lib_select(evt: gr.SelectData):
        vid = _lib_records[evt.index]
        return vid.get("file_path"), _meta_md(vid), vid["id"]

    # WIRING
    btn_save_cfg_ltx.click(on_save_config_ltx, [mm_ltx_path, mm_gemma_path, mm_upscaler_path, mm_lora_dir, mm_device, mm_dtype], [mm_status_html_ltx])
    btn_detect_ltx.click(on_detect_ltx, [mm_ltx_path, mm_gemma_path, mm_upscaler_path, mm_lora_dir], [mm_status_html_ltx])
    btn_load_pipe_ltx.click(on_load_pipe_ltx, [vid_pipeline, mm_ltx_path, mm_gemma_path, mm_upscaler_path, mm_lora_dir, mm_device, mm_dtype, vid_lora_selector, vid_lora_weight], [mm_status_html_ltx])
    btn_unload_ltx.click(lambda: on_unload("LTX-Video"), [], [mm_status_html_ltx])

    btn_save_cfg_wan.click(on_save_config_wan, [mm_wan_path, mm_wan_device, mm_wan_dtype], [mm_status_html_wan])
    btn_detect_wan.click(on_detect_wan, [mm_wan_path], [mm_status_html_wan])
    btn_load_pipe_wan.click(on_load_pipe_wan, [vid_pipeline, mm_wan_path, mm_wan_device, mm_wan_dtype], [mm_status_html_wan])
    btn_unload_wan.click(lambda: on_unload("Wan 2.2"), [], [mm_status_html_wan])

    btn_save_cfg_hunyuan.click(on_save_config_hunyuan, [mm_hunyuan_path, mm_hunyuan_device, mm_hunyuan_dtype], [mm_status_html_hunyuan])
    btn_detect_hunyuan.click(on_detect_hunyuan, [mm_hunyuan_path], [mm_status_html_hunyuan])
    btn_load_pipe_hunyuan.click(on_load_pipe_hunyuan, [vid_pipeline, mm_hunyuan_path, mm_hunyuan_device, mm_hunyuan_dtype], [mm_status_html_hunyuan])
    btn_unload_hunyuan.click(lambda: on_unload("Hunyuan Video"), [], [mm_status_html_hunyuan])

    vid_model_family.change(on_family_change, [vid_model_family], [vid_pipeline])
    vid_preset.change(on_preset_change, [vid_preset, vid_pipeline], [vid_width, vid_height, vid_fps, vid_frames, vid_steps, vid_guidance, vid_pipeline])

    btn_generate.click(on_generate, [vid_prompt, vid_neg_prompt, vid_model_family, vid_pipeline, vid_preset, vid_width, vid_height, vid_fps, vid_frames, vid_steps, vid_guidance, mm_lora_dir, vid_lora_selector, vid_lora_weight, vid_seed, vid_ref_image, vid_proj_selector, vid_char_selector, vid_loc_selector], [vid_preview, vid_thumb_out, vid_status, vid_meta_md])
    btn_reroll.click(on_reroll, [vid_prompt, vid_neg_prompt, vid_model_family, vid_pipeline, vid_preset, vid_width, vid_height, vid_fps, vid_frames, vid_steps, vid_guidance, mm_lora_dir, vid_lora_selector, vid_lora_weight, vid_seed, vid_ref_image, vid_proj_selector, vid_char_selector, vid_loc_selector], [vid_preview, vid_thumb_out, vid_status, vid_meta_md])
    
    btn_approve.click(on_approve_current, [], [vid_action_msg])
    btn_reject.click(on_reject_current, [], [vid_action_msg])
    btn_save_lib.click(on_approve_current, [], [vid_action_msg])

    btn_refresh_pending.click(on_refresh_pending, [], [pending_df, pending_count_md])
    pending_df.select(on_select_pending, [pending_df], [pending_vid_player, pending_meta_md, pending_sel_id])
    btn_pend_approve.click(on_pend_approve, [pending_sel_id], [pending_action_msg])
    btn_pend_reject.click(on_pend_reject, [pending_sel_id], [pending_action_msg])
    btn_pend_delete.click(on_pend_delete, [pending_sel_id], [pending_action_msg])

    btn_lib_refresh.click(on_lib_refresh, [lib_filter_proj, lib_filter_char, lib_filter_family, lib_filter_pipeline], [lib_gallery, lib_action_msg, lib_vid_player, lib_detail_md])
    lib_gallery.select(on_lib_select, [], [lib_vid_player, lib_detail_md, lib_sel_id])
    
    return vid_proj_selector, vid_char_selector, vid_loc_selector
