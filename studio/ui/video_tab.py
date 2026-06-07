"""
studio/ui/video_tab.py
Primary model order: Wan 2.2 → Hunyuan Video → LTX-Video
Focus: consistency, realism, motion for vertical drama production
"""

import os
import logging
import random
from datetime import datetime
import gradio as gr
from studio.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

STUDIO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Per-model drama presets — vertical 9:16 by default
# ---------------------------------------------------------------------------

_WAN_PRESETS = {
    "draft":      {"width": 480, "height": 832,  "fps": 16, "num_frames": 49, "steps": 25, "guidance_scale": 5.0, "pipeline": "text2video"},
    "production": {"width": 480, "height": 832,  "fps": 16, "num_frames": 81, "steps": 50, "guidance_scale": 6.0, "pipeline": "text2video"},
    "cinema":     {"width": 576, "height": 1024, "fps": 16, "num_frames": 81, "steps": 50, "guidance_scale": 7.0, "pipeline": "text2video"},
}

_HUNYUAN_PRESETS = {
    "draft":      {"width": 544, "height": 960,  "fps": 24, "num_frames": 65,  "steps": 30, "guidance_scale": 6.0, "pipeline": "text2video"},
    "production": {"width": 720, "height": 1280, "fps": 24, "num_frames": 97,  "steps": 50, "guidance_scale": 6.0, "pipeline": "text2video"},
    "cinema":     {"width": 720, "height": 1280, "fps": 24, "num_frames": 129, "steps": 50, "guidance_scale": 7.0, "pipeline": "text2video"},
}

_LTX_PRESETS = {
    "draft":      {"width": 480, "height": 832,  "fps": 24, "num_frames": 65,  "steps": 20, "guidance_scale": 3.0, "pipeline": "distilled"},
    "production": {"width": 768, "height": 1344, "fps": 24, "num_frames": 97,  "steps": 40, "guidance_scale": 3.5, "pipeline": "two_stage"},
    "cinema":     {"width": 768, "height": 1344, "fps": 24, "num_frames": 129, "steps": 50, "guidance_scale": 4.0, "pipeline": "two_stage_hq"},
}

# LTX-2.3 — native audio+video. Pipeline values map to ltx2_manager modules.
_LTX2_PRESETS = {
    "draft":      {"width": 480, "height": 832,  "fps": 25, "num_frames": 97,  "steps": 8,  "guidance_scale": 1.0, "pipeline": "distilled"},
    "production": {"width": 704, "height": 1216, "fps": 25, "num_frames": 121, "steps": 30, "guidance_scale": 3.0, "pipeline": "two_stage"},
    "cinema":     {"width": 1088, "height": 1920, "fps": 25, "num_frames": 121, "steps": 40, "guidance_scale": 3.5, "pipeline": "two_stage"},
}

_FAMILY_PRESETS = {
    "Wan 2.2":       _WAN_PRESETS,
    "Hunyuan Video": _HUNYUAN_PRESETS,
    "LTX-2.3":       _LTX2_PRESETS,
    "LTX-Video":     _LTX_PRESETS,
}

_LTX2_PIPELINES = ["distilled", "one_stage", "two_stage"]

_DEFAULT_FAMILY  = "Wan 2.2"
_DEFAULT_PRESET  = _WAN_PRESETS["production"]


def create_video_tab(db: DatabaseManager):
    with gr.TabItem("🎬 Video Generation"):

        # -----------------------------------------------------------------------
        # Model Manager (collapsed by default — configure once, then hide)
        # -----------------------------------------------------------------------
        with gr.Accordion("⚙️ Model Manager — set paths once, then collapse", open=False):
            gr.Markdown(
                "Point to your **local** model folders. No models are downloaded automatically. "
                "Save paths → Load Pipeline before queuing jobs. **Wan 2.2 is the primary model.**"
            )
            with gr.Tabs():
                # Wan 2.2 — first tab, primary model
                with gr.TabItem("Wan 2.2 ★ Primary"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            mm_wan_path   = gr.Textbox(label="Wan 2.2 Model Path (T2V or I2V folder)", placeholder="/workspace/models/Wan2.2-T2V-14B", interactive=True)
                            mm_wan_i2v    = gr.Textbox(label="Wan 2.2 I2V Model Path (optional, separate checkpoint)", placeholder="/workspace/models/Wan2.2-I2V-14B", interactive=True)
                        with gr.Column(scale=1):
                            mm_wan_device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda", interactive=True)
                            mm_wan_dtype  = gr.Dropdown(label="dtype", choices=["bfloat16", "float16"], value="bfloat16", interactive=True)
                    with gr.Row():
                        btn_detect_wan    = gr.Button("🔍 Detect", variant="secondary")
                        btn_save_cfg_wan  = gr.Button("💾 Save Paths", variant="secondary")
                        btn_load_pipe_wan = gr.Button("⚡ Load Wan Pipeline", variant="primary")
                        btn_unload_wan    = gr.Button("🗑️ Unload", variant="stop")
                    mm_status_html_wan = gr.HTML("<div style='padding:8px;color:#9ca3af;font-size:0.85rem;'>Wan status...</div>")

                # Hunyuan Video — second tab
                with gr.TabItem("Hunyuan Video ★ Secondary"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            mm_hunyuan_path = gr.Textbox(label="Hunyuan Video Model Path", placeholder="/workspace/models/HunyuanVideo", interactive=True)
                        with gr.Column(scale=1):
                            mm_hunyuan_device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda", interactive=True)
                            mm_hunyuan_dtype  = gr.Dropdown(label="dtype", choices=["bfloat16", "float16"], value="bfloat16", interactive=True)
                    with gr.Row():
                        btn_detect_hunyuan    = gr.Button("🔍 Detect", variant="secondary")
                        btn_save_cfg_hunyuan  = gr.Button("💾 Save Paths", variant="secondary")
                        btn_load_pipe_hunyuan = gr.Button("⚡ Load Hunyuan Pipeline", variant="primary")
                        btn_unload_hunyuan    = gr.Button("🗑️ Unload", variant="stop")
                    mm_status_html_hunyuan = gr.HTML("<div style='padding:8px;color:#9ca3af;font-size:0.85rem;'>Hunyuan status...</div>")

                # LTX-2.3 — native audio+video (subprocess / own venv)
                with gr.TabItem("LTX-2.3 ★ Audio+Video"):
                    gr.Markdown(
                        "LTX-2.3 generates **synchronized audio + video** in one pass — these shots "
                        "skip the TTS + lip-sync steps. Runs in its own `uv` venv via subprocess "
                        "(needs torch 2.7 / CUDA ≥12.7). Set the paths below."
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            mm_ltx2_repo   = gr.Textbox(label="LTX-2 Repo Dir", value="/workspace/LTX-2", interactive=True)
                            mm_ltx2_python = gr.Textbox(label="LTX-2 venv Python", value="/root/ltx2-venv/bin/python", interactive=True)
                            mm_ltx2_ckpt   = gr.Textbox(label="LTX-2.3 Checkpoint (.safetensors)", placeholder="/workspace/models/LTX-2.3/ltxv-2.3-....safetensors", interactive=True)
                            mm_ltx2_upsamp = gr.Textbox(label="Spatial Upsampler (.safetensors)", placeholder="/workspace/models/LTX-2.3/...upsampler....safetensors", interactive=True)
                            mm_ltx2_lora   = gr.Textbox(label="Distilled LoRA (.safetensors)", placeholder="/workspace/models/LTX-2.3/...distilled....safetensors", interactive=True)
                            mm_ltx2_gemma  = gr.Textbox(label="Gemma 3 Encoder Dir", placeholder="/workspace/models/LTX-2.3/gemma", interactive=True)
                        with gr.Column(scale=1):
                            mm_ltx2_module = gr.Dropdown(label="Pipeline Module", choices=["ltx_pipelines.ti2vid_two_stages", "ltx_pipelines.ti2vid_one_stage", "ltx_pipelines.distilled"], value="ltx_pipelines.ti2vid_two_stages", interactive=True)
                            mm_ltx2_quant  = gr.Dropdown(label="Quantization", choices=["", "fp8-cast"], value="fp8-cast", interactive=True)
                            mm_ltx2_extra  = gr.Textbox(label="Extra CLI args (advanced)", placeholder="--foo bar", interactive=True)
                    with gr.Row():
                        btn_detect_ltx2   = gr.Button("🔍 Detect", variant="secondary")
                        btn_save_cfg_ltx2 = gr.Button("💾 Save Paths", variant="primary")
                    mm_status_html_ltx2 = gr.HTML("<div style='padding:8px;color:#9ca3af;font-size:0.85rem;'>LTX-2.3 status...</div>")

                # LTX-Video — old 0.9.x line, draft/iteration only
                with gr.TabItem("LTX-Video 0.9.x (Draft)"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            mm_ltx_path      = gr.Textbox(label="LTX-Video Model Path", placeholder="/workspace/models/ltx-video-2b", interactive=True)
                            mm_gemma_path    = gr.Textbox(label="Gemma Prompt Enhancer (optional)", placeholder="/workspace/models/gemma-2b", interactive=True)
                            mm_upscaler_path = gr.Textbox(label="Upscaler Path (required for two_stage)", placeholder="/workspace/models/ltx-upscaler", interactive=True)
                            mm_lora_dir      = gr.Textbox(label="LoRA Directory", placeholder="/workspace/loras/", interactive=True)
                        with gr.Column(scale=1):
                            mm_device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda", interactive=True)
                            mm_dtype  = gr.Dropdown(label="dtype", choices=["bfloat16", "float16", "float32"], value="bfloat16", interactive=True)
                    with gr.Row():
                        btn_detect_ltx    = gr.Button("🔍 Detect", variant="secondary")
                        btn_save_cfg_ltx  = gr.Button("💾 Save Paths", variant="secondary")
                        btn_load_pipe_ltx = gr.Button("⚡ Load LTX Pipeline", variant="primary")
                        btn_unload_ltx    = gr.Button("🗑️ Unload", variant="stop")
                    mm_status_html_ltx = gr.HTML("<div style='padding:8px;color:#9ca3af;font-size:0.85rem;'>LTX status...</div>")

        # -----------------------------------------------------------------------
        # Main workflow tabs
        # -----------------------------------------------------------------------
        with gr.Tabs():
            with gr.TabItem("🎬 Shot Builder"):
                with gr.Row():
                    # Left — context, model, prompt
                    with gr.Column(scale=2, variant="panel"):

                        gr.Markdown("#### 🎭 Shot Context")
                        with gr.Row():
                            vid_proj_selector = gr.Dropdown(label="Project",    choices=[], value=None, interactive=True)
                            vid_char_selector = gr.Dropdown(label="Character",  choices=[], value=None, interactive=True)
                        with gr.Row():
                            vid_loc_selector  = gr.Dropdown(label="Location",   choices=[], value=None, interactive=True)
                            vid_shot_selector = gr.Dropdown(label="Shot Type",  choices=[], value=None, interactive=True)

                        gr.Markdown("---")
                        gr.Markdown("#### 🤖 Model & Quality")
                        vid_model_family = gr.Radio(
                            label="Model Family",
                            choices=["Wan 2.2", "Hunyuan Video", "LTX-2.3", "LTX-Video"],
                            value="Wan 2.2",
                            interactive=True,
                        )
                        with gr.Row():
                            vid_pipeline = gr.Radio(
                                label="Pipeline",
                                choices=["text2video", "image2video"],
                                value="text2video",
                                interactive=True,
                            )
                            vid_preset = gr.Radio(
                                label="Quick Preset",
                                choices=["draft", "production", "cinema", "custom"],
                                value="production",
                                interactive=True,
                            )

                        # I2V info banner — shown when image2video is selected
                        i2v_info = gr.HTML(
                            "<div style='display:none'></div>",
                            visible=False,
                        )

                        gr.Markdown("---")
                        gr.Markdown("#### ✍️ Prompt")
                        vid_prompt = gr.Textbox(
                            label="Video Prompt",
                            lines=4,
                            placeholder="Close up of Maya, intense expression, warm interior lighting, dramatic shadows, vertical framing, cinematic...",
                            interactive=True,
                        )
                        vid_neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value="blurry, distorted, low quality, watermark, text, duplicate faces, bad anatomy, unnatural motion",
                            interactive=True,
                        )

                        gr.Markdown("---")
                        gr.Markdown("#### 🖼️ Character Reference Image")
                        gr.Markdown(
                            "*Upload a reference image of your character for I2V (Wan) or consistency anchoring. "
                            "This is the key to getting the same face across shots.*",
                        )
                        vid_ref_image = gr.Image(
                            label="Reference / Starting Frame",
                            type="filepath",
                            interactive=True,
                            height=200,
                        )

                    # Right — parameters
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("#### 🎛️ Generation Parameters")
                        vid_width    = gr.Slider(label="Width",          minimum=256, maximum=1920, step=32,  value=_DEFAULT_PRESET["width"],          interactive=True)
                        vid_height   = gr.Slider(label="Height",         minimum=256, maximum=1920, step=32,  value=_DEFAULT_PRESET["height"],         interactive=True)
                        vid_fps      = gr.Slider(label="FPS",            minimum=8,   maximum=30,   step=1,   value=_DEFAULT_PRESET["fps"],            interactive=True)
                        vid_frames   = gr.Slider(label="Frames",         minimum=9,   maximum=257,  step=4,   value=_DEFAULT_PRESET["num_frames"],     interactive=True)
                        vid_steps    = gr.Slider(label="Steps",          minimum=1,   maximum=80,   step=1,   value=_DEFAULT_PRESET["steps"],          interactive=True)
                        vid_guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=12.0, step=0.5, value=_DEFAULT_PRESET["guidance_scale"], interactive=True)

                        gr.Markdown("---")
                        gr.Markdown("#### 🎨 LoRA (LTX only)")
                        vid_lora_selector = gr.Dropdown(label="LoRA File", choices=["None"], value="None", interactive=True)
                        vid_lora_weight   = gr.Slider(label="LoRA Weight", minimum=0.0, maximum=1.5, step=0.05, value=1.0, interactive=True)
                        btn_refresh_loras = gr.Button("🔄 Refresh LoRA List", variant="secondary")

                        gr.Markdown("---")
                        vid_seed = gr.Number(label="Seed  (−1 = random)", value=-1, precision=0, interactive=True)

                        gr.Markdown("---")
                        gr.Markdown(
                            "**Wan 2.2** → realism, consistency, I2V character lock-in\n\n"
                            "**Hunyuan** → cinematic wide shots, complex motion\n\n"
                            "**LTX** → fast drafts, B-roll, storyboard checks"
                        )
                        btn_queue_render = gr.Button("🚀 Add to Render Queue", variant="primary", size="lg")
                        vid_action_msg   = gr.Markdown("")

            # -------------------------------------------------------------------
            # Render queue tab
            # -------------------------------------------------------------------
            with gr.TabItem("⚙️ Render Queue"):
                gr.Markdown("#### Background Render Jobs")
                with gr.Row():
                    btn_refresh_queue = gr.Button("🔄 Refresh", variant="secondary")
                    queue_count_md    = gr.Markdown("*Click Refresh to load.*")
                with gr.Row():
                    with gr.Column(scale=2):
                        queue_df = gr.Dataframe(
                            headers=["ID", "Status", "Family", "Character", "Shot Type", "Created"],
                            datatype=["number", "str", "str", "str", "str", "str"],
                            row_count=10, interactive=False,
                        )
                    with gr.Column(scale=1):
                        queue_vid_player = gr.Video(label="Completed Video Preview", interactive=False, autoplay=False)
                        queue_meta_md    = gr.Markdown("*Select a row.*")
                        queue_sel_id     = gr.Number(label="Selected Job ID", value=-1, visible=False)
                with gr.Row():
                    btn_q_cancel  = gr.Button("🛑 Cancel",  variant="stop")
                    btn_q_approve = gr.Button("✅ Approve", variant="primary")
                    btn_q_reject  = gr.Button("❌ Reject",  variant="stop")
                queue_action_msg = gr.Markdown("")

            # -------------------------------------------------------------------
            # Video library tab
            # -------------------------------------------------------------------
            with gr.TabItem("📼 Video Library"):
                gr.Markdown("#### Approved video library")
                with gr.Row():
                    lib_filter_proj  = gr.Dropdown(label="Project",  choices=["All"], value="All", interactive=True)
                    lib_filter_char  = gr.Dropdown(label="Character", choices=["All"], value="All", interactive=True)
                    lib_filter_loc   = gr.Dropdown(label="Location",  choices=["All"], value="All", interactive=True)
                    lib_filter_date  = gr.Dropdown(label="Date", choices=["All Time", "Today", "Last 7 Days"], value="All Time", interactive=True)
                with gr.Row():
                    lib_filter_family   = gr.Dropdown(label="Family",   choices=["All", "Wan 2.2", "Hunyuan Video", "LTX-2.3", "LTX-Video"], value="All", interactive=True)
                    lib_filter_pipeline = gr.Dropdown(label="Pipeline", choices=["All", "text2video", "image2video", "distilled", "two_stage", "two_stage_hq"], value="All", interactive=True)
                    btn_lib_refresh = gr.Button("🔄 Refresh Search", variant="primary")

                lib_gallery = gr.Gallery(label="Approved Videos (thumbnails)", columns=4, object_fit="cover", height=400)

                gr.Markdown("---")
                gr.Markdown("#### 📋 Details & Playback")
                with gr.Row():
                    lib_vid_player = gr.Video(label="Selected Video", interactive=False, autoplay=False, scale=2)
                    with gr.Column(scale=1):
                        lib_detail_md = gr.Markdown("*Click a thumbnail above.*")

                lib_sel_id = gr.Number(label="Selected Library ID", value=-1, visible=False)
                with gr.Row():
                    lib_export_path = gr.Textbox(label="Export destination path", placeholder="/workspace/exports/", interactive=True, scale=3)
                    btn_lib_export  = gr.Button("📤 Export", variant="secondary", scale=1)
                    btn_lib_remove  = gr.Button("🗑️ Remove from Library", variant="stop", scale=1)
                lib_action_msg = gr.Markdown("")

    # ---------------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------------

    def _status_html(detect_result: dict, loaded_name: str | None) -> str:
        rows = []
        for key, info in detect_result.items():
            dot   = "🟢" if info.get("found") else "🔴"
            p     = info.get("path", "") or "—"
            n     = len(info.get("files", []))
            badge = ""
            if loaded_name:
                badge = f"<span style='background:#22c55e;color:#000;padding:1px 7px;border-radius:4px;font-size:0.72rem;margin-left:6px;'>{loaded_name.upper()} LOADED</span>"
            rows.append(
                f"<div style='padding:5px 12px;margin:3px 0;border-radius:8px;"
                f"background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'>"
                f"{dot} <b>{info.get('label', key)}</b>{badge} "
                f"<code style='font-size:0.78rem;color:#a5b4fc;'>{p}</code>"
                f"<span style='color:#6b7280;font-size:0.78rem;margin-left:8px;'>({n} files)</span></div>"
            )
        return "".join(rows)

    def _job_row(job: dict) -> list:
        return [
            job["id"],
            job.get("status", "unknown"),
            job.get("model_family", ""),
            job.get("character_name", "—"),
            job.get("shot_type_name", "—"),
            (job.get("created_at") or "")[:16],
        ]

    def _meta_md(vid: dict, is_job: bool = False) -> str:
        if not vid:
            return "*No selection.*"
        lines = []
        if is_job:
            lines.append(f"**Job ID:** {vid['id']}  ·  **Status:** {vid['status']}")
            if vid.get("error_message"):
                lines.append(f"**Error:** `{vid['error_message']}`")
        else:
            dur     = vid.get("duration_seconds")
            dur_str = f"{dur:.1f}s" if dur else "?"
            lines.append(f"**Video ID:** {vid['id']}  ·  **Family:** `{vid.get('model_family','?')}`  ·  **Pipeline:** `{vid.get('pipeline','?')}`")
            lines.append(f"**Status:** {vid.get('status','?')}")
            lines.append(f"**Res:** {vid.get('width','?')}×{vid.get('height','?')}  ·  **FPS:** {vid.get('fps','?')}  ·  **Frames:** {vid.get('num_frames','?')}  ·  **Duration:** {dur_str}")
        lines.append(f"**Seed:** `{vid.get('seed','?')}`  ·  **Steps:** {vid.get('steps','?')}  ·  **CFG:** {vid.get('guidance_scale','?')}")
        lines.append(f"**Project:** {vid.get('project_name') or '—'}  ·  **Character:** {vid.get('character_name') or '—'}  ·  **Location:** {vid.get('location_name') or '—'}")
        if vid.get("lora_path"):
            lines.append(f"**LoRA:** `{vid['lora_path']}` @ {vid.get('lora_weight', 1.0)}")
        lines.append(f"\n**Prompt:**\n> {vid.get('prompt', '') or '—'}")
        if vid.get("negative_prompt"):
            lines.append(f"\n**Negative:**\n> {vid['negative_prompt']}")
        lines.append(f"\n*Created: {(vid.get('created_at') or '')[:19]} UTC*")
        return "\n\n".join(lines)

    # ---------------------------------------------------------------------------
    # Model manager callbacks
    # ---------------------------------------------------------------------------

    def on_save_config_wan(wan_path, i2v_path, device, dtype):
        db.save_wan_config(wan_path, device, dtype)
        if i2v_path:
            db.set_setting("wan_i2v_model_path", i2v_path)
        return "<div style='padding:8px;color:#22c55e;'>✅ Wan paths saved.</div>"

    def on_detect_wan(wan_path):
        from studio.generation import wan_manager
        return _status_html(wan_manager.detect_models(wan_path), wan_manager.get_loaded_pipeline_name())

    def on_load_pipe_wan(pipe_name, wan_path, i2v_path, device, dtype):
        from studio.generation import wan_manager
        # Use I2V path if available and requested
        active_path = i2v_path if (pipe_name == "image2video" and i2v_path) else wan_path
        _, msg = wan_manager.load_pipeline(pipe_name, active_path, device, dtype)
        return on_detect_wan(wan_path) + f"<p style='color:#e5e7eb;'>{msg}</p>"

    def on_save_config_hunyuan(hun_path, device, dtype):
        db.save_hunyuan_config(hun_path, device, dtype)
        return "<div style='padding:8px;color:#22c55e;'>✅ Hunyuan paths saved.</div>"

    def on_detect_hunyuan(hun_path):
        from studio.generation import hunyuan_manager
        return _status_html(hunyuan_manager.detect_models(hun_path), hunyuan_manager.get_loaded_pipeline_name())

    def on_load_pipe_hunyuan(pipe_name, hun_path, device, dtype):
        from studio.generation import hunyuan_manager
        _, msg = hunyuan_manager.load_pipeline(pipe_name, hun_path, device, dtype)
        return on_detect_hunyuan(hun_path) + f"<p style='color:#e5e7eb;'>{msg}</p>"

    def on_save_config_ltx(ltx, gemma, upscaler, lora_dir, device, dtype):
        db.save_ltx_config(ltx_model_path=ltx, ltx_gemma_path=gemma, ltx_upscaler_path=upscaler, ltx_lora_dir=lora_dir, ltx_device=device, ltx_dtype=dtype)
        return "<div style='padding:8px;color:#22c55e;'>✅ LTX paths saved.</div>"

    def on_detect_ltx(ltx, gemma, upscaler, lora_dir):
        from studio.generation import ltx_manager
        return _status_html(ltx_manager.detect_models(ltx, gemma, upscaler, lora_dir), ltx_manager.get_loaded_pipeline_name())

    def on_load_pipe_ltx(pipe_name, ltx, gemma, upscaler, lora_dir, device, dtype, lora_sel, lora_wt):
        from studio.generation import ltx_manager
        lora_full = os.path.join(lora_dir, lora_sel) if lora_sel and lora_sel != "None" and lora_dir else ""
        _, msg = ltx_manager.load_pipeline(pipe_name, ltx, upscaler, gemma, device, dtype, lora_full, float(lora_wt))
        return on_detect_ltx(ltx, gemma, upscaler, lora_dir) + f"<p style='color:#e5e7eb;'>{msg}</p>"

    def on_save_config_ltx2(repo, python_path, ckpt, upsamp, lora, gemma, module, quant, extra):
        db.set_setting("ltx2_repo_dir", repo or "")
        db.set_setting("ltx2_venv_python", python_path or "")
        db.set_setting("ltx2_checkpoint_path", ckpt or "")
        db.set_setting("ltx2_upsampler_path", upsamp or "")
        db.set_setting("ltx2_distilled_lora", lora or "")
        db.set_setting("ltx2_gemma_root", gemma or "")
        db.set_setting("ltx2_module", module or "ltx_pipelines.ti2vid_two_stages")
        db.set_setting("ltx2_quantization", quant or "")
        db.set_setting("ltx2_extra_args", extra or "")
        return "<div style='padding:8px;color:#22c55e;'>✅ LTX-2.3 paths saved.</div>"

    def on_detect_ltx2():
        from studio.generation import ltx2_manager
        return _status_html(ltx2_manager.detect_models(db), None)

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
        return "Unknown family."

    # ---------------------------------------------------------------------------
    # Workflow callbacks
    # ---------------------------------------------------------------------------

    def on_family_change(family):
        """Switch pipeline choices AND reset all parameters to that model's production defaults."""
        presets = _FAMILY_PRESETS.get(family, _WAN_PRESETS)
        p       = presets.get("production", list(presets.values())[0])

        if family == "LTX-Video":
            pipe_choices = ["distilled", "two_stage", "two_stage_hq"]
            pipe_val     = "distilled"
        elif family == "LTX-2.3":
            pipe_choices = _LTX2_PIPELINES
            pipe_val     = "two_stage"
        else:
            pipe_choices = ["text2video", "image2video"]
            pipe_val     = "text2video"

        return (
            gr.update(choices=pipe_choices, value=pipe_val),
            gr.update(value=p["width"]),
            gr.update(value=p["height"]),
            gr.update(value=p["fps"]),
            gr.update(value=p["num_frames"]),
            gr.update(value=p["steps"]),
            gr.update(value=p["guidance_scale"]),
        )

    def on_pipeline_change(pipeline_name):
        """Show I2V guidance banner when image2video is selected."""
        if pipeline_name == "image2video":
            html = (
                "<div style='padding:8px 12px;background:rgba(99,102,241,0.15);"
                "border:1px solid #6366f1;border-radius:8px;font-size:0.85rem;'>"
                "🖼️ <b>Image-to-Video active</b> — upload a character reference image below. "
                "Wan I2V will animate from that frame, locking in the face and wardrobe."
                "</div>"
            )
            return gr.update(value=html, visible=True)
        return gr.update(value="<div></div>", visible=False)

    def on_preset_change(preset_val, family):
        if preset_val == "custom":
            return (gr.update(),) * 7
        presets = _FAMILY_PRESETS.get(family, _WAN_PRESETS)
        p       = presets.get(preset_val, presets.get("production", {}))

        if family == "LTX-Video":
            pipe_choices = ["distilled", "two_stage", "two_stage_hq"]
        elif family == "LTX-2.3":
            pipe_choices = _LTX2_PIPELINES
        else:
            pipe_choices = ["text2video", "image2video"]

        return (
            gr.update(value=p.get("width",          480)),
            gr.update(value=p.get("height",         832)),
            gr.update(value=p.get("fps",             16)),
            gr.update(value=p.get("num_frames",      81)),
            gr.update(value=p.get("steps",           50)),
            gr.update(value=p.get("guidance_scale", 6.0)),
            gr.update(choices=pipe_choices, value=p.get("pipeline", pipe_choices[0])),
        )

    def on_queue_render(prompt, neg_prompt, family, pipeline_name, preset_val,
                        w, h, fps, nf, steps, cfg,
                        lora_dir, lora_sel, lora_wt,
                        seed, ref_img,
                        proj, char, loc, shot):
        proj_id = None
        char_id = None
        loc_id  = None
        shot_id = None
        try:
            if proj and proj != "All":
                r = db.get_project_by_name(proj)
                proj_id = r["id"] if r else None
            if char and char != "All":
                r = db.get_character_by_name(char)
                char_id = r["id"] if r else None
            if loc and loc != "All":
                r = db.get_location_by_name(loc)
                loc_id = r["id"] if r else None
            if shot:
                r = db.get_shot_template_by_name(shot)
                shot_id = r["id"] if r else None
        except Exception:
            pass

        actual_seed = int(seed) if int(seed) != -1 else random.randint(0, 2 ** 32 - 1)
        lora_full   = os.path.join(lora_dir, lora_sel) if lora_sel and lora_sel != "None" and lora_dir else ""

        # ref_img is now a filepath string from gr.Image(type="filepath")
        ref_path = ref_img if ref_img and os.path.isfile(str(ref_img)) else ""

        # Auto-switch to I2V pipeline if reference provided and family supports it
        if ref_path and family in ("Wan 2.2",) and pipeline_name == "text2video":
            pipeline_name = "image2video"

        job_id = db.add_to_render_queue(
            project_id=proj_id,
            scene_id=None,
            character_id=char_id,
            location_id=loc_id,
            shot_type_id=shot_id,
            model_family=family,
            pipeline=pipeline_name,
            preset=preset_val,
            prompt=prompt,
            negative_prompt=neg_prompt,
            seed=actual_seed,
            width=int(w),
            height=int(h),
            fps=int(fps),
            num_frames=int(nf),
            steps=int(steps),
            guidance_scale=float(cfg),
            lora_path=lora_full,
            lora_weight=float(lora_wt),
            reference_image_path=ref_path,
        )
        mode_note = " (I2V — character locked)" if pipeline_name == "image2video" and ref_path else ""
        return f"✅ **Job #{job_id} Queued!** {family}{mode_note} — background worker will process it."

    def on_refresh_queue():
        jobs = db.list_render_jobs(limit=50)
        return [_job_row(j) for j in jobs], f"**{len(jobs)} recent jobs.**"

    def on_select_queue(evt: gr.SelectData, df):
        job_id   = int(df.iloc[evt.index[0], 0])
        job      = db.get_render_job(job_id)
        vid_path = None
        if job and job.get("video_id"):
            vid      = db.get_generated_video(job["video_id"])
            vid_path = vid.get("file_path") if vid else None
        return vid_path, _meta_md(job, is_job=True), job_id

    def on_q_cancel(sel):
        if sel == -1: return "⚠️ Select a job."
        job = db.get_render_job(int(sel))
        if job and job["status"] == "queued":
            db.update_render_job_status(int(sel), "cancelled")
            return f"✅ Job **#{int(sel)}** cancelled."
        return "⚠️ Can only cancel queued jobs."

    def on_q_approve(sel):
        if sel == -1: return "⚠️ Select a job."
        job = db.get_render_job(int(sel))
        if job and job["status"] == "completed" and job.get("video_id"):
            db.update_video_status(job["video_id"], "approved")
            return f"✅ Video **#{job['video_id']}** approved to Library."
        return "⚠️ Can only approve completed jobs."

    def on_q_reject(sel):
        if sel == -1: return "⚠️ Select a job."
        job = db.get_render_job(int(sel))
        if job and job["status"] == "completed" and job.get("video_id"):
            db.update_video_status(job["video_id"], "rejected")
            return f"❌ Video **#{job['video_id']}** rejected."
        return "⚠️ Can only reject completed jobs."

    _lib_records: list = []

    def on_lib_refresh(proj_f, char_f, loc_f, date_f, fam_f, pipe_f):
        nonlocal _lib_records
        proj_id = None
        char_id = None
        loc_id  = None
        try:
            if proj_f and proj_f != "All":
                r = db.get_project_by_name(proj_f); proj_id = r["id"] if r else None
            if char_f and char_f != "All":
                r = db.get_character_by_name(char_f); char_id = r["id"] if r else None
            if loc_f and loc_f != "All":
                r = db.get_location_by_name(loc_f); loc_id = r["id"] if r else None
        except Exception:
            pass

        all_vids = db.list_generated_videos(
            project_id=proj_id, character_id=char_id, location_id=loc_id,
            status="approved",
            pipeline=pipe_f if pipe_f != "All" else None,
            model_family=fam_f if fam_f != "All" else None,
        )

        if date_f != "All Time":
            now      = datetime.utcnow()
            filtered = []
            for v in all_vids:
                if not v.get("created_at"):
                    continue
                try:
                    dt = datetime.strptime(v["created_at"], "%Y-%m-%d %H:%M:%S")
                    if date_f == "Today" and (now - dt).days < 1:
                        filtered.append(v)
                    elif date_f == "Last 7 Days" and (now - dt).days <= 7:
                        filtered.append(v)
                except Exception:
                    pass
            all_vids = filtered

        _lib_records = all_vids
        items = [
            (v.get("thumbnail_path") or v.get("file_path"),
             f"{v.get('model_family','?')} · {v.get('pipeline','?')} · seed:{v.get('seed','?')}")
            for v in _lib_records
        ]
        return [i for i in items if i[0]], f"**{len(_lib_records)} videos found.**", None, "*Select a thumbnail.*"

    def on_lib_select(evt: gr.SelectData):
        vid = _lib_records[evt.index]
        return vid.get("file_path"), _meta_md(vid, is_job=False), vid["id"]

    def on_refresh_loras(lora_dir):
        if not lora_dir or not os.path.isdir(lora_dir):
            return gr.update(choices=["None"], value="None")
        files = ["None"] + sorted(f for f in os.listdir(lora_dir) if f.endswith((".safetensors", ".pt", ".bin")))
        return gr.update(choices=files, value="None")

    # ---------------------------------------------------------------------------
    # Wire everything up
    # ---------------------------------------------------------------------------

    # Model manager
    btn_save_cfg_wan.click(on_save_config_wan, [mm_wan_path, mm_wan_i2v, mm_wan_device, mm_wan_dtype], [mm_status_html_wan])
    btn_detect_wan.click(on_detect_wan, [mm_wan_path], [mm_status_html_wan])
    btn_load_pipe_wan.click(on_load_pipe_wan, [vid_pipeline, mm_wan_path, mm_wan_i2v, mm_wan_device, mm_wan_dtype], [mm_status_html_wan])
    btn_unload_wan.click(lambda: on_unload("Wan 2.2"), [], [mm_status_html_wan])

    btn_save_cfg_hunyuan.click(on_save_config_hunyuan, [mm_hunyuan_path, mm_hunyuan_device, mm_hunyuan_dtype], [mm_status_html_hunyuan])
    btn_detect_hunyuan.click(on_detect_hunyuan, [mm_hunyuan_path], [mm_status_html_hunyuan])
    btn_load_pipe_hunyuan.click(on_load_pipe_hunyuan, [vid_pipeline, mm_hunyuan_path, mm_hunyuan_device, mm_hunyuan_dtype], [mm_status_html_hunyuan])
    btn_unload_hunyuan.click(lambda: on_unload("Hunyuan Video"), [], [mm_status_html_hunyuan])

    btn_save_cfg_ltx.click(on_save_config_ltx, [mm_ltx_path, mm_gemma_path, mm_upscaler_path, mm_lora_dir, mm_device, mm_dtype], [mm_status_html_ltx])
    btn_detect_ltx.click(on_detect_ltx, [mm_ltx_path, mm_gemma_path, mm_upscaler_path, mm_lora_dir], [mm_status_html_ltx])

    btn_save_cfg_ltx2.click(on_save_config_ltx2, [mm_ltx2_repo, mm_ltx2_python, mm_ltx2_ckpt, mm_ltx2_upsamp, mm_ltx2_lora, mm_ltx2_gemma, mm_ltx2_module, mm_ltx2_quant, mm_ltx2_extra], [mm_status_html_ltx2])
    btn_detect_ltx2.click(on_detect_ltx2, [], [mm_status_html_ltx2])
    btn_load_pipe_ltx.click(on_load_pipe_ltx, [vid_pipeline, mm_ltx_path, mm_gemma_path, mm_upscaler_path, mm_lora_dir, mm_device, mm_dtype, vid_lora_selector, vid_lora_weight], [mm_status_html_ltx])
    btn_unload_ltx.click(lambda: on_unload("LTX-Video"), [], [mm_status_html_ltx])

    # Workflow
    vid_model_family.change(
        on_family_change,
        [vid_model_family],
        [vid_pipeline, vid_width, vid_height, vid_fps, vid_frames, vid_steps, vid_guidance],
    )
    vid_pipeline.change(on_pipeline_change, [vid_pipeline], [i2v_info])
    vid_preset.change(
        on_preset_change,
        [vid_preset, vid_model_family],
        [vid_width, vid_height, vid_fps, vid_frames, vid_steps, vid_guidance, vid_pipeline],
    )
    btn_refresh_loras.click(on_refresh_loras, [mm_lora_dir], [vid_lora_selector])

    btn_queue_render.click(
        on_queue_render,
        [vid_prompt, vid_neg_prompt, vid_model_family, vid_pipeline, vid_preset,
         vid_width, vid_height, vid_fps, vid_frames, vid_steps, vid_guidance,
         mm_lora_dir, vid_lora_selector, vid_lora_weight,
         vid_seed, vid_ref_image,
         vid_proj_selector, vid_char_selector, vid_loc_selector, vid_shot_selector],
        [vid_action_msg],
    )

    btn_refresh_queue.click(on_refresh_queue, [], [queue_df, queue_count_md])
    queue_df.select(on_select_queue, [queue_df], [queue_vid_player, queue_meta_md, queue_sel_id])
    btn_q_cancel.click(on_q_cancel,  [queue_sel_id], [queue_action_msg])
    btn_q_approve.click(on_q_approve, [queue_sel_id], [queue_action_msg])
    btn_q_reject.click(on_q_reject,  [queue_sel_id], [queue_action_msg])

    btn_lib_refresh.click(
        on_lib_refresh,
        [lib_filter_proj, lib_filter_char, lib_filter_loc, lib_filter_date, lib_filter_family, lib_filter_pipeline],
        [lib_gallery, lib_action_msg, lib_vid_player, lib_detail_md],
    )
    lib_gallery.select(on_lib_select, [], [lib_vid_player, lib_detail_md, lib_sel_id])

    return vid_proj_selector, vid_char_selector, vid_loc_selector, vid_shot_selector
