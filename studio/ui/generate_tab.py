"""
studio/ui/generate_tab.py
Phase 4: 🖼️ Image Generation Tab — Flux Dev / Flux Kontext pipeline UI.

Layout (three sub-tabs):
  1. 🎨 Generate    — prompt builder + model params + live preview + approval
  2. ⏳ Pending     — queue of pending images awaiting review
  3. 📚 Library     — filtered gallery of approved images
  + Model Manager accordion (path config, detect, load, unload)
"""

import os
import logging
import gradio as gr
from studio.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

STUDIO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Internal state holders (module-level, shared across callbacks)
# ---------------------------------------------------------------------------
_current_image_id: int | None = None   # DB id of the last generated image


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

def create_generate_tab(db: DatabaseManager):
    """
    Build the Image Generation tab and wire all events.

    Returns:
        (gen_proj_selector, gen_char_selector, gen_loc_selector)
        — the three context dropdowns, so app.py can include them in
          the session-restore callback.
    """
    with gr.TabItem("🖼️ Image Generation"):

        # ----------------------------------------------------------------
        # MODEL MANAGER ACCORDION (top-of-tab, always visible)
        # ----------------------------------------------------------------
        with gr.Accordion("⚙️ Model Manager", open=False):
            gr.Markdown(
                "Point to your **local** Flux model folders or `.safetensors` files. "
                "No models are downloaded automatically."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    mm_dev_path = gr.Textbox(
                        label="Flux Dev — Local Path",
                        placeholder="/path/to/flux-dev  or  /path/to/flux-dev.safetensors",
                        interactive=True,
                    )
                    mm_kontext_path = gr.Textbox(
                        label="Flux Kontext — Local Path",
                        placeholder="/path/to/flux-kontext  or  /path/to/flux-kontext.safetensors",
                        interactive=True,
                    )
                with gr.Column(scale=1):
                    mm_device = gr.Dropdown(
                        label="Device",
                        choices=["cuda", "cpu", "mps"],
                        value="cuda",
                        interactive=True,
                    )
                    mm_dtype = gr.Dropdown(
                        label="dtype",
                        choices=["bfloat16", "float16", "float32"],
                        value="bfloat16",
                        interactive=True,
                    )

            with gr.Row():
                btn_detect   = gr.Button("🔍 Detect Models",     variant="secondary")
                btn_save_cfg = gr.Button("💾 Save Paths",        variant="secondary")
                btn_load_mdl = gr.Button("⚡ Load Selected Model", variant="primary")
                btn_unload   = gr.Button("🗑️ Unload Model",      variant="stop")

            mm_status_html = gr.HTML(
                "<div style='padding:8px;color:#9ca3af;font-size:0.85rem;'>"
                "Status will appear here after Detect or Load.</div>"
            )

        # ----------------------------------------------------------------
        # THREE SUB-TABS
        # ----------------------------------------------------------------
        with gr.Tabs():

            # ==============================================================
            # SUB-TAB 1: GENERATE
            # ==============================================================
            with gr.TabItem("🎨 Generate"):
                with gr.Row():

                    # ---- LEFT: Context + Prompt Builder ----
                    with gr.Column(scale=2, variant="panel"):
                        gr.Markdown("#### 🎭 Context")
                        with gr.Row():
                            gen_proj_selector = gr.Dropdown(
                                label="Project",
                                choices=[],
                                value=None,
                                interactive=True,
                                elem_id="gen_proj_selector",
                            )
                            gen_char_selector = gr.Dropdown(
                                label="Character",
                                choices=[],
                                value=None,
                                interactive=True,
                                elem_id="gen_char_selector",
                            )
                            gen_loc_selector = gr.Dropdown(
                                label="Location",
                                choices=[],
                                value=None,
                                interactive=True,
                                elem_id="gen_loc_selector",
                            )

                        gr.Markdown("---")
                        gr.Markdown("#### ✍️ Prompt Builder")

                        gen_prompt = gr.Textbox(
                            label="Prompt",
                            lines=5,
                            placeholder="Describe the scene — or use Auto-Fill to seed from character DNA…",
                            interactive=True,
                            elem_id="gen_prompt_box",
                        )
                        with gr.Row():
                            btn_autofill   = gr.Button("🧬 Auto-Fill from Character DNA", variant="secondary")
                            btn_scene_fill = gr.Button("🎬 Load Scene Prompt",            variant="secondary")

                        gen_neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            placeholder="worst quality, blurry, deformed…",
                            interactive=True,
                        )

                    # ---- RIGHT: Model & Parameters ----
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("#### 🤖 Model & Parameters")

                        gen_model = gr.Radio(
                            label="Model",
                            choices=["flux_dev", "flux_kontext"],
                            value="flux_dev",
                            interactive=True,
                        )

                        gen_width = gr.Slider(
                            label="Width", minimum=512, maximum=2048,
                            step=64, value=1024, interactive=True,
                        )
                        gen_height = gr.Slider(
                            label="Height", minimum=512, maximum=2048,
                            step=64, value=1024, interactive=True,
                        )
                        gen_steps = gr.Slider(
                            label="Steps", minimum=1, maximum=100,
                            step=1, value=28, interactive=True,
                        )
                        gen_guidance = gr.Slider(
                            label="Guidance Scale", minimum=1.0, maximum=20.0,
                            step=0.5, value=3.5, interactive=True,
                        )
                        gen_seed = gr.Number(
                            label="Seed  (−1 = random)", value=-1,
                            precision=0, interactive=True,
                        )

                        btn_generate = gr.Button(
                            "🚀 Generate Image", variant="primary", scale=1
                        )
                        gen_status = gr.Markdown(
                            "Ready. Load a model and hit **Generate**."
                        )

                # ---- BOTTOM: Preview + Actions ----
                gr.Markdown("---")
                gr.Markdown("#### 🖼️ Preview")
                with gr.Row():
                    gen_preview = gr.Image(
                        label="Generated Image",
                        type="pil",
                        interactive=False,
                        elem_id="gen_preview_img",
                        height=512,
                    )
                    with gr.Column(scale=1):
                        gen_meta_md = gr.Markdown("*Generate an image to see metadata here.*")
                        with gr.Row():
                            btn_approve  = gr.Button("✅ Approve",        variant="primary")
                            btn_reject   = gr.Button("❌ Reject",         variant="stop")
                        with gr.Row():
                            btn_reroll   = gr.Button("🔄 Re-roll Seed",   variant="secondary")
                            btn_save_lib = gr.Button("💾 Save to Library", variant="secondary")
                        gen_action_msg = gr.Markdown("")

            # ==============================================================
            # SUB-TAB 2: PENDING REVIEW
            # ==============================================================
            with gr.TabItem("⏳ Pending Review"):
                gr.Markdown("#### Images awaiting your approval decision")
                with gr.Row():
                    btn_refresh_pending = gr.Button("🔄 Refresh List", variant="secondary")
                    pending_count_md    = gr.Markdown("*Click Refresh to load.*")

                with gr.Row():
                    with gr.Column(scale=2):
                        pending_df = gr.Dataframe(
                            headers=["ID", "Model", "Character", "Project", "Seed", "Created"],
                            datatype=["number", "str", "str", "str", "number", "str"],
                            row_count=10,
                            col_count=6,
                            interactive=False,
                            label="Pending Images",
                            elem_id="pending_df",
                        )
                    with gr.Column(scale=1):
                        pending_preview = gr.Image(
                            label="Selected Image",
                            type="filepath",
                            interactive=False,
                            height=350,
                        )
                        pending_meta_md = gr.Markdown("*Select a row to preview.*")
                        pending_sel_id  = gr.Number(label="Selected ID", value=-1, visible=False)

                with gr.Row():
                    btn_pend_approve = gr.Button("✅ Approve Selected", variant="primary")
                    btn_pend_reject  = gr.Button("❌ Reject Selected",  variant="stop")
                    btn_pend_delete  = gr.Button("🗑️ Delete Record",   variant="secondary")

                pending_action_msg = gr.Markdown("")

            # ==============================================================
            # SUB-TAB 3: APPROVED LIBRARY
            # ==============================================================
            with gr.TabItem("📚 Approved Library"):
                gr.Markdown("#### Your approved image library")
                with gr.Row():
                    lib_filter_proj  = gr.Dropdown(
                        label="Filter by Project",
                        choices=["All"],
                        value="All",
                        interactive=True,
                    )
                    lib_filter_char  = gr.Dropdown(
                        label="Filter by Character",
                        choices=["All"],
                        value="All",
                        interactive=True,
                    )
                    lib_filter_model = gr.Dropdown(
                        label="Filter by Model",
                        choices=["All", "flux_dev", "flux_kontext"],
                        value="All",
                        interactive=True,
                    )
                    btn_lib_refresh  = gr.Button("🔄 Refresh Gallery", variant="secondary")

                lib_gallery = gr.Gallery(
                    label="Approved Images",
                    columns=4,
                    object_fit="cover",
                    height=500,
                    elem_id="lib_gallery",
                )

                gr.Markdown("---")
                gr.Markdown("#### 📋 Image Details")
                with gr.Row():
                    lib_detail_md = gr.Markdown("*Click an image in the gallery above to view its metadata.*")

                with gr.Row():
                    lib_export_path  = gr.Textbox(
                        label="Export destination path",
                        placeholder="C:/Users/me/Desktop/export/",
                        interactive=True,
                        scale=3,
                    )
                    btn_lib_export   = gr.Button("📤 Export Selected", variant="secondary", scale=1)
                    btn_lib_remove   = gr.Button("🗑️ Remove from Library", variant="stop", scale=1)

                lib_action_msg   = gr.Markdown("")
                lib_sel_id       = gr.Number(label="Selected Gallery ID", value=-1, visible=False)

    # ======================================================================
    # EVENT CALLBACKS
    # ======================================================================

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fmt_status_html(detect_result: dict, loaded_name: str | None) -> str:
        """Build HTML status cards for the model manager."""
        labels = {"flux_dev": "FLUX DEV", "flux_kontext": "FLUX KONTEXT"}
        parts = []
        for key, label in labels.items():
            info = detect_result.get(key, {"found": False, "path": "", "files": []})
            dot   = "🟢" if info["found"] else "🔴"
            badge = ""
            if loaded_name == key:
                badge = (
                    "<span style='background:#22c55e;color:#000;padding:2px 8px;"
                    "border-radius:4px;font-size:0.75rem;margin-left:6px;'>LOADED</span>"
                )
            path_str = info.get("path", "") or "—"
            n_files  = len(info.get("files", []))
            parts.append(
                f"<div style='padding:6px 12px;margin:4px 0;border-radius:8px;"
                f"background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'>"
                f"{dot} <b>{label}</b>{badge} &nbsp;·&nbsp; "
                f"<code style='font-size:0.8rem;color:#a5b4fc;'>{path_str}</code>"
                f"<span style='color:#6b7280;font-size:0.8rem;margin-left:8px;'>"
                f"({n_files} files detected)</span></div>"
            )
        return "".join(parts)

    def _image_row_to_df(img: dict) -> list:
        return [
            img["id"],
            img.get("model", ""),
            img.get("character_name") or "—",
            img.get("project_name") or "—",
            img.get("seed", -1),
            (img.get("created_at") or "")[:16],
        ]

    def _meta_md(img: dict) -> str:
        if not img:
            return "*No image selected.*"
        lines = [
            f"**ID:** {img['id']}  ·  **Model:** `{img.get('model','?')}`",
            f"**Status:** {img.get('status','?')}",
            f"**Seed:** `{img.get('seed', '?')}`  ·  "
            f"**Size:** {img.get('width','?')}×{img.get('height','?')}  ·  "
            f"**Steps:** {img.get('steps','?')}  ·  "
            f"**CFG:** {img.get('guidance_scale','?')}",
            f"**Project:** {img.get('project_name') or '—'}  ·  "
            f"**Character:** {img.get('character_name') or '—'}  ·  "
            f"**Location:** {img.get('location_name') or '—'}",
            f"\n**Prompt:**\n> {img.get('prompt','') or '—'}",
        ]
        if img.get("negative_prompt"):
            lines.append(f"\n**Negative:**\n> {img['negative_prompt']}")
        lines.append(f"\n*Created: {(img.get('created_at') or '')[:19]} UTC*")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Model Manager callbacks
    # ------------------------------------------------------------------

    def on_save_config(dev_path, kontext_path, device, dtype):
        db.save_model_config(
            flux_dev_path=dev_path,
            flux_kontext_path=kontext_path,
            flux_device=device,
            flux_dtype=dtype,
        )
        return (
            gr.update(value=dev_path),
            gr.update(value=kontext_path),
            "<div style='padding:8px;color:#22c55e;'>✅ Paths saved to session memory.</div>",
        )

    def on_detect(dev_path, kontext_path):
        from studio.generation import flux_manager
        result = flux_manager.detect_models(dev_path, kontext_path)
        loaded = flux_manager.get_loaded_model_name()
        return _fmt_status_html(result, loaded)

    def on_load_model(model_name, dev_path, kontext_path, device, dtype):
        from studio.generation import flux_manager
        _pipe, msg = flux_manager.load_model(
            model_name, dev_path, kontext_path, device, dtype
        )
        result = flux_manager.detect_models(dev_path, kontext_path)
        loaded = flux_manager.get_loaded_model_name()
        return _fmt_status_html(result, loaded) + f"<p style='padding:4px 12px;color:#e5e7eb;'>{msg}</p>"

    def on_unload_model():
        from studio.generation import flux_manager
        msg = flux_manager.unload_model()
        return f"<div style='padding:8px;color:#e5e7eb;'>{msg}</div>"

    # ------------------------------------------------------------------
    # Generate tab callbacks
    # ------------------------------------------------------------------

    def on_autofill(char_name):
        if not char_name:
            return gr.update(value=""), "⚠️ Select a character first."
        char = db.get_character_by_name(char_name)
        if not char:
            return gr.update(value=""), "⚠️ Character not found in DB."

        parts = []
        if char.get("prompt_template"):
            parts.append(char["prompt_template"])
        else:
            # Build from DNA
            dna_parts = []
            for field in ["dna_ethnicity", "dna_body_type", "dna_hair", "dna_eyes",
                          "dna_clothing", "dna_description"]:
                val = char.get(field)
                if val:
                    dna_parts.append(val)
            if dna_parts:
                parts.append(", ".join(dna_parts))
            if char.get("description"):
                parts.append(char["description"])

        prompt_text = ". ".join(parts) if parts else f"Portrait of {char['name']}"
        return gr.update(value=prompt_text), f"✅ Auto-filled from **{char['name']}** DNA."

    def on_scene_fill(proj_name, char_name, loc_name):
        """Pull the most recent scene prompt for the active context."""
        if not proj_name:
            return gr.update(), "⚠️ Select a project first."
        proj = db.get_project_by_name(proj_name)
        if not proj:
            return gr.update(), "⚠️ Project not found."

        char_id = None
        if char_name:
            c = db.get_character_by_name(char_name)
            if c:
                char_id = c["id"]

        # Find a scene with a prompt under this project
        with db._get_connection() as conn:
            row = conn.execute(
                """SELECT s.prompt FROM scenes s
                   JOIN episodes e ON s.episode_id = e.id
                   WHERE e.project_id = ?
                     AND (? IS NULL OR s.character_id = ?)
                     AND s.prompt IS NOT NULL AND s.prompt != ''
                   ORDER BY s.rowid DESC LIMIT 1""",
                (proj["id"], char_id, char_id),
            ).fetchone()

        if not row:
            return gr.update(), "ℹ️ No scene prompt found for this context."
        return gr.update(value=row["prompt"]), "✅ Scene prompt loaded."

    def on_generate(
        prompt, neg_prompt, model_name,
        width, height, steps, guidance, seed,
        proj_name, char_name, loc_name,
    ):
        global _current_image_id

        from studio.generation import flux_manager, output_manager

        pipe = flux_manager._cached_pipeline
        if pipe is None:
            return (
                None,
                "❌ **No model loaded.** Open Model Manager and load a model first.",
                "*No image generated yet.*",
            )

        pil_img, actual_seed, err = flux_manager.generate_image(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=int(width),
            height=int(height),
            steps=int(steps),
            seed=int(seed),
            guidance_scale=float(guidance),
        )
        if err or pil_img is None:
            return None, err or "❌ Unknown generation error.", "*Generation failed.*"

        # Resolve IDs
        proj_id = char_id = loc_id = None
        proj_obj = db.get_project_by_name(proj_name) if proj_name else None
        if proj_obj:
            proj_id = proj_obj["id"]
        char_obj = db.get_character_by_name(char_name) if char_name else None
        if char_obj:
            char_id = char_obj["id"]
        loc_obj = db.get_location_by_name(loc_name) if loc_name else None
        if loc_obj:
            loc_id = loc_obj["id"]

        # Save to disk
        out_dir = output_manager.get_output_dir(STUDIO_ROOT, proj_name or "", char_name or "")
        meta    = output_manager.build_metadata(
            prompt=prompt,
            negative_prompt=neg_prompt,
            seed=actual_seed,
            model=model_name,
            width=int(width),
            height=int(height),
            steps=int(steps),
            guidance_scale=float(guidance),
            character_name=char_name or "",
            location_name=loc_name or "",
            project_name=proj_name or "",
        )
        file_path, thumb_path = output_manager.save_image(pil_img, out_dir, meta)

        # Store in DB
        img_id = db.add_generated_image(
            file_path=file_path,
            model=model_name,
            prompt=prompt,
            negative_prompt=neg_prompt,
            seed=actual_seed,
            width=int(width),
            height=int(height),
            steps=int(steps),
            guidance_scale=float(guidance),
            thumbnail_path=thumb_path,
            project_id=proj_id,
            character_id=char_id,
            location_id=loc_id,
        )
        _current_image_id = img_id

        status_msg = (
            f"✅ **Generated!** Seed: `{actual_seed}` · Saved to `{file_path}`"
        )
        img_record = db.get_generated_image(img_id)
        meta_display = _meta_md(img_record)
        return pil_img, status_msg, meta_display

    def on_approve_current():
        global _current_image_id
        if _current_image_id is None:
            return "⚠️ No image to approve. Generate an image first."
        db.update_image_status(_current_image_id, "approved")
        return f"✅ Image **#{_current_image_id}** approved and added to Library."

    def on_reject_current():
        global _current_image_id
        if _current_image_id is None:
            return "⚠️ No image to reject."
        db.update_image_status(_current_image_id, "rejected")
        return f"❌ Image **#{_current_image_id}** rejected."

    def on_reroll(
        prompt, neg_prompt, model_name,
        width, height, steps, guidance, _seed,
        proj_name, char_name, loc_name,
    ):
        return on_generate(
            prompt, neg_prompt, model_name,
            width, height, steps, guidance, -1,
            proj_name, char_name, loc_name,
        )

    # ------------------------------------------------------------------
    # Pending Review callbacks
    # ------------------------------------------------------------------

    def on_refresh_pending():
        imgs = db.list_generated_images(status="pending")
        if not imgs:
            return [], "ℹ️ No images pending review."
        rows = [_image_row_to_df(i) for i in imgs]
        return rows, f"**{len(imgs)} image(s) pending review.**"

    def on_select_pending(evt: gr.SelectData, df_data):
        if evt.index is None or df_data is None:
            return None, "*Select a row.*", -1
        row_idx = evt.index[0]
        try:
            img_id = int(df_data.iloc[row_idx, 0])
        except Exception:
            return None, "*Could not read row.*", -1
        img = db.get_generated_image(img_id)
        if not img:
            return None, "*Record not found.*", -1
        fp = img.get("file_path", "")
        preview = fp if fp and os.path.exists(fp) else None
        return preview, _meta_md(img), img_id

    def on_pend_approve(sel_id):
        if sel_id < 0:
            return "⚠️ No image selected."
        db.update_image_status(int(sel_id), "approved")
        return f"✅ Image **#{int(sel_id)}** approved."

    def on_pend_reject(sel_id):
        if sel_id < 0:
            return "⚠️ No image selected."
        db.update_image_status(int(sel_id), "rejected")
        return f"❌ Image **#{int(sel_id)}** rejected."

    def on_pend_delete(sel_id):
        if sel_id < 0:
            return "⚠️ No image selected."
        db.delete_generated_image(int(sel_id))
        return f"🗑️ Record **#{int(sel_id)}** deleted."

    # ------------------------------------------------------------------
    # Approved Library callbacks
    # ------------------------------------------------------------------

    def _build_gallery(proj_filter, char_filter, model_filter):
        proj_id = char_id = None
        if proj_filter and proj_filter != "All":
            p = db.get_project_by_name(proj_filter)
            if p:
                proj_id = p["id"]
        if char_filter and char_filter != "All":
            c = db.get_character_by_name(char_filter)
            if c:
                char_id = c["id"]

        model_q = model_filter if (model_filter and model_filter != "All") else None
        imgs = db.list_generated_images(
            project_id=proj_id,
            character_id=char_id,
            status="approved",
            model=model_q,
        )

        gallery_items = []
        for img in imgs:
            fp = img.get("file_path", "")
            if fp and os.path.exists(fp):
                caption = (
                    f"{img.get('character_name') or '—'} | "
                    f"seed:{img.get('seed', '?')} | "
                    f"{img.get('model', '?')}"
                )
                gallery_items.append((fp, caption))

        return gallery_items, imgs  # imgs used to track IDs by position

    # We keep a module-level list to map gallery selection → DB ids
    _lib_img_records: list = []

    def on_lib_refresh(proj_filter, char_filter, model_filter):
        nonlocal _lib_img_records
        items, records = _build_gallery(proj_filter, char_filter, model_filter)
        _lib_img_records = records
        if not items:
            return [], "*No approved images found for this filter.*"
        return items, f"**{len(items)} approved image(s)** in library."

    def on_lib_select(evt: gr.SelectData):
        nonlocal _lib_img_records
        if evt.index is None or evt.index >= len(_lib_img_records):
            return "*Select an image above.*", -1
        img = _lib_img_records[evt.index]
        return _meta_md(img), img["id"]

    def on_lib_export(sel_id, export_path):
        import shutil
        if sel_id < 0:
            return "⚠️ No image selected in gallery."
        if not export_path or not os.path.isdir(export_path):
            return "⚠️ Invalid or non-existent export path."
        img = db.get_generated_image(int(sel_id))
        if not img:
            return "⚠️ Image record not found."
        src = img.get("file_path", "")
        if not src or not os.path.exists(src):
            return "⚠️ Image file not found on disk."
        dest = os.path.join(export_path, os.path.basename(src))
        shutil.copy2(src, dest)
        return f"✅ Exported to `{dest}`"

    def on_lib_remove(sel_id):
        if sel_id < 0:
            return "⚠️ No image selected."
        db.update_image_status(int(sel_id), "rejected")
        return f"🗑️ Image **#{int(sel_id)}** removed from library (marked rejected)."

    def on_autofill_char_dropdown(char_name):
        pil, msg = on_autofill(char_name)
        return pil, msg

    # ------------------------------------------------------------------
    # Wire up all events
    # ------------------------------------------------------------------

    # Model Manager
    btn_save_cfg.click(
        fn=on_save_config,
        inputs=[mm_dev_path, mm_kontext_path, mm_device, mm_dtype],
        outputs=[mm_dev_path, mm_kontext_path, mm_status_html],
    )
    btn_detect.click(
        fn=on_detect,
        inputs=[mm_dev_path, mm_kontext_path],
        outputs=[mm_status_html],
    )
    btn_load_mdl.click(
        fn=on_load_model,
        inputs=[gen_model, mm_dev_path, mm_kontext_path, mm_device, mm_dtype],
        outputs=[mm_status_html],
    )
    btn_unload.click(fn=on_unload_model, outputs=[mm_status_html])

    # Generate tab
    btn_autofill.click(
        fn=on_autofill,
        inputs=[gen_char_selector],
        outputs=[gen_prompt, gen_status],
    )
    btn_scene_fill.click(
        fn=on_scene_fill,
        inputs=[gen_proj_selector, gen_char_selector, gen_loc_selector],
        outputs=[gen_prompt, gen_status],
    )
    btn_generate.click(
        fn=on_generate,
        inputs=[
            gen_prompt, gen_neg_prompt, gen_model,
            gen_width, gen_height, gen_steps, gen_guidance, gen_seed,
            gen_proj_selector, gen_char_selector, gen_loc_selector,
        ],
        outputs=[gen_preview, gen_status, gen_meta_md],
    )
    btn_approve.click(fn=on_approve_current,   outputs=[gen_action_msg])
    btn_reject.click(fn=on_reject_current,     outputs=[gen_action_msg])
    btn_reroll.click(
        fn=on_reroll,
        inputs=[
            gen_prompt, gen_neg_prompt, gen_model,
            gen_width, gen_height, gen_steps, gen_guidance, gen_seed,
            gen_proj_selector, gen_char_selector, gen_loc_selector,
        ],
        outputs=[gen_preview, gen_status, gen_meta_md],
    )
    btn_save_lib.click(fn=on_approve_current, outputs=[gen_action_msg])

    # Pending Review
    btn_refresh_pending.click(
        fn=on_refresh_pending,
        outputs=[pending_df, pending_count_md],
    )
    pending_df.select(
        fn=on_select_pending,
        inputs=[pending_df],
        outputs=[pending_preview, pending_meta_md, pending_sel_id],
    )
    btn_pend_approve.click(
        fn=on_pend_approve, inputs=[pending_sel_id], outputs=[pending_action_msg]
    )
    btn_pend_reject.click(
        fn=on_pend_reject, inputs=[pending_sel_id], outputs=[pending_action_msg]
    )
    btn_pend_delete.click(
        fn=on_pend_delete, inputs=[pending_sel_id], outputs=[pending_action_msg]
    )

    # Approved Library
    btn_lib_refresh.click(
        fn=on_lib_refresh,
        inputs=[lib_filter_proj, lib_filter_char, lib_filter_model],
        outputs=[lib_gallery, lib_detail_md],
    )
    lib_gallery.select(
        fn=on_lib_select,
        outputs=[lib_detail_md, lib_sel_id],
    )
    btn_lib_export.click(
        fn=on_lib_export,
        inputs=[lib_sel_id, lib_export_path],
        outputs=[lib_action_msg],
    )
    btn_lib_remove.click(
        fn=on_lib_remove, inputs=[lib_sel_id], outputs=[lib_action_msg]
    )

    return gen_proj_selector, gen_char_selector, gen_loc_selector
