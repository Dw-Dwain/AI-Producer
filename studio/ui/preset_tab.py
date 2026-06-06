import gradio as gr
from studio.database.db_manager import DatabaseManager

def create_preset_tab(db: DatabaseManager):
    with gr.TabItem("Render Presets"):
        gr.Markdown("### ⚙️ Video & Image Render Presets")
        
        with gr.Row():
            # Left panel: Save/Edit Preset
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("#### 📝 Edit Render Preset")
                preset_name = gr.Textbox(label="Preset Name (Unique)", placeholder="e.g. Wan Cinematic 1080p")
                
                with gr.Row():
                    preset_width = gr.Number(label="Width (pixels)", value=720, precision=0)
                    preset_height = gr.Number(label="Height (pixels)", value=1280, precision=0)
                
                with gr.Row():
                    preset_fps = gr.Number(label="FPS", value=24, precision=0)
                    preset_frames = gr.Number(label="Frame Count", value=96, precision=0)
                
                preset_model = gr.Dropdown(
                    label="Active Model Target", 
                    choices=["LTX-2", "Wan 2.2", "Hunyuan Video", "Flux.1", "Other"],
                    value="LTX-2"
                )
                preset_pipeline = gr.Textbox(label="Pipeline / Sub-model", placeholder="e.g. Distilled, I2V-14B, Standard")
                preset_motion = gr.Textbox(label="Default Camera Motion Setting", placeholder="e.g. Slow tracking pan, static")
                preset_notes = gr.Textbox(label="Preset Notes & Specs", placeholder="Optimizations, vertical aspects...", lines=2)
                
                with gr.Row():
                    btn_save_preset = gr.Button("Save/Update Preset", variant="primary")
                    btn_clear_preset = gr.Button("Clear Form", variant="secondary")
                preset_msg = gr.Markdown("")

            # Right panel: Preset Directory
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("#### 🗄️ Preset Library")
                preset_selector = gr.Dropdown(label="Select Preset to view/load", choices=[])
                
                with gr.Group():
                    gr.Markdown("#### Current Preset Specifications")
                    preset_details_md = gr.Markdown("*Select a preset from the library to view details.*")
                    btn_load_preset = gr.Button("Load into Editor", variant="secondary")
                    btn_delete_preset = gr.Button("Delete Preset from Library", variant="stop")
                preset_library_msg = gr.Markdown("")

        # ==========================================
        # EVENT FUNCTIONS
        # ==========================================
        def load_presets_list():
            presets = db.list_presets()
            names = [p["name"] for p in presets]
            return gr.update(choices=names, value=names[0] if names else None)

        def save_preset_fn(name, w, h, fps, frames, model, pipe, motion, notes):
            if not name or not name.strip():
                return gr.update(), "⚠️ Preset name is required.", gr.update()
            
            clean_name = name.strip()
            db.add_preset(
                name=clean_name,
                width=int(w),
                height=int(h),
                fps=int(fps),
                frame_count=int(frames),
                model=model,
                pipeline=pipe.strip() if pipe else None,
                camera_motion=motion.strip() if motion else None,
                notes=notes.strip() if notes else None
            )
            
            presets = db.list_presets()
            names = [p["name"] for p in presets]
            return gr.update(choices=names, value=clean_name), f"✅ Preset '{clean_name}' saved/updated.", gr.update()

        def select_preset_fn(name):
            if not name:
                return "*Select a preset from the library to view details.*"
            preset = db.get_preset(name)
            if not preset:
                return "*Preset not found.*"
            
            details = f"""
### ⚙️ {preset['name']}
- **Resolution:** {preset['width']}x{preset['height']}
- **FPS:** {preset['fps']} frames/sec
- **Frames:** {preset['frame_count']} total frames
- **Model Target:** {preset['model']}
- **Pipeline Setup:** `{preset['pipeline'] or 'Default'}`
- **Camera Motion:** `{preset['camera_motion'] or 'None'}`
- **Notes:** {preset['notes'] or 'None'}
"""
            return details

        def load_preset_into_editor(name):
            if not name:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Select a preset first."
            preset = db.get_preset(name)
            if not preset:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Preset not found."
            
            return (
                preset["name"],
                preset["width"],
                preset["height"],
                preset["fps"],
                preset["frame_count"],
                preset["model"],
                preset["pipeline"] or "",
                preset["camera_motion"] or "",
                preset["notes"] or "",
                f"📝 Loaded '{preset['name']}' specifications into editor."
            )

        def clear_preset_form():
            return "", 720, 1280, 24, 96, "LTX-2", "", "", "", "🧹 Form cleared."

        def delete_preset_fn(name):
            if not name:
                return gr.update(), "⚠️ Select a preset to delete."
            db.delete_preset(name)
            
            presets = db.list_presets()
            names = [p["name"] for p in presets]
            return gr.update(choices=names, value=names[0] if names else None), f"❌ Preset '{name}' deleted."

        # Wire up listeners
        preset_selector.change(fn=select_preset_fn, inputs=[preset_selector], outputs=[preset_details_md])
        
        btn_save_preset.click(
            fn=save_preset_fn,
            inputs=[preset_name, preset_width, preset_height, preset_fps, preset_frames, preset_model, preset_pipeline, preset_motion, preset_notes],
            outputs=[preset_selector, preset_msg, preset_library_msg]
        )
        
        btn_clear_preset.click(
            fn=clear_preset_form,
            outputs=[preset_name, preset_width, preset_height, preset_fps, preset_frames, preset_model, preset_pipeline, preset_motion, preset_notes, preset_msg]
        )
        
        btn_load_preset.click(
            fn=load_preset_into_editor,
            inputs=[preset_selector],
            outputs=[preset_name, preset_width, preset_height, preset_fps, preset_frames, preset_model, preset_pipeline, preset_motion, preset_notes, preset_library_msg]
        )
        
        btn_delete_preset.click(
            fn=delete_preset_fn,
            inputs=[preset_selector],
            outputs=[preset_selector, preset_library_msg]
        )

        return preset_selector,

