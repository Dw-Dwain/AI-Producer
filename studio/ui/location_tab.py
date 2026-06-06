import os
import shutil
import gradio as gr
from studio.database.db_manager import DatabaseManager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCATIONS_DIR = os.path.join(BASE_DIR, "locations")

def create_location_tab(db: DatabaseManager):
    with gr.TabItem("Locations"):
        gr.Markdown("### 🏢 Set & Location Management Engine (Phase 3)")
        
        with gr.Row():
            # ==========================================
            # LEFT PANEL: SET BUILDER & ROOMS MANAGER
            # ==========================================
            with gr.Column(scale=5, variant="panel"):
                gr.Markdown("#### 📝 Location Builder")
                
                # Active location database ID tracker
                loc_edit_id = gr.State(None)
                
                with gr.Tabs():
                    with gr.Tab("Main Set Details"):
                        loc_name = gr.Textbox(label="Location Name (Unique)", placeholder="e.g. Japanese Apartment")
                        loc_desc = gr.Textbox(label="Visual Description (Flux Prompt Helper)", placeholder="Details for image generator...", lines=3)
                        loc_notes = gr.Textbox(label="Director Notes", placeholder="Mood, lighting, styling details...", lines=3)
                        loc_tags = gr.Textbox(label="Tags (Comma separated)", placeholder="e.g. indoor, cozy, modern")
                        loc_prompt_template = gr.Textbox(
                            label="Prompt Template",
                            value="A high-fidelity photo of {name}, {description}. Cinematic lighting, 35mm architectural photography.",
                            lines=3
                        )
                        loc_images = gr.File(label="Upload Reference Photos to Set", file_count="multiple", file_types=["image"])
                        
                        with gr.Row():
                            btn_save_loc = gr.Button("Save Location Set", variant="primary")
                            btn_clear_loc = gr.Button("Clear Form", variant="secondary")
                        loc_msg = gr.Markdown("")
                        
                    with gr.Tab("Rooms (Sub-locations)"):
                        gr.Markdown("##### 🚪 Manage Rooms/Sub-locations for Selected Set")
                        
                        room_selector_edit = gr.Dropdown(label="Select Room to Edit", choices=[], interactive=True)
                        room_name = gr.Textbox(label="Room Name", placeholder="e.g. Living Room")
                        room_desc = gr.Textbox(label="Room Visual Description", placeholder="Details about this specific room...", lines=3)
                        room_prompt_template = gr.Textbox(
                            label="Room Prompt Template",
                            value="{parent_prompt}, specifically in the {name} area, featuring {description}.",
                            lines=3
                        )
                        
                        with gr.Row():
                            btn_save_room = gr.Button("Save/Update Room", variant="primary")
                            btn_delete_room = gr.Button("Delete Selected Room", variant="stop")
                        room_msg = gr.Markdown("")
                
            # ==========================================
            # RIGHT PANEL: CATALOG, FILTERS & PORTFOLIO
            # ==========================================
            with gr.Column(scale=7, variant="panel"):
                gr.Markdown("#### 🗇 Location Directory Catalog")
                
                # Search & Filter Block
                with gr.Accordion("🔍 Search & Dynamic Filters", open=True):
                    with gr.Row():
                        search_q = gr.Textbox(label="Search by Name/Description/Notes", placeholder="e.g. neon")
                        filter_tag = gr.Dropdown(label="Filter Tag", choices=["All"], value="All")
                    with gr.Row():
                        filter_project = gr.Dropdown(label="Filter Project Casting", choices=["All"], value="All")
                        filter_char = gr.Dropdown(label="Filter Character Casting", choices=["All"], value="All")
                    btn_clear_filters = gr.Button("Reset Filters", variant="secondary")
                
                gr.Markdown("---")
                loc_selector = gr.Dropdown(label="Select Location Set", choices=[])
                
                # Detailed Display Card
                with gr.Group():
                    loc_detail_name = gr.Markdown("### Select a Location from the directory list")
                    
                    with gr.Row():
                        loc_detail_meta = gr.Markdown("")
                        loc_detail_folder = gr.Markdown("")
                    
                    # Scope Selector for Portfolio Details
                    portfolio_scope = gr.Dropdown(
                        label="Portfolio Scope (Main Set or Room)", 
                        choices=["Main Location Set"], 
                        value="Main Location Set"
                    )
                    
                    with gr.Tabs():
                        with gr.Tab("Description & Notes"):
                            loc_detail_desc = gr.Markdown("")
                            loc_detail_notes = gr.Markdown("")
                            
                        with gr.Tab("Compiled Prompt Preview"):
                            gr.Markdown("##### 🎨 Compiled Prompt Preview")
                            compiled_prompt_preview = gr.Textbox(label="Live Compiled Prompt", interactive=False, lines=4)
                            btn_compile_prompt = gr.Button("Re-Compile Prompt", variant="secondary")
                            
                        with gr.Tab("Rooms (Sub-locations) Index"):
                            gr.Markdown("##### 🚪 Registered Rooms Index")
                            rooms_dataframe = gr.Dataframe(
                                headers=["Room Name", "Description", "Folder Path"],
                                datatype=["str", "str", "str"],
                                interactive=False
                            )
                            
                        with gr.Tab("Asset Gallery"):
                            gr.Markdown("##### 📁 Asset Portfolio Gallery")
                            loc_gallery = gr.Gallery(label="Reference Photos", columns=4, object_fit="contain", height="auto")
                            
                            gr.Markdown("---")
                            gr.Markdown("##### 📤 Add Photos to Current Scope")
                            loc_add_images = gr.File(label="Choose Photos to Upload", file_count="multiple", file_types=["image"])
                            btn_upload_loc_photos = gr.Button("Upload Photos", variant="secondary")
                            loc_portfolio_msg = gr.Markdown("")
                            
                    with gr.Row():
                        btn_load_loc_edit = gr.Button("Load Set into Builder", variant="secondary")
                        btn_delete_loc = gr.Button("Delete Location Set", variant="stop")

        # ==========================================
        # EVENT FUNCTIONS
        # ==========================================
        def get_gallery_images(folder_path):
            if not folder_path or not os.path.exists(folder_path):
                return []
            valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
            images = []
            for f in os.listdir(folder_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    images.append(os.path.join(folder_path, f))
            return images

        def get_filter_dropdown_updates():
            locs = db.list_locations()
            
            # Tags list
            all_tags = set()
            for l in locs:
                if l["tags"]:
                    tags = [t.strip() for t in l["tags"].split(",") if t.strip()]
                    all_tags.update(tags)
            tag_choices = ["All"] + sorted(list(all_tags))
            
            # Projects list
            projs = db.list_projects()
            proj_choices = ["All"] + [p["name"] for p in projs]
            
            # Characters list
            chars = db.list_characters()
            char_choices = ["All"] + [c["name"] for c in chars]
            
            return (
                gr.update(choices=tag_choices),
                gr.update(choices=proj_choices),
                gr.update(choices=char_choices)
            )

        def compile_location_prompt(loc_obj, sub_loc_obj=None):
            parent_template = loc_obj.get("prompt_template") or "A high-fidelity photo of {name}, {description}."
            parent_dict = {
                "name": loc_obj.get("name") or "",
                "description": loc_obj.get("description") or "",
                "notes": loc_obj.get("notes") or ""
            }
            try:
                compiled_parent = parent_template.format(**parent_dict)
            except Exception as e:
                compiled_parent = f"⚠️ Template error in main set: {e}"
            
            if not sub_loc_obj:
                return compiled_parent
                
            room_template = sub_loc_obj.get("prompt_template") or "{parent_prompt}, specifically in the {name} area, featuring {description}."
            room_dict = {
                "name": sub_loc_obj.get("name") or "",
                "description": sub_loc_obj.get("description") or "",
                "parent_prompt": compiled_parent
            }
            try:
                compiled_room = room_template.format(**room_dict)
                return compiled_room
            except Exception as e:
                return f"⚠️ Template error in room prompt: {e}"

        def save_location_fn(edit_id, name, desc, notes, tags, prompt_template, files):
            if not name or not name.strip():
                return gr.update(), "⚠️ Location name is required.", edit_id, gr.update(), gr.update(), gr.update(), gr.update()
            
            clean_name = name.strip()
            folder_name = clean_name.lower().replace(" ", "_")
            loc_folder = os.path.join(LOCATIONS_DIR, folder_name)
            
            if edit_id is None:
                # Add new location
                os.makedirs(loc_folder, exist_ok=True)
                loc_id = db.add_location(
                    name=clean_name,
                    description=desc.strip(),
                    tags=tags.strip(),
                    notes=notes.strip(),
                    folder_path=loc_folder,
                    prompt_template=prompt_template.strip()
                )
                if not loc_id:
                    return gr.update(), "⚠️ A location with this name already exists.", edit_id, gr.update(), gr.update(), gr.update(), gr.update()
                msg = f"✅ Location '{clean_name}' created successfully."
            else:
                # Update existing location
                loc = db.get_location(edit_id)
                if not loc:
                    return gr.update(), "⚠️ Location to update not found.", edit_id, gr.update(), gr.update(), gr.update(), gr.update()
                
                db.update_location(
                    loc_id=edit_id,
                    name=clean_name,
                    description=desc.strip(),
                    tags=tags.strip(),
                    notes=notes.strip(),
                    prompt_template=prompt_template.strip()
                )
                loc_id = edit_id
                
                # Rename directory if name changed
                old_folder = loc["folder_path"]
                if old_folder != loc_folder:
                    if os.path.exists(old_folder):
                        try:
                            if not os.path.exists(loc_folder):
                                os.rename(old_folder, loc_folder)
                            else:
                                for f in os.listdir(old_folder):
                                    shutil.move(os.path.join(old_folder, f), os.path.join(loc_folder, f))
                                os.rmdir(old_folder)
                        except Exception as e:
                            print(f"Error renaming location folder: {e}")
                            loc_folder = old_folder
                    else:
                        os.makedirs(loc_folder, exist_ok=True)
                    # Update DB path
                    with db._get_connection() as conn:
                        conn.execute("UPDATE locations SET folder_path = ? WHERE id = ?", (loc_folder, loc_id))
                        conn.commit()
                msg = f"✅ Location '{clean_name}' updated successfully."

            # Save uploaded files
            if files:
                os.makedirs(loc_folder, exist_ok=True)
                for f in files:
                    dest = os.path.join(loc_folder, os.path.basename(f.name))
                    try:
                        shutil.copy(f.name, dest)
                    except Exception as e:
                        print(f"Error saving file {f.name}: {e}")

            locs_list = db.list_locations()
            names = [l["name"] for l in locs_list]
            updates = get_filter_dropdown_updates()
            
            return (
                gr.update(choices=names, value=clean_name),
                msg,
                loc_id, # Keep loaded
                gr.update(value=None),
                updates[0],
                updates[1],
                updates[2]
            )

        def select_location_fn(name):
            if not name:
                return (
                    None,
                    "### Select a Location Set",
                    "",
                    "",
                    "",
                    "",
                    "",
                    gr.update(choices=[], value=None),
                    [],
                    gr.update(choices=["Main Location Set"], value="Main Location Set"),
                    [],
                    ""
                )
            
            loc = db.get_location_by_name(name)
            if not loc:
                return (
                    None,
                    "### Selected location set not found",
                    "",
                    "",
                    "",
                    "",
                    "",
                    gr.update(choices=[], value=None),
                    [],
                    gr.update(choices=["Main Location Set"], value="Main Location Set"),
                    [],
                    ""
                )
            
            title_md = f"## 🏢 {loc['name']}"
            meta_md = f"**Tags:** `{loc['tags'] or 'None'}`"
            folder_md = f"📂 **System Path:** `{loc['folder_path']}`"
            desc_md = f"**Visual Description:**\n{loc['description'] or '*No description added.*'}"
            notes_md = f"**Director & Set Notes:**\n{loc['notes'] or '*No notes added.*'}"
            
            # Fetch rooms
            rooms = db.list_sub_locations(loc["id"])
            room_names = [r["name"] for r in rooms]
            room_df_rows = [[r["name"], r["description"] or "", r["folder_path"]] for r in rooms]
            
            compiled_prompt = compile_location_prompt(loc)
            gallery = get_gallery_images(loc['folder_path'])
            
            return (
                loc['id'],
                title_md,
                meta_md,
                folder_md,
                desc_md,
                notes_md,
                compiled_prompt,
                gr.update(choices=room_names, value=room_names[0] if room_names else None),
                room_df_rows,
                gr.update(choices=["Main Location Set"] + room_names, value="Main Location Set"),
                gallery,
                ""
            )

        def load_loc_edit_form(edit_id):
            if edit_id is None:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Select a location first to load into the builder."
            
            loc = db.get_location(edit_id)
            if not loc:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Location not found."
            
            return (
                loc['name'],
                loc['description'] or "",
                loc['notes'] or "",
                loc['tags'] or "",
                loc['prompt_template'] or "A high-fidelity photo of {name}, {description}. Cinematic lighting, 35mm architectural photography.",
                f"📝 Loaded '{loc['name']}' into builder. Modify fields and click 'Save Location' to update."
            )

        def clear_loc_form_fn():
            return None, "", "", "", "", "A high-fidelity photo of {name}, {description}. Cinematic lighting, 35mm architectural photography.", "🧹 Form cleared. Ready to create a new location set."

        def delete_location_fn(edit_id, name):
            target_id = edit_id
            if target_id is None and name:
                loc = db.get_location_by_name(name)
                if loc:
                    target_id = loc["id"]
            
            if target_id is None:
                return gr.update(), "⚠️ Please select a location set to delete.", None, gr.update(), gr.update(), gr.update()
            
            loc = db.get_location(target_id)
            if loc:
                db.delete_location(target_id)
                if os.path.exists(loc["folder_path"]):
                    try:
                        shutil.rmtree(loc["folder_path"])
                    except Exception as e:
                        print(f"Error removing location folder: {e}")
            
            locs_list = db.list_locations()
            names = [l["name"] for l in locs_list]
            updates = get_filter_dropdown_updates()
            
            return (
                gr.update(choices=names, value=names[0] if names else None), 
                f"❌ Location Set '{loc['name'] if loc else 'Unknown'}' deleted.", 
                None,
                updates[0],
                updates[1],
                updates[2]
            )

        def change_scope_fn(loc_id, scope_name):
            if loc_id is None:
                return "", "", "", []
            
            loc = db.get_location(loc_id)
            if not loc:
                return "", "", "", []
                
            if scope_name == "Main Location Set" or not scope_name:
                desc_md = f"**Visual Description:**\n{loc['description'] or '*No description added.*'}"
                notes_md = f"**Director & Set Notes:**\n{loc['notes'] or '*No notes added.*'}"
                compiled = compile_location_prompt(loc)
                gallery = get_gallery_images(loc['folder_path'])
                return desc_md, notes_md, compiled, gallery
            else:
                # Fetch Room
                with db._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM sub_locations WHERE location_id = ? AND name = ?",
                        (loc_id, scope_name)
                    ).fetchone()
                if row:
                    room = dict(row)
                    desc_md = f"**Room Description:**\n{room['description'] or '*No description added.*'}"
                    notes_md = f"📂 **System Path:** `{room['folder_path']}`\n\n**Room Prompt Template:** `{room['prompt_template']}`"
                    compiled = compile_location_prompt(loc, room)
                    gallery = get_gallery_images(room['folder_path'])
                    return desc_md, notes_md, compiled, gallery
                else:
                    return "⚠️ Room not found.", "", "", []

        def save_sub_location_fn(loc_id, room_edit_name, room_name, room_desc, room_prompt):
            if loc_id is None:
                return gr.update(), "⚠️ Please select a Location Set first from the catalog directory.", gr.update()
            
            if not room_name or not room_name.strip():
                return gr.update(), "⚠️ Room name is required.", gr.update()
            
            clean_room_name = room_name.strip()
            loc = db.get_location(loc_id)
            if not loc:
                return gr.update(), "⚠️ Selected Location Set not found.", gr.update()
                
            room_slug = clean_room_name.lower().replace(" ", "_")
            room_folder = os.path.join(loc["folder_path"], room_slug)
            
            if room_edit_name:
                # Editing existing room
                with db._get_connection() as conn:
                    existing = conn.execute(
                        "SELECT * FROM sub_locations WHERE location_id = ? AND name = ?", 
                        (loc_id, room_edit_name.strip())
                    ).fetchone()
                
                if existing:
                    existing = dict(existing)
                    old_folder = existing["folder_path"]
                    
                    with db._get_connection() as conn:
                        conn.execute(
                            """UPDATE sub_locations 
                               SET name = ?, description = ?, prompt_template = ?, folder_path = ? 
                               WHERE id = ?""",
                            (clean_room_name, room_desc.strip(), room_prompt.strip(), room_folder, existing["id"])
                        )
                        conn.commit()
                        
                    # Handle folder rename/move
                    if old_folder != room_folder:
                        if os.path.exists(old_folder):
                            try:
                                if not os.path.exists(room_folder):
                                    os.rename(old_folder, room_folder)
                                else:
                                    for f in os.listdir(old_folder):
                                        shutil.move(os.path.join(old_folder, f), os.path.join(room_folder, f))
                                    os.rmdir(old_folder)
                            except Exception as e:
                                print(f"Error renaming room folder: {e}")
                                room_folder = old_folder
                                with db._get_connection() as conn:
                                    conn.execute("UPDATE sub_locations SET folder_path = ? WHERE id = ?", (room_folder, existing["id"]))
                                    conn.commit()
                    msg = f"✅ Room '{clean_room_name}' updated successfully."
                else:
                    return gr.update(), "⚠️ Selected room to edit not found in database.", gr.update()
            else:
                # Create new room
                os.makedirs(room_folder, exist_ok=True)
                sub_loc_id = db.add_sub_location(
                    location_id=loc_id,
                    name=clean_room_name,
                    description=room_desc.strip(),
                    prompt_template=room_prompt.strip(),
                    folder_path=room_folder
                )
                if not sub_loc_id:
                    return gr.update(), "⚠️ A room with this name already exists in this location.", gr.update()
                msg = f"✅ Room '{clean_room_name}' added to set '{loc['name']}'."
                
            rooms = db.list_sub_locations(loc_id)
            room_names = [r["name"] for r in rooms]
            
            return gr.update(choices=room_names, value=clean_room_name), msg, gr.update(choices=["Main Location Set"] + room_names, value=clean_room_name)

        def load_room_edit_form_fn(loc_id, room_name):
            if not room_name or loc_id is None:
                return "", "", "{parent_prompt}, specifically in the {name} area, featuring {description}."
            
            with db._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM sub_locations WHERE location_id = ? AND name = ?", 
                    (loc_id, room_name)
                ).fetchone()
            
            if row:
                row = dict(row)
                return row["name"], row["description"] or "", row["prompt_template"] or ""
            return "", "", ""

        def delete_sub_location_fn(loc_id, room_name):
            if loc_id is None or not room_name:
                return gr.update(), "⚠️ Select a room to delete.", gr.update()
            
            with db._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM sub_locations WHERE location_id = ? AND name = ?", 
                    (loc_id, room_name)
                ).fetchone()
            
            if row:
                row = dict(row)
                db.delete_sub_location(row["id"])
                if os.path.exists(row["folder_path"]):
                    try:
                        shutil.rmtree(row["folder_path"])
                    except Exception as e:
                        print(f"Error removing sub-location folder: {e}")
                msg = f"❌ Room '{room_name}' deleted."
            else:
                msg = f"⚠️ Room '{room_name}' not found."
                
            rooms = db.list_sub_locations(loc_id)
            room_names = [r["name"] for r in rooms]
            return gr.update(choices=room_names, value=room_names[0] if room_names else None), msg, gr.update(choices=["Main Location Set"] + room_names, value="Main Location Set")

        def add_location_photos_fn(loc_id, scope_name, files):
            if loc_id is None:
                return [], "⚠️ Select a location first to add photos.", gr.update(value=None)
            
            loc = db.get_location(loc_id)
            if not loc:
                return [], "⚠️ Selected location set not found.", gr.update(value=None)
            
            dest_dir = loc["folder_path"]
            if scope_name and scope_name != "Main Location Set":
                with db._get_connection() as conn:
                    row = conn.execute(
                        "SELECT folder_path FROM sub_locations WHERE location_id = ? AND name = ?",
                        (loc_id, scope_name)
                    ).fetchone()
                if row:
                    dest_dir = row["folder_path"]
                    
            os.makedirs(dest_dir, exist_ok=True)
            
            if files:
                for f in files:
                    dest = os.path.join(dest_dir, os.path.basename(f.name))
                    try:
                        shutil.copy(f.name, dest)
                    except Exception as e:
                        print(f"Error copying image: {e}")
            
            gallery = get_gallery_images(dest_dir)
            return gallery, f"✅ Added {len(files) if files else 0} image(s) to scope '{scope_name}'.", gr.update(value=None)

        def filter_locations_fn(q, tag, proj_name, char_name):
            proj_id = None
            if proj_name and proj_name != "All":
                p = db.get_project_by_name(proj_name)
                if p:
                    proj_id = p["id"]
                    
            char_id = None
            if char_name and char_name != "All":
                c = db.get_character_by_name(char_name)
                if c:
                    char_id = c["id"]
                    
            locs = db.search_locations(
                search_query=q.strip() if q else None,
                tag=tag if tag != "All" else None,
                project_id=proj_id,
                character_id=char_id
            )
            names = [l["name"] for l in locs]
            return gr.update(choices=names, value=names[0] if names else None)

        def reset_filters_fn():
            return "", "All", "All", "All"

        def trigger_recompile_fn(loc_id, scope_name):
            if loc_id is None:
                return ""
            loc = db.get_location(loc_id)
            if not loc:
                return ""
            if scope_name == "Main Location Set" or not scope_name:
                return compile_location_prompt(loc)
            else:
                with db._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM sub_locations WHERE location_id = ? AND name = ?",
                        (loc_id, scope_name)
                    ).fetchone()
                if row:
                    room = dict(row)
                    return compile_location_prompt(loc, room)
                return ""

        # Wire up listeners
        loc_selector.change(
            fn=select_location_fn,
            inputs=[loc_selector],
            outputs=[
                loc_edit_id, loc_detail_name, loc_detail_meta, loc_detail_folder,
                loc_detail_desc, loc_detail_notes, compiled_prompt_preview,
                room_selector_edit, rooms_dataframe, portfolio_scope, loc_gallery, loc_portfolio_msg
            ]
        )
        
        btn_save_loc.click(
            fn=save_location_fn,
            inputs=[loc_edit_id, loc_name, loc_desc, loc_notes, loc_tags, loc_prompt_template, loc_images],
            outputs=[loc_selector, loc_msg, loc_edit_id, loc_images, filter_tag, filter_project, filter_char]
        )
        
        btn_clear_loc.click(
            fn=clear_loc_form_fn,
            outputs=[loc_edit_id, loc_name, loc_desc, loc_notes, loc_tags, loc_prompt_template, loc_msg]
        )
        
        btn_load_loc_edit.click(
            fn=load_loc_edit_form,
            inputs=[loc_edit_id],
            outputs=[loc_name, loc_desc, loc_notes, loc_tags, loc_prompt_template, loc_msg]
        )
        
        btn_delete_loc.click(
            fn=delete_location_fn,
            inputs=[loc_edit_id, loc_selector],
            outputs=[loc_selector, loc_portfolio_msg, loc_edit_id, filter_tag, filter_project, filter_char]
        )
        
        portfolio_scope.change(
            fn=change_scope_fn,
            inputs=[loc_edit_id, portfolio_scope],
            outputs=[loc_detail_desc, loc_detail_notes, compiled_prompt_preview, loc_gallery]
        )
        
        btn_save_room.click(
            fn=save_sub_location_fn,
            inputs=[loc_edit_id, room_selector_edit, room_name, room_desc, room_prompt_template],
            outputs=[room_selector_edit, room_msg, portfolio_scope]
        )
        
        room_selector_edit.change(
            fn=load_room_edit_form_fn,
            inputs=[loc_edit_id, room_selector_edit],
            outputs=[room_name, room_desc, room_prompt_template]
        )
        
        btn_delete_room.click(
            fn=delete_sub_location_fn,
            inputs=[loc_edit_id, room_selector_edit],
            outputs=[room_selector_edit, room_msg, portfolio_scope]
        )
        
        btn_upload_loc_photos.click(
            fn=add_location_photos_fn,
            inputs=[loc_edit_id, portfolio_scope, loc_add_images],
            outputs=[loc_gallery, loc_portfolio_msg, loc_add_images]
        )
        
        btn_compile_prompt.click(
            fn=trigger_recompile_fn,
            inputs=[loc_edit_id, portfolio_scope],
            outputs=[compiled_prompt_preview]
        )
        
        inputs_filter = [search_q, filter_tag, filter_project, filter_char]
        for inp in inputs_filter:
            inp.change(fn=filter_locations_fn, inputs=inputs_filter, outputs=[loc_selector])
            
        btn_clear_filters.click(
            fn=reset_filters_fn,
            outputs=inputs_filter
        )

        return loc_selector, filter_tag, filter_project, filter_char
