import os
import shutil
import gradio as gr
from studio.database.db_manager import DatabaseManager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARACTERS_DIR = os.path.join(BASE_DIR, "characters")

def create_character_tab(db: DatabaseManager):
    with gr.TabItem("Characters"):
        gr.Markdown("### 🎭 Professional Character Engine (Phase 2)")
        
        with gr.Row():
            # ==========================================
            # LEFT PANEL: CHARACTER BUILDER & RELATIONSHIPS
            # ==========================================
            with gr.Column(scale=5, variant="panel"):
                gr.Markdown("#### 📝 Character Profile Builder")
                
                # Active character database ID tracker
                char_edit_id = gr.State(None)
                
                # Tabbed accordions for editor organization
                with gr.Tabs():
                    with gr.Tab("Core Profile"):
                        char_name = gr.Textbox(label="Character Name (Unique)", placeholder="e.g. Agent Kenji")
                        with gr.Row():
                            char_age = gr.Number(label="Age", value=30, precision=0)
                            char_gender = gr.Textbox(label="Gender", placeholder="e.g. Male / Non-binary")
                        
                        char_biography = gr.Textbox(label="Biography / Backstory", placeholder="Deep history, motivations, past events...", lines=3)
                        char_personality = gr.Textbox(label="Personality & Quirks", placeholder="Behavioral traits, speech patterns, emotional triggers...", lines=2)
                        char_voice = gr.Textbox(label="Voice notes / Tone characteristics", placeholder="e.g. Deep baritone, gravelly voice, slow cadence", lines=2)
                        char_tags = gr.Textbox(label="Tags (Comma separated)", placeholder="e.g. cyberpunk, detective, cyber-arm")
                        char_desc = gr.Textbox(label="Appearance description (Flux Prompt Helper)", placeholder="Overall look, general appearance description...", lines=2)
                        char_notes = gr.Textbox(label="Additional notes", placeholder="General reminders, metadata notes...", lines=2)

                    with gr.Tab("Character DNA"):
                        gr.Markdown("##### 🧬 Reusable prompt attributes")
                        dna_hair = gr.Textbox(label="Hair style / Color", placeholder="e.g. Spiky midnight black hair")
                        dna_eyes = gr.Textbox(label="Eyes style / Color", placeholder="e.g. Glowing cyan cybernetic eyes")
                        dna_body = gr.Textbox(label="Body Type / Build", placeholder="e.g. Tall, athletic, cybernetic left shoulder")
                        dna_clothing = gr.Textbox(label="Signature Wardrobe / Clothing", placeholder="e.g. Worn dark leather trench coat over high-tech collar")
                        dna_ethnicity = gr.Textbox(label="Ethnicity / Heritage", placeholder="e.g. East-Asian heritage")
                        dna_description = gr.Textbox(label="Extra Visual Traits", placeholder="e.g. Faint scar on left cheek, cybernetic interface plug on temple", lines=2)

                    with gr.Tab("Prompt Template"):
                        gr.Markdown("##### 🎨 Prompt Compiler")
                        char_prompt_template = gr.Textbox(
                            label="Character Prompt Template",
                            value="A premium portrait of {name}, {age} years old, {gender}, {dna_ethnicity}, featuring {dna_hair}, {dna_eyes}, {dna_body_type}, wearing {dna_clothing}, {dna_description}. Cinematic lighting, 35mm photograph, high fidelity.",
                            lines=4
                        )
                        btn_compile_prompt = gr.Button("Preview Compiled Prompt", variant="secondary")
                        compiled_prompt_preview = gr.Textbox(label="Compiled Result", interactive=False, lines=3)

                    with gr.Tab("Relationships"):
                        gr.Markdown("##### 🔗 Character Connections")
                        rel_target = gr.Dropdown(label="Connect to Character", choices=[])
                        rel_type = gr.Dropdown(
                            label="Relationship Type", 
                            choices=["Ally", "Rival", "Enemy", "Friend", "Mentor", "Apprentice", "Family", "Spouse", "Other"],
                            value="Friend"
                        )
                        rel_desc = gr.Textbox(label="Connection Description / Context", placeholder="e.g. Served together in the sector squad")
                        
                        btn_add_rel = gr.Button("Add/Update Connection", variant="secondary")
                        rel_status_msg = gr.Markdown("")
                        
                        gr.Markdown("###### Active Connections")
                        rel_dataframe = gr.Dataframe(
                            headers=["ID", "Target Character", "Type", "Description"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False
                        )
                        btn_delete_rel = gr.Button("Remove Connection (Select ID row from table)", variant="stop")
                        selected_rel_row = gr.State(None)

                with gr.Row():
                    btn_save_char = gr.Button("Save Character profile", variant="primary")
                    btn_clear_char = gr.Button("Clear Editor Form", variant="secondary")
                char_msg = gr.Markdown("")

            # ==========================================
            # RIGHT PANEL: DIRECTORY, PORTFOLIO & CATEGORIZED GALLERY
            # ==========================================
            with gr.Column(scale=7, variant="panel"):
                gr.Markdown("#### 📇 Character Portfolio Directory")
                
                # Search & Filter Block
                with gr.Accordion("🔍 Directory Search & Dynamic Filtering Filters", open=True):
                    with gr.Row():
                        search_q = gr.Textbox(label="Search by Name/Bio/Traits", placeholder="e.g. cybernetic")
                        filter_gender = gr.Dropdown(label="Filter Gender", choices=["All"])
                    with gr.Row():
                        filter_tag = gr.Dropdown(label="Filter Tag", choices=["All"])
                        filter_project = gr.Dropdown(label="Filter Project Castings", choices=["All"])
                        filter_location = gr.Dropdown(label="Filter Location Castings", choices=["All"])
                    
                    btn_clear_filters = gr.Button("Reset Filters", variant="secondary")

                gr.Markdown("---")
                char_selector = gr.Dropdown(label="Select Character Profile", choices=[])
                
                # Detailed Display Card
                with gr.Group():
                    char_detail_name = gr.Markdown("### Select a Character profile from the directory list")
                    
                    with gr.Row():
                        char_detail_meta = gr.Markdown("")
                        char_detail_folder = gr.Markdown("")
                    
                    with gr.Tabs() as profile_display_tabs:
                        with gr.Tab("Biography & Personality"):
                            char_detail_bio = gr.Markdown("")
                        
                        with gr.Tab("Character DNA Attributes"):
                            char_detail_dna = gr.Markdown("")
                            
                        with gr.Tab("Connections (Relationships)"):
                            char_detail_rels = gr.Markdown("")

                        with gr.Tab("Asset Gallery"):
                            gr.Markdown("##### 📁 Categorized Asset Portfolio")
                            
                            with gr.Tabs() as gallery_tabs:
                                with gr.Tab("Reference Images"):
                                    gallery_ref = gr.Gallery(label="Reference Images", columns=4, object_fit="contain", height="auto")
                                with gr.Tab("Wardrobe"):
                                    gallery_wardrobe = gr.Gallery(label="Wardrobe Images", columns=4, object_fit="contain", height="auto")
                                with gr.Tab("Expressions"):
                                    gallery_expressions = gr.Gallery(label="Expression Images", columns=4, object_fit="contain", height="auto")
                                with gr.Tab("Approved Images"):
                                    gallery_approved_img = gr.Gallery(label="Approved Images", columns=4, object_fit="contain", height="auto")
                                with gr.Tab("Approved Videos"):
                                    video_selector = gr.Dropdown(label="Select Video to Play", choices=[])
                                    video_player = gr.Video(label="Approved Video Player", interactive=False)
                            
                            # Categorized uploader
                            gr.Markdown("---")
                            gr.Markdown("##### 📤 Add Category Assets")
                            with gr.Row():
                                upload_category = gr.Dropdown(
                                    label="Upload Destination Category", 
                                    choices=["Reference Images", "Wardrobe", "Expressions", "Approved Images", "Approved Videos"],
                                    value="Reference Images"
                                )
                                upload_files = gr.File(label="Choose Files to Add", file_count="multiple")
                            btn_upload_assets = gr.Button("Upload Assets to Character folder", variant="secondary")
                            upload_msg = gr.Markdown("")

                    with gr.Row():
                        btn_load_edit = gr.Button("Load into Editor Profile Builder", variant="secondary")
                        btn_delete_char = gr.Button("Delete Character from Studio", variant="stop")
                    
                    portfolio_msg = gr.Markdown("")

        # ==========================================
        # EVENT FUNCTIONS
        # ==========================================
        def setup_category_subfolders(folder_path):
            categories = ["reference", "wardrobe", "expressions", "approved_images", "approved_videos"]
            for cat in categories:
                os.makedirs(os.path.join(folder_path, cat), exist_ok=True)

        def get_category_images(folder_path, category):
            cat_dir = os.path.join(folder_path, category)
            if not os.path.exists(cat_dir):
                return []
            valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
            images = []
            for f in os.listdir(cat_dir):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    images.append(os.path.join(cat_dir, f))
            return images

        def get_category_videos(folder_path):
            cat_dir = os.path.join(folder_path, "approved_videos")
            if not os.path.exists(cat_dir):
                return []
            valid_exts = {".mp4", ".webm", ".mov", ".avi"}
            videos = []
            for f in os.listdir(cat_dir):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    videos.append(os.path.join(cat_dir, f))
            return videos

        # Refresh directory filters dynamically
        def get_filter_dropdown_updates():
            chars = db.list_characters()
            
            # 1. Genders list
            genders = sorted(list({c["gender"] for c in chars if c["gender"]}))
            gender_choices = ["All"] + genders
            
            # 2. Tags list
            all_tags = set()
            for c in chars:
                if c["tags"]:
                    tags = [t.strip() for t in c["tags"].split(",") if t.strip()]
                    all_tags.update(tags)
            tag_choices = ["All"] + sorted(list(all_tags))
            
            # 3. Projects list
            projs = db.list_projects()
            proj_choices = ["All"] + [p["name"] for p in projs]
            
            # 4. Locations list
            locs = db.list_locations()
            loc_choices = ["All"] + [l["name"] for l in locs]
            
            # 5. Connectable characters
            char_choices = [c["name"] for c in chars]
            
            return (
                gr.Dropdown(choices=gender_choices),
                gr.Dropdown(choices=tag_choices),
                gr.Dropdown(choices=proj_choices),
                gr.Dropdown(choices=loc_choices),
                gr.Dropdown(choices=char_choices)
            )

        # Dynamic search and filtering coordinator
        def filter_characters_fn(q, gender, tag, proj_name, loc_name):
            proj_id = None
            if proj_name and proj_name != "All":
                p = db.get_project_by_name(proj_name)
                if p:
                    proj_id = p["id"]
                    
            loc_id = None
            if loc_name and loc_name != "All":
                l = db.get_location_by_name(loc_name)
                if l:
                    loc_id = l["id"]
                    
            chars = db.search_characters(
                search_query=q.strip() if q else None,
                gender=gender if gender != "All" else None,
                tag=tag if tag != "All" else None,
                project_id=proj_id,
                location_id=loc_id
            )
            names = [c["name"] for c in chars]
            return gr.Dropdown(choices=names, value=names[0] if names else None)

        def reset_filters_fn():
            return (
                "", # q
                "All", # gender
                "All", # tag
                "All", # project
                "All" # location
            )

        # Compile templates
        def compile_prompt_fn(name, age, gender, hair, eyes, body, clothing, ethnicity, extra, template):
            try:
                # Fallback to defaults if empty
                sub_dict = {
                    "name": name or "Character",
                    "age": str(int(age)) if age else "unknown age",
                    "gender": gender or "character",
                    "dna_hair": hair or "natural hair",
                    "dna_eyes": eyes or "natural eyes",
                    "dna_body_type": body or "medium build",
                    "dna_clothing": clothing or "casual attire",
                    "dna_ethnicity": ethnicity or "global heritage",
                    "dna_description": extra or "standard appearance"
                }
                return template.format(**sub_dict)
            except Exception as e:
                return f"⚠️ Formatting Error in template: {e}\nEnsure placeholders match: {{name}}, {{age}}, {{gender}}, {{dna_hair}}, {{dna_eyes}}, {{dna_body_type}}, {{dna_clothing}}, {{dna_ethnicity}}, {{dna_description}}."

        # Add relationships
        def add_relationship_fn(edit_id, target_name, rel_type, desc):
            if edit_id is None:
                return gr.update(), "⚠️ Save the character profile first before mapping connections."
            if not target_name:
                return gr.update(), "⚠️ Please select a target character connect to."
            
            target = db.get_character_by_name(target_name)
            if not target:
                return gr.update(), "⚠️ Target character not found."
            
            if target["id"] == edit_id:
                return gr.update(), "⚠️ A character cannot have a relationship with themselves."
                
            db.add_relationship(edit_id, target["id"], rel_type, desc.strip())
            
            # Reload relationships table
            rels = db.list_relationships(edit_id)
            rel_rows = [[r["id"], r["target_character_name"], r["relationship_type"], r["description"] or ""] for r in rels]
            return rel_rows, f"✅ Relationship added/updated successfully with '{target_name}'."

        def select_rel_row_fn(evt: gr.SelectData):
            # Row index is evt.index[0]
            # ID is cell value if it's ID
            return evt.index

        def delete_relationship_fn(edit_id, row_index, df_data):
            if edit_id is None:
                return gr.update(), "⚠️ No active character context."
            if row_index is None or df_data is None:
                return gr.update(), "⚠️ Select a connection row from the table first."
            
            try:
                rel_id = int(df_data.iloc[row_index[0]][0])
                db.delete_relationship(rel_id)
                
                rels = db.list_relationships(edit_id)
                rel_rows = [[r["id"], r["target_character_name"], r["relationship_type"], r["description"] or ""] for r in rels]
                return rel_rows, "❌ Relationship connection removed."
            except Exception as e:
                return gr.update(), f"⚠️ Error deleting relationship: {e}"

        def save_character_fn(edit_id, name, age, gender, desc, notes, tags, bio, personality, voice, template,
                              hair, eyes, body, clothing, ethnicity, dna_desc):
            if not name or not name.strip():
                return gr.update(), "⚠️ Character name is required.", edit_id, gr.update()
            
            clean_name = name.strip()
            folder_name = clean_name.lower().replace(" ", "_")
            char_folder = os.path.join(CHARACTERS_DIR, folder_name)
            
            if edit_id is None:
                os.makedirs(char_folder, exist_ok=True)
                setup_category_subfolders(char_folder)
                char_id = db.add_character(
                    name=clean_name,
                    age=int(age) if age else None,
                    gender=gender.strip(),
                    description=desc.strip(),
                    notes=notes.strip(),
                    tags=tags.strip(),
                    folder_path=char_folder,
                    biography=bio.strip(),
                    personality=personality.strip(),
                    prompt_template=template.strip(),
                    wardrobe_notes="",
                    expression_notes="",
                    voice_notes=voice.strip(),
                    dna_hair=hair.strip(),
                    dna_eyes=eyes.strip(),
                    dna_body_type=body.strip(),
                    dna_clothing=clothing.strip(),
                    dna_ethnicity=ethnicity.strip(),
                    dna_description=dna_desc.strip()
                )
                if not char_id:
                    return gr.update(), "⚠️ A character with this name already exists.", edit_id, gr.update()
                msg = f"✅ Character '{clean_name}' profile created."
            else:
                char = db.get_character(edit_id)
                if not char:
                    return gr.update(), "⚠️ Character not found.", edit_id, gr.update()
                
                db.update_character(
                    char_id=edit_id,
                    name=clean_name,
                    age=int(age) if age else None,
                    gender=gender.strip(),
                    description=desc.strip(),
                    notes=notes.strip(),
                    tags=tags.strip(),
                    biography=bio.strip(),
                    personality=personality.strip(),
                    prompt_template=template.strip(),
                    wardrobe_notes="",
                    expression_notes="",
                    voice_notes=voice.strip(),
                    dna_hair=hair.strip(),
                    dna_eyes=eyes.strip(),
                    dna_body_type=body.strip(),
                    dna_clothing=clothing.strip(),
                    dna_ethnicity=ethnicity.strip(),
                    dna_description=dna_desc.strip()
                )
                char_id = edit_id
                old_folder = char["folder_path"]
                if old_folder != char_folder:
                    if os.path.exists(old_folder):
                        try:
                            if not os.path.exists(char_folder):
                                os.rename(old_folder, char_folder)
                            else:
                                for f in os.listdir(old_folder):
                                    shutil.move(os.path.join(old_folder, f), os.path.join(char_folder, f))
                                os.rmdir(old_folder)
                        except Exception as e:
                            print(f"Error moving character folder: {e}")
                            char_folder = old_folder
                    else:
                        os.makedirs(char_folder, exist_ok=True)
                        setup_category_subfolders(char_folder)
                    
                    with db._get_connection() as conn:
                        conn.execute("UPDATE characters SET folder_path = ? WHERE id = ?", (char_folder, char_id))
                        conn.commit()
                msg = f"✅ Character '{clean_name}' profile updated."

            # Refresh lists
            chars_list = db.list_characters()
            names = [c["name"] for c in chars_list]
            
            # Fetch filters list updates
            updates = get_filter_dropdown_updates()
            
            return (
                gr.Dropdown(choices=names, value=clean_name),
                msg,
                char_id, # Keep saved ID in context
                updates[4] # rel_target choices
            )

        def select_character_fn(name):
            if not name:
                return (
                    None, "### Select a Character profile", "", "", "", "", "", [], [], [], [], gr.update(choices=[]), None, []
                )
            
            char = db.get_character_by_name(name)
            if not char:
                return (
                    None, "### Selected character not found", "", "", "", "", "", [], [], [], [], gr.update(choices=[]), None, []
                )
            
            # Compile text details
            title_md = f"## 🎭 {char['name']}"
            meta_md = f"**Age:** {char['age'] or 'Unknown'}  |  **Gender:** {char['gender'] or 'N/A'}  |  **Tags:** `{char['tags'] or 'None'}`"
            folder_md = f"📂 **System Path:** `{char['folder_path']}`"
            
            bio_md = f"""
#### Biography / Backstory
{char['biography'] or '*No biography recorded.*'}

#### Personality & Behavioral Quirks
{char['personality'] or '*No personality traits recorded.*'}

#### Voice actor / Sound Notes
{char['voice_notes'] or '*No voice specifications noted.*'}
"""
            
            dna_md = f"""
| Attribute | DNA Specification Value |
| --- | --- |
| **Hair** | {char['dna_hair'] or '*Default*'} |
| **Eyes** | {char['dna_eyes'] or '*Default*'} |
| **Body Build** | {char['dna_body_type'] or '*Default*'} |
| **Wardrobe Style** | {char['dna_clothing'] or '*Default*'} |
| **Ethnicity** | {char['dna_ethnicity'] or '*Default*'} |
| **Scars / Interface / Details** | {char['dna_description'] or '*Default*'} |
"""
            
            # Format relationships markdown
            rels = db.list_relationships(char["id"])
            if rels:
                rels_md = "\n".join([f"- Connects to **{r['target_character_name']}** as **{r['relationship_type']}** ({r['description'] or 'no context'})" for r in rels])
            else:
                rels_md = "*No active character connections mapped yet.*"
            
            # Setup portfolio directories
            folder = char['folder_path']
            setup_category_subfolders(folder)
            
            # Fetch images/videos
            ref_imgs = get_category_images(folder, "reference")
            ward_imgs = get_category_images(folder, "wardrobe")
            exp_imgs = get_category_images(folder, "expressions")
            app_imgs = get_category_images(folder, "approved_images")
            
            videos = get_category_videos(folder)
            video_names = [os.path.basename(v) for v in videos]
            
            # Loaded relationships dataframe format
            rel_rows = [[r["id"], r["target_character_name"], r["relationship_type"], r["description"] or ""] for r in rels]
            
            return (
                char['id'],
                title_md,
                meta_md,
                folder_md,
                bio_md,
                dna_md,
                rels_md,
                ref_imgs,
                ward_imgs,
                exp_imgs,
                app_imgs,
                gr.update(choices=video_names, value=video_names[0] if video_names else None),
                None, # video player reset
                rel_rows
            )

        def play_selected_video_fn(char_id, video_filename):
            if char_id is None or not video_filename:
                return None
            char = db.get_character(char_id)
            if not char:
                return None
            video_path = os.path.join(char["folder_path"], "approved_videos", video_filename)
            if os.path.exists(video_path):
                return video_path
            return None

        def load_edit_form(edit_id):
            if edit_id is None:
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Select a profile directory character first."
                )
            
            char = db.get_character(edit_id)
            if not char:
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Selected character not found."
                )
            
            return (
                char["name"],
                char["age"] or 30,
                char["gender"] or "",
                char["biography"] or "",
                char["personality"] or "",
                char["voice_notes"] or "",
                char["tags"] or "",
                char["description"] or "",
                char["notes"] or "",
                char["dna_hair"] or "",
                char["dna_eyes"] or "",
                char["dna_body_type"] or "",
                char["dna_clothing"] or "",
                char["dna_ethnicity"] or "",
                char["dna_description"] or "",
                char["prompt_template"] or "",
                f"📝 Loaded '{char['name']}' specifications into character builder."
            )

        def clear_form_fn():
            return (
                None, "", 30, "", "", "", "", "", "", "", "", "", "", "", "", "",
                "A premium portrait of {name}, {age} years old, {gender}, {dna_ethnicity}, featuring {dna_hair}, {dna_eyes}, {dna_body}, wearing {dna_clothing}, {dna_description}. Cinematic lighting, 35mm photograph, high fidelity.",
                "🧹 Form cleared."
            )

        def upload_assets_fn(char_id, upload_cat, files):
            if char_id is None:
                return [], [], [], [], gr.update(choices=[]), None, "⚠️ Select a character profile first."
            
            char = db.get_character(char_id)
            if not char:
                return [], [], [], [], gr.update(choices=[]), None, "⚠️ Selected character not found."
            
            folder = char["folder_path"]
            setup_category_subfolders(folder)
            
            # Map category name to subfolder name
            cat_map = {
                "Reference Images": "reference",
                "Wardrobe": "wardrobe",
                "Expressions": "expressions",
                "Approved Images": "approved_images",
                "Approved Videos": "approved_videos"
            }
            subfolder = cat_map.get(upload_cat, "reference")
            dest_dir = os.path.join(folder, subfolder)
            
            if files:
                for f in files:
                    dest = os.path.join(dest_dir, os.path.basename(f.name))
                    try:
                        shutil.copy(f.name, dest)
                    except Exception as e:
                        print(f"Error copying asset: {e}")
            
            # Reload all assets
            ref_imgs = get_category_images(folder, "reference")
            ward_imgs = get_category_images(folder, "wardrobe")
            exp_imgs = get_category_images(folder, "expressions")
            app_imgs = get_category_images(folder, "approved_images")
            
            videos = get_category_videos(folder)
            video_names = [os.path.basename(v) for v in videos]
            
            return (
                ref_imgs,
                ward_imgs,
                exp_imgs,
                app_imgs,
                gr.update(choices=video_names, value=video_names[0] if video_names else None),
                gr.update(value=None), # Reset files input
                f"✅ Successfully uploaded {len(files) if files else 0} file(s) into {upload_cat}."
            )

        def delete_character_fn(edit_id, name):
            target_id = edit_id
            if target_id is None and name:
                char = db.get_character_by_name(name)
                if char:
                    target_id = char["id"]
                    
            if target_id is None:
                return gr.update(), "⚠️ Select a character profile to delete.", None
            
            char = db.get_character(target_id)
            if char:
                db.delete_character(target_id)
                if os.path.exists(char["folder_path"]):
                    try:
                        shutil.rmtree(char["folder_path"])
                    except Exception as e:
                        print(f"Error deleting folder: {e}")
            
            chars_list = db.list_characters()
            names = [c["name"] for c in chars_list]
            
            return gr.Dropdown(choices=names, value=names[0] if names else None), f"❌ Profile for '{char['name'] if char else 'Unknown'}' deleted.", None

        # ==========================================
        # WIRE UP LISTENERS
        # ==========================================
        
        # Directory Selection
        char_selector.change(
            fn=select_character_fn,
            inputs=[char_selector],
            outputs=[
                char_edit_id, char_detail_name, char_detail_meta, char_detail_folder,
                char_detail_bio, char_detail_dna, char_detail_rels,
                gallery_ref, gallery_wardrobe, gallery_expressions, gallery_approved_img,
                video_selector, video_player, rel_dataframe
            ]
        )
        
        # Play Video selection change
        video_selector.change(
            fn=play_selected_video_fn,
            inputs=[char_edit_id, video_selector],
            outputs=[video_player]
        )
        
        # Save Profile
        btn_save_char.click(
            fn=save_character_fn,
            inputs=[
                char_edit_id, char_name, char_age, char_gender, char_desc, char_notes, char_tags,
                char_biography, char_personality, char_voice, char_prompt_template,
                dna_hair, dna_eyes, dna_body, dna_clothing, dna_ethnicity, dna_description
            ],
            outputs=[char_selector, char_msg, char_edit_id, rel_target]
        )
        
        # Compile/Preview prompt
        btn_compile_prompt.click(
            fn=compile_prompt_fn,
            inputs=[char_name, char_age, char_gender, dna_hair, dna_eyes, dna_body, dna_clothing, dna_ethnicity, dna_description, char_prompt_template],
            outputs=[compiled_prompt_preview]
        )
        
        # Relationships Dataframe events
        rel_dataframe.select(fn=select_rel_row_fn, outputs=[selected_rel_row])
        
        btn_add_rel.click(
            fn=add_relationship_fn,
            inputs=[char_edit_id, rel_target, rel_type, rel_desc],
            outputs=[rel_dataframe, rel_status_msg]
        )
        
        btn_delete_rel.click(
            fn=delete_relationship_fn,
            inputs=[char_edit_id, selected_rel_row, rel_dataframe],
            outputs=[rel_dataframe, rel_status_msg]
        )
        
        # Load profile into editor
        btn_load_edit.click(
            fn=load_edit_form,
            inputs=[char_edit_id],
            outputs=[
                char_name, char_age, char_gender, char_biography, char_personality, char_voice, char_tags, char_desc, char_notes,
                dna_hair, dna_eyes, dna_body, dna_clothing, dna_ethnicity, dna_description, char_prompt_template, char_msg
            ]
        )
        
        # Clear editor form
        btn_clear_char.click(
            fn=clear_form_fn,
            outputs=[
                char_edit_id, char_name, char_age, char_gender, char_biography, char_personality, char_voice, char_tags, char_desc, char_notes,
                dna_hair, dna_eyes, dna_body, dna_clothing, dna_ethnicity, dna_description, char_prompt_template, char_msg
            ]
        )
        
        # Delete profile
        btn_delete_char.click(
            fn=delete_character_fn,
            inputs=[char_edit_id, char_selector],
            outputs=[char_selector, portfolio_msg, char_edit_id]
        )
        
        # Upload categorized assets
        btn_upload_assets.click(
            fn=upload_assets_fn,
            inputs=[char_edit_id, upload_category, upload_files],
            outputs=[gallery_ref, gallery_wardrobe, gallery_expressions, gallery_approved_img, video_selector, upload_files, upload_msg]
        )
        
        # Dynamic search and filtering on change
        inputs_filter = [search_q, filter_gender, filter_tag, filter_project, filter_location]
        for inp in inputs_filter:
            inp.change(fn=filter_characters_fn, inputs=inputs_filter, outputs=[char_selector])
            
        btn_clear_filters.click(fn=reset_filters_fn, outputs=inputs_filter)
        
        # Return components for orchestration
        return char_selector, filter_gender, filter_tag, filter_project, filter_location, rel_target
