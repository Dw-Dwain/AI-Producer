import os
import gradio as gr
from studio.database.db_manager import DatabaseManager

def create_story_bible_tab(db: DatabaseManager):
    with gr.TabItem("Story Bible"):
        gr.Markdown("### 📖 Story Bible & Narrative Engine (Phase 3)")
        
        with gr.Row():
            with gr.Column(scale=12):
                sb_project_selector = gr.Dropdown(label="Select Active Project Story Bible", choices=[])
                gr.Markdown("---")
        
        with gr.Tabs():
            # ==========================================
            # TAB 1: TIMELINE EVENTS
            # ==========================================
            with gr.Tab("Chronological Timeline"):
                with gr.Row():
                    # Left pane: Event editor
                    with gr.Column(scale=4, variant="panel"):
                        gr.Markdown("#### 📝 Edit Timeline Event")
                        event_edit_id = gr.State(None)
                        
                        event_order = gr.Number(label="Event Chronological Order / Rank", value=1, precision=0)
                        event_title = gr.Textbox(label="Event Title", placeholder="e.g. The Neon Heist")
                        event_date = gr.Textbox(label="Event Date / Time Index", placeholder="e.g. Day 3, 22:00 / Year 2088")
                        event_desc = gr.Textbox(label="Event Description / Narrative", placeholder="Write detailed scene backstory or plot details...", lines=5)
                        
                        with gr.Row():
                            btn_save_event = gr.Button("Save Event", variant="primary")
                            btn_clear_event = gr.Button("Clear Form", variant="secondary")
                            btn_delete_event = gr.Button("Delete Event", variant="stop")
                        
                        event_msg = gr.Markdown("")
                        
                    # Right pane: Chronological list
                    with gr.Column(scale=8, variant="panel"):
                        gr.Markdown("#### ⏳ Project Timeline Index")
                        gr.Markdown("*Select any row in the table below to load it into the editor.*")
                        
                        timeline_dataframe = gr.Dataframe(
                            headers=["ID", "Order", "Title", "Date / Time", "Description"],
                            datatype=["str", "number", "str", "str", "str"],
                            interactive=False
                        )
                        selected_timeline_row = gr.State(None)
            
            # ==========================================
            # TAB 2: PLOT & LORE WIKI
            # ==========================================
            with gr.Tab("Plot & Lore Notes"):
                with gr.Row():
                    # Left pane: Notes manager & editor
                    with gr.Column(scale=5, variant="panel"):
                        gr.Markdown("#### 🖋️ Lore & Plot Editor")
                        note_edit_id = gr.State(None)
                        
                        with gr.Row():
                            note_category_filter = gr.Dropdown(
                                label="Filter Notes Category", 
                                choices=["All", "Lore", "Worldbuilding", "Tone Guide", "Plot Notes"],
                                value="All"
                            )
                            note_selector = gr.Dropdown(label="Select Note to View/Edit", choices=[])
                        
                        gr.Markdown("---")
                        note_title = gr.Textbox(label="Note Title", placeholder="e.g. Sector 7 Faction Details")
                        note_category = gr.Dropdown(
                            label="Note Category", 
                            choices=["Lore", "Worldbuilding", "Tone Guide", "Plot Notes"],
                            value="Lore"
                        )
                        note_content = gr.Textbox(label="Content (Markdown supported)", placeholder="Write lore bible entries, rules of the world, tone specifications...", lines=8)
                        
                        with gr.Row():
                            btn_save_note = gr.Button("Save Note", variant="primary")
                            btn_clear_note = gr.Button("Clear Form", variant="secondary")
                            btn_delete_note = gr.Button("Delete Note", variant="stop")
                            
                        note_msg = gr.Markdown("")
                        
                    # Right pane: Read-only Markdown Viewer
                    with gr.Column(scale=7, variant="panel"):
                        gr.Markdown("#### 📖 Lore & World Wiki Reader")
                        wiki_viewer = gr.HTML("<div style='padding: 20px; border-radius: 8px; background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); min-height: 300px;'>Select a lore note from the index list to start reading.</div>")
            
            # ==========================================
            # TAB 3: CAST & RELATIONSHIPS
            # ==========================================
            with gr.Tab("Casted Cast & Relations"):
                with gr.Row():
                    with gr.Column(scale=6, variant="panel"):
                        gr.Markdown("#### 👥 Cast Members in this Project")
                        gr.Markdown("*Characters cast in scenes for the selected project.*")
                        
                        cast_dataframe = gr.Dataframe(
                            headers=["Name", "Gender", "Age", "Tags", "System Path"],
                            datatype=["str", "str", "number", "str", "str"],
                            interactive=False
                        )
                        
                    with gr.Column(scale=6, variant="panel"):
                        gr.Markdown("#### 🔗 Connection Network Map")
                        gr.Markdown("*Active relationships between cast members in this project context.*")
                        
                        relations_dataframe = gr.Dataframe(
                            headers=["Source Character", "Connection Type", "Target Character", "Context"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False
                        )
                        
                        gr.Markdown("##### 🕸️ Textual Connections Graph")
                        connections_network_md = gr.Markdown("*No relationships mapped between active project characters.*")

        # ==========================================
        # EVENT FUNCTIONS
        # ==========================================
        
        # Casting & Relationships network queries
        def get_casted_characters(project_id):
            if not project_id:
                return []
            with db._get_connection() as conn:
                rows = conn.execute(
                    """SELECT DISTINCT c.* 
                       FROM characters c
                       JOIN scenes s ON s.character_id = c.id
                       JOIN episodes e ON s.episode_id = e.id
                       WHERE e.project_id = ?
                       ORDER BY c.name ASC""",
                    (project_id,)
                ).fetchall()
                return [dict(row) for row in rows]

        def get_project_relationships(project_id):
            if not project_id:
                return []
            with db._get_connection() as conn:
                rows = conn.execute(
                    """SELECT r.*, c1.name as source_character_name, c2.name as target_character_name
                       FROM character_relationships r
                       JOIN characters c1 ON r.character_id = c1.id
                       JOIN characters c2 ON r.target_character_id = c2.id
                       WHERE r.character_id IN (
                           SELECT DISTINCT s.character_id 
                           FROM scenes s 
                           JOIN episodes e ON s.episode_id = e.id 
                           WHERE e.project_id = ? AND s.character_id IS NOT NULL
                       )
                       AND r.target_character_id IN (
                           SELECT DISTINCT s.character_id 
                           FROM scenes s 
                           JOIN episodes e ON s.episode_id = e.id 
                           WHERE e.project_id = ? AND s.character_id IS NOT NULL
                       )""",
                    (project_id, project_id)
                ).fetchall()
                return [dict(row) for row in rows]

        def load_casting_and_relations_fn(project_name):
            if not project_name:
                return [], [], "*Select a project context to view the casting network.*"
                
            proj = db.get_project_by_name(project_name)
            if not proj:
                return [], [], "*Selected project not found.*"
                
            proj_id = proj["id"]
            
            # Cast members list
            chars = get_casted_characters(proj_id)
            char_rows = [[c["name"], c["gender"] or "N/A", c["age"] or 0, c["tags"] or "", c["folder_path"]] for c in chars]
            
            # Relationships list
            rels = get_project_relationships(proj_id)
            rel_rows = [[r["source_character_name"], r["relationship_type"], r["target_character_name"], r["description"] or ""] for r in rels]
            
            # Build network markdown graph
            if rels:
                graph_lines = ["##### Active Connections Map:"]
                for r in rels:
                    graph_lines.append(f"- **{r['source_character_name']}** ➔ `{r['relationship_type']}` ➔ **{r['target_character_name']}** *(Context: {r['description'] or 'None'})*")
                graph_md = "\n".join(graph_lines)
            else:
                graph_md = "*No active character relationships mapped between the casted actors for this project.*"
                
            return char_rows, rel_rows, graph_md

        # Timeline logic
        def load_timeline_list_fn(project_name):
            if not project_name:
                return []
            proj = db.get_project_by_name(project_name)
            if not proj:
                return []
            events = db.list_timeline_events(proj["id"])
            return [[e["id"], e["event_order"], e["title"], e["event_date"] or "", e["description"] or ""] for e in events]

        def save_timeline_event_fn(project_name, event_id, order, title, date, desc):
            if not project_name:
                return gr.update(), "⚠️ Please select a project context first.", None
            if not title or not title.strip():
                return gr.update(), "⚠️ Event title is required.", event_id
                
            proj = db.get_project_by_name(project_name)
            if not proj:
                return gr.update(), "⚠️ Project not found.", event_id
                
            saved_id = db.add_timeline_event(
                project_id=proj["id"],
                event_order=int(order),
                title=title.strip(),
                description=desc.strip(),
                event_date=date.strip()
            )
            
            # Reload
            events = load_timeline_list_fn(project_name)
            return events, "✅ Event saved successfully.", saved_id

        def delete_timeline_event_fn(project_name, event_id):
            if not project_name or event_id is None:
                return gr.update(), "⚠️ Select an event to delete first.", event_id
                
            db.delete_timeline_event(event_id)
            events = load_timeline_list_fn(project_name)
            return events, "❌ Timeline event deleted.", None

        def select_timeline_row_fn(evt: gr.SelectData, df_data):
            if df_data is None or evt.index is None:
                return None, 1, "", "", ""
            try:
                row_idx = evt.index[0]
                event_id = int(df_data.iloc[row_idx][0])
                with db._get_connection() as conn:
                    row = conn.execute("SELECT * FROM timeline_events WHERE id = ?", (event_id,)).fetchone()
                if row:
                    row = dict(row)
                    return row["id"], row["event_order"], row["title"], row["event_date"] or "", row["description"] or ""
            except Exception as e:
                print(f"Error selecting timeline event: {e}")
            return None, 1, "", "", ""

        def clear_timeline_form_fn():
            return None, 1, "", "", "", "🧹 Timeline editor cleared."

        # Notes/Wiki logic
        def load_note_selector_choices(project_name, category_filter):
            if not project_name:
                return gr.update(choices=[], value=None)
                
            proj = db.get_project_by_name(project_name)
            if not proj:
                return gr.update(choices=[], value=None)
                
            cat = None if category_filter == "All" else category_filter
            notes = db.list_story_notes(proj["id"], cat)
            titles = [n["title"] for n in notes]
            return gr.update(choices=titles, value=titles[0] if titles else None)

        def view_selected_note_fn(project_name, note_title):
            if not project_name or not note_title:
                return (
                    None, "", "Lore", "", 
                    "<div style='padding: 20px; border-radius: 8px; background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1);'>Select a lore note to read.</div>"
                )
                
            proj = db.get_project_by_name(project_name)
            if not proj:
                return None, "", "Lore", "", "⚠️ Project not found."
                
            with db._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM story_bible_notes WHERE project_id = ? AND title = ?", 
                    (proj["id"], note_title)
                ).fetchone()
                
            if row:
                row = dict(row)
                
                # Format content as HTML using stdlib only (no external markdown dep)
                import html as _html
                import re as _re
                raw = row["content"] or "*No content.*"
                escaped = _html.escape(raw)
                # Bold: **text**
                escaped = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
                # Italic: *text*
                escaped = _re.sub(r'\*(.+?)\*', r'<em>\1</em>', escaped)
                # Headings: ## heading
                escaped = _re.sub(r'^## (.+)$', r'<h2>\1</h2>', escaped, flags=_re.MULTILINE)
                escaped = _re.sub(r'^# (.+)$', r'<h1>\1</h1>', escaped, flags=_re.MULTILINE)
                # Newlines to <br>
                content_html = escaped.replace('\n', '<br>')
                
                html_body = f"""
                <div style='padding: 25px; border-radius: 12px; background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%); border: 1px solid rgba(255,255,255,0.08); shadow: 0 4px 30px rgba(0, 0, 0, 0.5);'>
                    <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 20px;'>
                        <h2 style='margin: 0; color: #FFF; font-family: "Outfit", sans-serif; font-size: 24px; font-weight: 600;'>{row['title']}</h2>
                        <span style='background: rgba(147, 51, 234, 0.2); border: 1px solid rgb(147, 51, 234); color: #c084fc; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-family: sans-serif; font-weight: 500;'>{row['category']}</span>
                    </div>
                    <div class='markdown-body' style='color: #E2E8F0; line-height: 1.6; font-size: 15px;'>
                        {content_html}
                    </div>
                    <div style='margin-top: 30px; font-size: 11px; color: #718096; font-style: italic; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px;'>
                        Created on {row['created_at']}
                    </div>
                </div>
                """
                return row["id"], row["title"], row["category"], row["content"] or "", html_body
                
            return None, "", "Lore", "", "⚠️ Note not found."

        def save_story_note_fn(project_name, note_id, title, category, content):
            if not project_name:
                return gr.update(), "⚠️ Please select a project context first.", note_id
            if not title or not title.strip():
                return gr.update(), "⚠️ Note title is required.", note_id
                
            proj = db.get_project_by_name(project_name)
            if not proj:
                return gr.update(), "⚠️ Project not found.", note_id
                
            if note_id is None:
                # Add
                saved_id = db.add_story_note(
                    project_id=proj["id"],
                    title=title.strip(),
                    content=content.strip(),
                    category=category
                )
                msg = f"✅ Note '{title.strip()}' created."
            else:
                # Update
                db.update_story_note(
                    note_id=note_id,
                    title=title.strip(),
                    content=content.strip(),
                    category=category
                )
                saved_id = note_id
                msg = f"✅ Note '{title.strip()}' updated."
                
            # Reload selector Choices
            notes = db.list_story_notes(proj["id"])
            titles = [n["title"] for n in notes]
            return gr.update(choices=titles, value=title.strip()), msg, saved_id

        def delete_story_note_fn(project_name, note_id):
            if not project_name or note_id is None:
                return gr.update(), "⚠️ Select a note to delete first.", note_id
                
            db.delete_story_note(note_id)
            
            proj = db.get_project_by_name(project_name)
            notes = db.list_story_notes(proj["id"])
            titles = [n["title"] for n in notes]
            return gr.update(choices=titles, value=titles[0] if titles else None), "❌ Note deleted.", None

        def clear_note_form_fn():
            return None, "", "Lore", "", gr.update(), "🧹 Form cleared."

        # Global project choice change event
        def project_change_coordinator(project_name, category_filter):
            timeline = load_timeline_list_fn(project_name)
            notes_dropdown = load_note_selector_choices(project_name, category_filter)
            cast_rows, rel_rows, graph_md = load_casting_and_relations_fn(project_name)
            
            # Reset editors
            t_form = (None, 1, "", "", "", "")
            n_form = (None, "", "Lore", "", "<div style='padding: 20px; border-radius: 8px; background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1);'>Select a lore note to read.</div>")
            
            return (
                timeline,
                notes_dropdown,
                cast_rows,
                rel_rows,
                graph_md,
                # Timeline inputs reset
                t_form[0], t_form[1], t_form[2], t_form[3], t_form[4], t_form[5],
                # Notes inputs reset
                n_form[0], n_form[1], n_form[2], n_form[3], n_form[4]
            )

        # Wire up listeners
        sb_project_selector.change(
            fn=project_change_coordinator,
            inputs=[sb_project_selector, note_category_filter],
            outputs=[
                timeline_dataframe, note_selector, cast_dataframe, relations_dataframe, connections_network_md,
                # Timeline form
                event_edit_id, event_order, event_title, event_date, event_desc, event_msg,
                # Notes form
                note_edit_id, note_title, note_category, note_content, wiki_viewer
            ]
        )
        
        note_category_filter.change(
            fn=load_note_selector_choices,
            inputs=[sb_project_selector, note_category_filter],
            outputs=[note_selector]
        )
        
        note_selector.change(
            fn=view_selected_note_fn,
            inputs=[sb_project_selector, note_selector],
            outputs=[note_edit_id, note_title, note_category, note_content, wiki_viewer]
        )
        
        # Timeline actions
        btn_save_event.click(
            fn=save_timeline_event_fn,
            inputs=[sb_project_selector, event_edit_id, event_order, event_title, event_date, event_desc],
            outputs=[timeline_dataframe, event_msg, event_edit_id]
        )
        
        btn_delete_event.click(
            fn=delete_timeline_event_fn,
            inputs=[sb_project_selector, event_edit_id],
            outputs=[timeline_dataframe, event_msg, event_edit_id]
        )
        
        btn_clear_event.click(
            fn=clear_timeline_form_fn,
            outputs=[event_edit_id, event_order, event_title, event_date, event_desc, event_msg]
        )
        
        timeline_dataframe.select(
            fn=select_timeline_row_fn,
            inputs=[timeline_dataframe],
            outputs=[event_edit_id, event_order, event_title, event_date, event_desc]
        )
        
        # Notes actions
        btn_save_note.click(
            fn=save_story_note_fn,
            inputs=[sb_project_selector, note_edit_id, note_title, note_category, note_content],
            outputs=[note_selector, note_msg, note_edit_id]
        )
        
        btn_delete_note.click(
            fn=delete_story_note_fn,
            inputs=[sb_project_selector, note_edit_id],
            outputs=[note_selector, note_msg, note_edit_id]
        )
        
        btn_clear_note.click(
            fn=clear_note_form_fn,
            outputs=[note_edit_id, note_title, note_category, note_content, note_selector, note_msg]
        )

        return sb_project_selector,
