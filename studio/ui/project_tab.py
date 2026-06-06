import os
import shutil
import gradio as gr
from studio.database.db_manager import DatabaseManager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")

def create_project_tab(db: DatabaseManager):
    with gr.TabItem("Projects"):
        with gr.Row():
            # Column 1: Projects (Series) Management
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 📂 Project Series Builder")
                proj_name = gr.Textbox(label="Project/Series Name (Unique)", placeholder="e.g. Tokyo Midnight")
                proj_desc = gr.Textbox(label="Series Logline / Summary", placeholder="Enter description of the series plot...", lines=3)
                btn_save_proj = gr.Button("Create Series Project", variant="primary")
                
                gr.Markdown("---")
                gr.Markdown("### 🗑️ Delete Project Series")
                proj_delete_selector = gr.Dropdown(label="Select Project to Delete", choices=[])
                btn_delete_proj = gr.Button("Delete Project", variant="stop")
                proj_msg = gr.Markdown("")

            # Column 2: Episode Management
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🎞️ Episode Manager")
                active_proj_dropdown = gr.Dropdown(label="Select Project Context", choices=[])
                
                ep_number = gr.Number(label="Episode Number", value=1, precision=0)
                ep_title = gr.Textbox(label="Episode Title", placeholder="e.g. The Rainy Window")
                ep_desc = gr.Textbox(label="Episode Synopsis", placeholder="Summary of what happens in this episode...", lines=3)
                btn_save_ep = gr.Button("Create Episode", variant="primary")
                
                gr.Markdown("---")
                gr.Markdown("### 🗑️ Delete Episode")
                ep_delete_selector = gr.Dropdown(label="Select Episode to Delete", choices=[])
                btn_delete_ep = gr.Button("Delete Episode", variant="stop")
                ep_msg = gr.Markdown("")

        # ==========================================
        # EVENT FUNCTIONS
        # ==========================================
        def load_projects_list():
            projs = db.list_projects()
            names = [p["name"] for p in projs]
            return gr.Dropdown(choices=names), gr.Dropdown(choices=names)

        def load_episodes_list(proj_name):
            if not proj_name:
                return gr.Dropdown(choices=[], value=None)
            proj = db.get_project_by_name(proj_name)
            if not proj:
                return gr.Dropdown(choices=[], value=None)
            eps = db.list_episodes(proj["id"])
            ep_strs = [f"Ep {e['episode_number']}: {e['title'] or 'Untitled'}" for e in eps]
            return gr.Dropdown(choices=ep_strs, value=ep_strs[0] if ep_strs else None)

        def create_project_fn(name, desc):
            if not name or not name.strip():
                return "⚠️ Project name is required.", gr.update(), gr.update()
            
            clean_name = name.strip()
            folder_name = clean_name.replace(" ", "_")
            proj_folder = os.path.join(PROJECTS_DIR, folder_name)
            
            proj_id = db.add_project(clean_name, desc.strip())
            if not proj_id:
                return "⚠️ Project name must be unique.", gr.update(), gr.update()
                
            os.makedirs(proj_folder, exist_ok=True)
            
            proj_choices, proj_del_choices = load_projects_list()
            return f"✅ Project series '{clean_name}' created successfully.", proj_choices, proj_del_choices

        def delete_project_fn(name):
            if not name:
                return "⚠️ No project selected.", gr.update(), gr.update()
            
            proj = db.get_project_by_name(name)
            if proj:
                db.delete_project(proj["id"])
                
                # Delete folder on disk
                folder_name = name.replace(" ", "_")
                proj_folder = os.path.join(PROJECTS_DIR, folder_name)
                if os.path.exists(proj_folder):
                    try:
                        shutil.rmtree(proj_folder)
                    except Exception as e:
                        print(f"Error removing project folder: {e}")
                        
            proj_choices, proj_del_choices = load_projects_list()
            return f"❌ Project series '{name}' and its directories deleted.", proj_choices, proj_del_choices

        def create_episode_fn(proj_name, num, title, desc):
            if not proj_name:
                return "⚠️ Please select a project context first.", gr.update()
                
            proj = db.get_project_by_name(proj_name)
            if not proj:
                return "⚠️ Selected project context invalid.", gr.update()
                
            ep_id = db.add_episode(proj["id"], int(num), title.strip(), desc.strip())
            
            # Create episode subfolder
            folder_name = proj_name.replace(" ", "_")
            ep_folder = os.path.join(PROJECTS_DIR, folder_name, f"ep_{int(num)}")
            os.makedirs(ep_folder, exist_ok=True)
            
            ep_choices = load_episodes_list(proj_name)
            return f"✅ Episode {int(num)} created successfully.", ep_choices

        def delete_episode_fn(proj_name, ep_str):
            if not proj_name or not ep_str:
                return "⚠️ No context selected to delete.", gr.update()
                
            proj = db.get_project_by_name(proj_name)
            try:
                ep_num = int(ep_str.split(":")[0].replace("Ep ", "").strip())
            except ValueError:
                return "⚠️ Selection format error.", gr.update()
                
            ep = db.get_episode_by_details(proj["id"], ep_num)
            if ep:
                db.delete_episode(ep["id"])
                
                # Delete episode folder on disk
                folder_name = proj_name.replace(" ", "_")
                ep_folder = os.path.join(PROJECTS_DIR, folder_name, f"ep_{ep_num}")
                if os.path.exists(ep_folder):
                    try:
                        shutil.rmtree(ep_folder)
                    except Exception as e:
                        print(f"Error removing episode folder: {e}")
                        
            ep_choices = load_episodes_list(proj_name)
            return f"❌ Episode {ep_num} deleted successfully.", ep_choices

        # Wire up listeners
        btn_save_proj.click(fn=create_project_fn, inputs=[proj_name, proj_desc], outputs=[proj_msg, active_proj_dropdown, proj_delete_selector])
        btn_delete_proj.click(fn=delete_project_fn, inputs=[proj_delete_selector], outputs=[proj_msg, active_proj_dropdown, proj_delete_selector])
        
        active_proj_dropdown.change(fn=load_episodes_list, inputs=[active_proj_dropdown], outputs=[ep_delete_selector])
        
        btn_save_ep.click(fn=create_episode_fn, inputs=[active_proj_dropdown, ep_number, ep_title, ep_desc], outputs=[ep_msg, ep_delete_selector])
        btn_delete_ep.click(fn=delete_episode_fn, inputs=[active_proj_dropdown, ep_delete_selector], outputs=[ep_msg, ep_delete_selector])

        return active_proj_dropdown, proj_delete_selector, ep_delete_selector
