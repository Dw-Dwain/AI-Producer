import gradio as gr
from studio.database.db_manager import DatabaseManager
from studio.intelligence.script_analyzer import ScriptAnalyzer


def create_scene_tab(db: DatabaseManager):
    with gr.TabItem("Scenes"):
        active_scene_id = gr.State(None)
        gr.Markdown("### Scene Builder")

        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                proj_selector = gr.Dropdown(label="Project", choices=[])
                ep_selector = gr.Dropdown(label="Episode", choices=[])
                scene_selector = gr.Dropdown(label="Scene", choices=[])
                btn_new_scene = gr.Button("New Scene")
                btn_delete_scene = gr.Button("Delete Scene", variant="stop")
                scene_list_msg = gr.Markdown("")

            with gr.Column(scale=2, variant="panel"):
                scene_num = gr.Number(label="Scene Number", value=1, precision=0)
                scene_title = gr.Textbox(label="Scene Title")
                scene_desc = gr.Textbox(label="Description", lines=3)
                with gr.Row():
                    casting_char = gr.Dropdown(label="Default Character", choices=[])
                    casting_loc = gr.Dropdown(label="Default Location", choices=[])
                continuity_weather = gr.Textbox(label="Continuity Weather")
                continuity_time = gr.Textbox(label="Continuity Time Of Day")
                continuity_wardrobe = gr.Textbox(label="Continuity Wardrobe")
                scene_script = gr.Textbox(label="Scene Script", lines=6)
                btn_save_scene = gr.Button("Save Scene", variant="primary")
                btn_build_scene = gr.Button("Generate Shot List From Script", variant="secondary")
                scene_save_msg = gr.Markdown("")

                shot_table = gr.Dataframe(
                    headers=["Shot #", "Type", "Character", "Goal", "Prompt"],
                    datatype=["number", "str", "str", "str", "str"],
                    interactive=False,
                )

        def load_episodes_for_proj(proj_name):
            if not proj_name:
                return gr.update(choices=[], value=None)
            project = db.get_project_by_name(proj_name)
            episodes = db.list_episodes(project["id"]) if project else []
            labels = [f"Ep {ep['episode_number']}: {ep['title'] or 'Untitled'}" for ep in episodes]
            return gr.update(choices=labels, value=labels[0] if labels else None)

        def load_scenes_for_ep(proj_name, ep_label):
            if not proj_name or not ep_label:
                return gr.update(choices=[], value=None)
            project = db.get_project_by_name(proj_name)
            ep_num = int(ep_label.split(":")[0].replace("Ep ", "").strip())
            episode = db.get_episode_by_details(project["id"], ep_num) if project else None
            scenes = db.list_scenes(episode["id"]) if episode else []
            labels = [f"Scene {scene['scene_number']}: {scene['title'] or 'Untitled'}" for scene in scenes]
            return gr.update(choices=labels, value=labels[0] if labels else None)

        def load_casting_choices():
            chars = ["None"] + [char["name"] for char in db.list_characters()]
            locs = ["None"] + [loc["name"] for loc in db.list_locations()]
            return gr.update(choices=chars), gr.update(choices=locs)

        def select_scene_fn(proj_name, ep_label, sc_label):
            if not proj_name or not ep_label or not sc_label:
                return None, 1, "", "", gr.update(value="None"), gr.update(value="None"), "", "", "", "", []
            project = db.get_project_by_name(proj_name)
            ep_num = int(ep_label.split(":")[0].replace("Ep ", "").strip())
            sc_num = int(sc_label.split(":")[0].replace("Scene ", "").strip())
            episode = db.get_episode_by_details(project["id"], ep_num) if project else None
            scene = next((row for row in db.list_scenes(episode["id"]) if row["scene_number"] == sc_num), None) if episode else None
            continuity = db.get_continuity_state(scene["id"]) if scene else {}
            shots = db.list_shots(scene["id"]) if scene else []
            table = [[s["shot_number"], s.get("shot_type", ""), s.get("character_name", ""), s.get("goal", ""), s.get("prompt", "")] for s in shots]
            dialogue = "\n".join(f"{line['character_name']}: {line['text']}" for line in db.list_dialogue_lines(scene["id"])) if scene else ""
            return (
                scene["id"] if scene else None,
                scene["scene_number"] if scene else 1,
                scene.get("title", "") if scene else "",
                scene.get("description", "") if scene else "",
                gr.update(value=scene.get("character_name") or "None"),
                gr.update(value=scene.get("location_name") or "None"),
                continuity.get("weather", "") if continuity else "",
                continuity.get("time_of_day", "") if continuity else "",
                continuity.get("wardrobe_tag", "") if continuity else "",
                dialogue,
                table,
            )

        def save_scene_fn(scene_id, proj_name, ep_label, number, title, desc, cast_c, cast_l, weather, time_of_day, wardrobe, script_text):
            if not proj_name or not ep_label:
                return None, gr.update(), "Project and episode are required.", gr.update()
            project = db.get_project_by_name(proj_name)
            ep_num = int(ep_label.split(":")[0].replace("Ep ", "").strip())
            episode = db.get_episode_by_details(project["id"], ep_num)
            character = db.get_character_by_name(cast_c) if cast_c and cast_c != "None" else None
            location = db.get_location_by_name(cast_l) if cast_l and cast_l != "None" else None
            saved_id = db.add_scene(
                episode_id=episode["id"],
                scene_number=int(number),
                title=title.strip(),
                description=desc.strip(),
                character_id=character["id"] if character else None,
                location_id=location["id"] if location else None,
                shot_type="Scene Builder",
                prompt=desc.strip(),
                video_prompt=script_text.strip(),
            )
            db.set_continuity_state(
                saved_id,
                location_id=location["id"] if location else None,
                wardrobe_tag=wardrobe.strip(),
                weather=weather.strip(),
                time_of_day=time_of_day.strip(),
            )
            scenes = db.list_scenes(episode["id"])
            labels = [f"Scene {scene['scene_number']}: {scene['title'] or 'Untitled'}" for scene in scenes]
            return saved_id, gr.update(choices=labels, value=f"Scene {int(number)}: {title.strip() or 'Untitled'}"), "Scene saved.", gr.update()

        def build_scene_fn(scene_id, script_text, desc):
            if not scene_id:
                return [], "Save or select a scene first."
            shots = ScriptAnalyzer(db).analyze_scene_to_shots(scene_id, script_text, desc)
            table = [[s["shot_number"], s.get("shot_type", ""), s.get("character_name", ""), s.get("goal", ""), s.get("prompt", "")] for s in db.list_shots(scene_id)]
            return table, f"Generated {len(shots)} shot plan entries."

        def delete_scene_fn(scene_id, proj_name, ep_label):
            if not scene_id:
                return None, gr.update(), "Select a scene to delete."
            db.delete_scene(scene_id)
            project = db.get_project_by_name(proj_name) if proj_name else None
            ep_num = int(ep_label.split(":")[0].replace("Ep ", "").strip()) if ep_label else None
            episode = db.get_episode_by_details(project["id"], ep_num) if project and ep_num else None
            scenes = db.list_scenes(episode["id"]) if episode else []
            labels = [f"Scene {scene['scene_number']}: {scene['title'] or 'Untitled'}" for scene in scenes]
            return None, gr.update(choices=labels, value=labels[0] if labels else None), "Scene deleted."

        def new_scene_fn(proj_name, ep_label):
            next_num = 1
            if proj_name and ep_label:
                project = db.get_project_by_name(proj_name)
                ep_num = int(ep_label.split(":")[0].replace("Ep ", "").strip())
                episode = db.get_episode_by_details(project["id"], ep_num) if project else None
                scenes = db.list_scenes(episode["id"]) if episode else []
                if scenes:
                    next_num = max(scene["scene_number"] for scene in scenes) + 1
            return None, next_num, "", "", gr.update(value="None"), gr.update(value="None"), "", "", "", "", []

        proj_selector.change(load_episodes_for_proj, [proj_selector], [ep_selector])
        ep_selector.change(load_scenes_for_ep, [proj_selector, ep_selector], [scene_selector])
        proj_selector.change(load_casting_choices, [], [casting_char, casting_loc])
        scene_selector.change(select_scene_fn, [proj_selector, ep_selector, scene_selector], [active_scene_id, scene_num, scene_title, scene_desc, casting_char, casting_loc, continuity_weather, continuity_time, continuity_wardrobe, scene_script, shot_table])
        btn_save_scene.click(save_scene_fn, [active_scene_id, proj_selector, ep_selector, scene_num, scene_title, scene_desc, casting_char, casting_loc, continuity_weather, continuity_time, continuity_wardrobe, scene_script], [active_scene_id, scene_selector, scene_save_msg, scene_list_msg])
        btn_build_scene.click(build_scene_fn, [active_scene_id, scene_script, scene_desc], [shot_table, scene_save_msg])
        btn_delete_scene.click(delete_scene_fn, [active_scene_id, proj_selector, ep_selector], [active_scene_id, scene_selector, scene_list_msg])
        btn_new_scene.click(new_scene_fn, [proj_selector, ep_selector], [active_scene_id, scene_num, scene_title, scene_desc, casting_char, casting_loc, continuity_weather, continuity_time, continuity_wardrobe, scene_script, shot_table])

        return proj_selector, ep_selector, scene_selector, casting_char, casting_loc
