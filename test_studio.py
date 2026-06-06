import os
import sys
import unittest
import tempfile
import shutil

# Dynamic workspace path inclusion
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from studio.database.db_manager import DatabaseManager

class TestStudioDatabase(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for database isolation
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_studio.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        import gc
        gc.collect()
        try:
            shutil.rmtree(self.test_dir)
        except PermissionError:
            pass

    def test_project_crud(self):
        # Add Project
        proj_id = self.db.add_project("Tokyo Nights", "Cyberpunk detective noir story")
        self.assertIsNotNone(proj_id)
        
        # Get Project by ID
        proj = self.db.get_project(proj_id)
        self.assertIsNotNone(proj)
        self.assertEqual(proj["name"], "Tokyo Nights")
        
        # Delete Project
        self.db.delete_project(proj_id)
        self.assertIsNone(self.db.get_project(proj_id))

    def test_episode_crud(self):
        proj_id = self.db.add_project("Tokyo Nights")
        
        # Add Episode
        ep_id = self.db.add_episode(proj_id, 1, "The Rainy Window", "Kenji meets an informant")
        self.assertIsNotNone(ep_id)
        
        # List Episodes
        eps = self.db.list_episodes(proj_id)
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["title"], "The Rainy Window")

    def test_character_phase2_crud(self):
        char_folder = os.path.join(self.test_dir, "kenji")
        os.makedirs(char_folder, exist_ok=True)
        
        # Add Character with Phase 2 fields
        char_id = self.db.add_character(
            name="Detective Kenji",
            age=32,
            gender="Male",
            description="Wears a worn brown trench coat, robotic left arm",
            notes="Obsessed with a cold case",
            tags="cyberpunk,noir,detective",
            folder_path=char_folder,
            biography="Born in sector 9, joined police academy in 2012.",
            personality="Taciturn, meticulous, easily irritated.",
            prompt_template="A portrait of {name}, featuring {dna_hair} and {dna_eyes}.",
            wardrobe_notes="Trench coat, dark pants",
            expression_notes="Grim, thoughtful",
            voice_notes="Gravelly baritone",
            dna_hair="Short grey hair",
            dna_eyes="Cybernetic blue eyes",
            dna_body_type="Slim, athletic",
            dna_clothing="Signature grey trench coat",
            dna_ethnicity="East-Asian",
            dna_description="Faint scar on temple"
        )
        self.assertIsNotNone(char_id)
        
        # Get and check Phase 2 values
        char = self.db.get_character(char_id)
        self.assertEqual(char["name"], "Detective Kenji")
        self.assertEqual(char["biography"], "Born in sector 9, joined police academy in 2012.")
        self.assertEqual(char["personality"], "Taciturn, meticulous, easily irritated.")
        self.assertEqual(char["dna_hair"], "Short grey hair")
        self.assertEqual(char["dna_eyes"], "Cybernetic blue eyes")
        
        # Update Character with new Phase 2 details
        self.db.update_character(
            char_id=char_id,
            name="Detective Kenji",
            age=33,
            gender="Male",
            description="Robotic arm glows soft cyan",
            notes="Still obsessed with the case",
            tags="cyberpunk,cyan-glow,detective",
            biography="Updated bio details.",
            personality="Slightly more relaxed now.",
            prompt_template="Updated template",
            wardrobe_notes="New coat",
            expression_notes="Smiling",
            voice_notes="Warm tone",
            dna_hair="Shaved hair",
            dna_eyes="Golden cybernetic eyes",
            dna_body_type="Muscular",
            dna_clothing="High-collar leather vest",
            dna_ethnicity="East-Asian",
            dna_description="Interface ports on temple"
        )
        
        updated_char = self.db.get_character(char_id)
        self.assertEqual(updated_char["age"], 33)
        self.assertEqual(updated_char["biography"], "Updated bio details.")
        self.assertEqual(updated_char["personality"], "Slightly more relaxed now.")
        self.assertEqual(updated_char["dna_eyes"], "Golden cybernetic eyes")

    def test_character_relationships(self):
        char_id_1 = self.db.add_character("Kenji", 32, "Male", "", "", "", "folder_k")
        char_id_2 = self.db.add_character("Reiko", 28, "Female", "", "", "", "folder_r")
        
        # Add Relationship
        rel_id = self.db.add_relationship(char_id_1, char_id_2, "Rival", "Rivalry started at the academy")
        self.assertIsNotNone(rel_id)
        
        # List Relationships
        rels = self.db.list_relationships(char_id_1)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["target_character_name"], "Reiko")
        self.assertEqual(rels[0]["relationship_type"], "Rival")
        
        # Delete Relationship
        self.db.delete_relationship(rel_id)
        self.assertEqual(len(self.db.list_relationships(char_id_1)), 0)

    def test_search_and_filters(self):
        # Insert test data
        char_k = self.db.add_character("Detective Kenji", 32, "Male", "A cyborg detective", "", "cyberpunk,noir", "folder_k")
        char_r = self.db.add_character("Hacker Reiko", 28, "Female", "An elite cyber deck runner", "", "cyberpunk,hacker", "folder_r")
        char_y = self.db.add_character("Yasuke", 40, "Male", "A street brawler bodyguard", "", "bodyguard,street", "folder_y")
        
        # Search by query
        results = self.db.search_characters(search_query="hacker")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Hacker Reiko")
        
        # Filter by gender
        results = self.db.search_characters(gender="Male")
        self.assertEqual(len(results), 2)
        names = {r["name"] for r in results}
        self.assertIn("Detective Kenji", names)
        self.assertIn("Yasuke", names)
        
        # Filter by tag
        results = self.db.search_characters(tag="hacker")
        self.assertEqual(len(results), 1)
        
        # Setup scenes to test project/location casting filtration
        proj_id = self.db.add_project("Tokyo Nights")
        ep_id = self.db.add_episode(proj_id, 1, "The Beginning")
        loc_id = self.db.add_location("Noodle Shop", "", "", "", "folder_l")
        
        self.db.add_scene(ep_id, 1, "Intro", "Kenji is eating", character_id=char_k, location_id=loc_id)
        
        # Filter by project_id
        results = self.db.search_characters(project_id=proj_id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Detective Kenji")
        
        # Filter by location_id
        results = self.db.search_characters(location_id=loc_id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Detective Kenji")

    # ==========================================
    # PHASE 3: NEW LOCATIONS & SUB-LOCATIONS
    # ==========================================
    def test_location_prompt_and_sub_locations(self):
        loc_id = self.db.add_location(
            name="Neo Shibuya Penthouse",
            description="High-rise apartment overlooking cyber billboards",
            tags="luxury,modern,high-altitude",
            notes="Director note: Use blue/pink lighting presets",
            folder_path=os.path.join(self.test_dir, "penthouse"),
            prompt_template="A premium photo of {name}, {description}."
        )
        self.assertIsNotNone(loc_id)
        
        # Check details
        loc = self.db.get_location(loc_id)
        self.assertEqual(loc["prompt_template"], "A premium photo of {name}, {description}.")
        
        # Add room (sub-location)
        room1_id = self.db.add_sub_location(
            location_id=loc_id,
            name="Bedroom",
            description="Spacious bed with hologram curtains",
            prompt_template="{parent_prompt}, bedroom detailing",
            folder_path=os.path.join(self.test_dir, "penthouse", "bedroom")
        )
        self.assertIsNotNone(room1_id)
        
        # Add another room
        room2_id = self.db.add_sub_location(
            location_id=loc_id,
            name="Living Room",
            description="L-shaped leather couch, neon accents",
            prompt_template="{parent_prompt}, specifically living room setup",
            folder_path=os.path.join(self.test_dir, "penthouse", "living_room")
        )
        self.assertIsNotNone(room2_id)
        
        # List rooms
        rooms = self.db.list_sub_locations(loc_id)
        self.assertEqual(len(rooms), 2)
        room_names = {r["name"] for r in rooms}
        self.assertIn("Bedroom", room_names)
        self.assertIn("Living Room", room_names)
        
        # Delete room
        self.db.delete_sub_location(room1_id)
        rooms_after = self.db.list_sub_locations(loc_id)
        self.assertEqual(len(rooms_after), 1)
        self.assertEqual(rooms_after[0]["name"], "Living Room")

    def test_location_search_and_filters(self):
        # Insert test locations
        loc1 = self.db.add_location("Hacker Cave", "Tech screens, messy", "underground,tech", "Notes 1", "folder_loc1")
        loc2 = self.db.add_location("Cyber Bar", "Neon stools, dark vibe", "bar,neon,indoor", "Notes 2", "folder_loc2")
        loc3 = self.db.add_location("Wasteland Outpost", "Rusty panels, sunny", "outdoor,desert", "Notes 3", "folder_loc3")
        
        # Search by query
        results = self.db.search_locations(search_query="neon")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Cyber Bar")
        
        # Search by tag
        results = self.db.search_locations(tag="underground")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Hacker Cave")
        
        # Setup scenes to test castings
        proj_id = self.db.add_project("Cyber Heist")
        ep_id = self.db.add_episode(proj_id, 1, "Episode 1")
        char_k = self.db.add_character("Kenji", 30, "Male", "", "", "", "folder_k")
        
        # Cast Kenji in Hacker Cave
        self.db.add_scene(ep_id, 1, "Hack Room scene", "Kenji talks to Reiko", character_id=char_k, location_id=loc1)
        
        # Filter location by project
        results = self.db.search_locations(project_id=proj_id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Hacker Cave")
        
        # Filter location by character casting
        results = self.db.search_locations(character_id=char_k)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Hacker Cave")

    # ==========================================
    # PHASE 3: STORY BIBLE TIMELINE & LORE
    # ==========================================
    def test_story_bible_timeline_and_notes(self):
        proj_id = self.db.add_project("Cyber Heist")
        
        # Add Timeline Events
        ev1 = self.db.add_timeline_event(
            project_id=proj_id,
            event_order=2,
            title="The Vault Breach",
            description="The vault door is bypassed using the EMP generator.",
            event_date="Day 5, 01:00 AM"
        )
        self.assertIsNotNone(ev1)
        
        ev2 = self.db.add_timeline_event(
            project_id=proj_id,
            event_order=1,
            title="The Blueprint Theft",
            description="Reiko steals the master access blueprints.",
            event_date="Day 1, 14:00 PM"
        )
        self.assertIsNotNone(ev2)
        
        # List and verify sorting (order ASC)
        events = self.db.list_timeline_events(proj_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["title"], "The Blueprint Theft") # order 1 comes first
        self.assertEqual(events[1]["title"], "The Vault Breach")     # order 2 comes second
        
        # Delete event
        self.db.delete_timeline_event(ev1)
        events_after = self.db.list_timeline_events(proj_id)
        self.assertEqual(len(events_after), 1)
        self.assertEqual(events_after[0]["title"], "The Blueprint Theft")
        
        # Add Story Notes (Lore entries)
        note1_id = self.db.add_story_note(
            project_id=proj_id,
            title="Shibuya Corp Lore",
            content="Shibuya Corp dominates biotechnology and cybernetic implants.",
            category="Lore"
        )
        self.assertIsNotNone(note1_id)
        
        note2_id = self.db.add_story_note(
            project_id=proj_id,
            title="Neon Noir Styling Guide",
            content="Always keep lighting low key with saturated cyan and purple accents.",
            category="Tone Guide"
        )
        self.assertIsNotNone(note2_id)
        
        # List all notes
        notes = self.db.list_story_notes(proj_id)
        self.assertEqual(len(notes), 2)
        
        # List notes by category
        lore_notes = self.db.list_story_notes(proj_id, category="Lore")
        self.assertEqual(len(lore_notes), 1)
        self.assertEqual(lore_notes[0]["title"], "Shibuya Corp Lore")
        
        # Update note
        self.db.update_story_note(
            note_id=note1_id,
            title="Shibuya Corp Lore - Updated",
            content="Updated biotech lore.",
            category="Lore"
        )
        updated_note = self.db.list_story_notes(proj_id, category="Lore")[0]
        self.assertEqual(updated_note["title"], "Shibuya Corp Lore - Updated")
        
        # Delete note
        self.db.delete_story_note(note1_id)
        self.assertEqual(len(self.db.list_story_notes(proj_id)), 1)

if __name__ == "__main__":
    unittest.main()
