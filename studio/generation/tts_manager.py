"""
studio/generation/tts_manager.py
Phase 9: Kokoro TTS Voice Generation
"""
import os
import time
import logging

try:
    from kokoro import KPipeline
    import soundfile as sf
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    logging.warning("Kokoro TTS is not installed. TTS generation will be simulated. Install with `pip install kokoro soundfile`")

class TTSManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.pipeline = None

    def load_pipeline(self):
        if not KOKORO_AVAILABLE:
            return
        if self.pipeline is None:
            logging.info("Loading Kokoro TTS pipeline (American English)...")
            self.pipeline = KPipeline(lang_code='a')
            logging.info("Kokoro pipeline loaded.")

    def unload_pipeline(self):
        if self.pipeline is not None:
            logging.info("Unloading Kokoro TTS pipeline...")
            self.pipeline = None

    def generate_voice(self, text: str, voice_id: str, speed: float = 1.0) -> str:
        """
        Generates audio for the given text using the specified voice ID.
        Returns the absolute path to the generated .wav file.
        """
        timestamp = int(time.time())
        filename = f"tts_{voice_id}_{timestamp}.wav"
        output_path = os.path.join(self.output_dir, filename)

        if not KOKORO_AVAILABLE:
            logging.info(f"[SIMULATED TTS] Text: '{text}', Voice: {voice_id}, Speed: {speed}")
            time.sleep(2)
            # Create a dummy file
            with open(output_path, "w") as f:
                f.write("Dummy audio file. Kokoro not installed.")
            return output_path

        self.load_pipeline()
        
        # Kokoro expects voice_id like 'am_adam', 'af_heart'
        if not voice_id:
            voice_id = "af_heart" # Default female voice
            
        logging.info(f"Generating TTS with Kokoro. Voice: {voice_id}")
        
        # Generate generator yielding (graphemes, phonemes, audio)
        generator = self.pipeline(
            text, voice=voice_id,
            speed=speed, split_pattern=r'\n+'
        )
        
        all_audio = []
        for i, (gs, ps, audio) in enumerate(generator):
            if audio is not None:
                all_audio.extend(audio.tolist() if hasattr(audio, 'tolist') else audio)
                
        if all_audio:
            sf.write(output_path, all_audio, 24000)
            return output_path
        else:
            raise ValueError("Kokoro generated empty audio.")
