import whisper
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SpeechToTextModel:
    def __init__(self, model_size="base"):
        """
        Initializes the Whisper model.

        Args:
            model_size (str): Size of the Whisper model to load (tiny, base, small, medium, large).
        """
        logger.info(f"Loading Whisper model of size '{model_size}'...")
        self.model = whisper.load_model(model_size)
        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_file_path):
        """
        Transcribes the given audio file to text using Whisper.

        Args:
            audio_file_path (str): Path to the audio file.

        Returns:
            str: The transcribed text.
        """
        logger.info(f"Attempting to transcribe audio file '{audio_file_path}'...")
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file does not exist at path: {audio_file_path}")
            return "An error occurred during transcription."

        try:
            # Transcribe the audio file directly
            result = self.model.transcribe(audio_file_path)
            transcription = result['text'].strip()
            logger.info(f"Transcription: {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            return "An error occurred during transcription."
