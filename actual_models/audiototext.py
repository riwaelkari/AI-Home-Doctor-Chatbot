# actual_models/audiototext.py

import whisper
import logging
import os
import shutil
import subprocess

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SpeechToTextModel:
    def __init__(self, model_size="base.en"):
        """
        Initializes the Whisper model.

        Args:
            model_size (str): Size of the Whisper model to load (tiny, base, small, medium, large).
        """
        # Prepend ffmpeg bin directory to PATH
        ffmpeg_path = r"D:/Download/ffmpeg-7.1-essentials_build/ffmpeg-7.1-essentials_build/bin"
        if os.path.isdir(ffmpeg_path):
            os.environ["PATH"] = ffmpeg_path + ";" + os.environ["PATH"]
            logger.info(f"Added {ffmpeg_path} to system PATH.")
        else:
            logger.error(f"ffmpeg bin directory does not exist at path: {ffmpeg_path}")
            raise EnvironmentError("ffmpeg is not installed correctly.")

        # Check if ffmpeg is available
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg is not installed or not found in PATH.")
            raise EnvironmentError("ffmpeg is required but not found. Please install ffmpeg and ensure it's in your system's PATH.")

        logger.info(f"Loading Whisper model of size '{model_size}'...")
        self.model = whisper.load_model(model_size)
        logger.info("Whisper model loaded successfully.")

    def convert_webm_to_mp3(self, webm_path, mp3_path):
        """
        Converts a WebM audio file to MP3 using ffmpeg.

        Args:
            webm_path (str): Path to the input WebM file.
            mp3_path (str): Path to the output MP3 file.

        Returns:
            bool: True if conversion is successful, False otherwise.
        """
        try:
            logger.info(f"Converting '{webm_path}' to '{mp3_path}'...")
            print("hi1")
            subprocess.run([
                "ffmpeg",
                "-i", webm_path,
                "-vn",  # No video
                "-ar", "44100",  # Audio sampling rate
                "-ac", "2",      # Number of audio channels
                "-b:a", "192k",  # Audio bitrate
                mp3_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Conversion to MP3 successful.")
            print("hi")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg conversion failed: {e.stderr.decode()}")
            return False

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

        # Define MP3 path
        base, ext = os.path.splitext(audio_file_path)
        mp3_path = base + ".mp3"
        # Convert WebM to MP3
        conversion_success = self.convert_webm_to_mp3(audio_file_path, mp3_path)
        if not conversion_success:
            return "An error occurred during audio conversion."

        try:
            print("hi2")
            # Transcribe the MP3 file
            result = self.model.transcribe(mp3_path)
            transcription = result['text'].strip()
            logger.info(f"Transcription: {transcription}")

            # Optionally, delete the MP3 file after transcription
            os.remove(mp3_path)

            return transcription
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            return "An error occurred during transcription."
