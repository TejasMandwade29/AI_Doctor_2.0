# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

import os
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs

# --- This is the only function you need for gTTS ---
def text_to_speech_with_gtts(input_text, output_filepath):
    """Creates an audio file from text using gTTS and saves it."""
    try:
        audio_obj = gTTS(text=input_text, lang="en", slow=False)
        audio_obj.save(output_filepath)
        print(f"gTTS audio saved to {output_filepath}")
        return output_filepath
    except Exception as e:
        print(f"Error with gTTS: {e}")
        return None

# --- This is the only function you need for ElevenLabs ---
def text_to_speech_with_elevenlabs(input_text, output_filepath):
    """Creates an audio file from text using ElevenLabs and saves it."""
    try:
        ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
        if not ELEVEN_API_KEY:
            raise ValueError("ELEVEN_API_KEY not found in your .env file")

        client = ElevenLabs(api_key=ELEVEN_API_KEY)
        audio = client.generate(
            text=input_text,
            voice="Aria",
            output_format="mp3_22050_32",
            model="eleven_turbo_v2"
        )
        elevenlabs.save(audio, output_filepath)
        print(f"ElevenLabs audio saved to {output_filepath}")
        return output_filepath
    except Exception as e:
        print(f"Error with ElevenLabs: {e}")
        return None
