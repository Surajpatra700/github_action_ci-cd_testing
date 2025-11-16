# from TTS.api import TTS
# import os

# print("Loading Models...")
# tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

# reference_audio = "C:\\Users\\HP\\Downloads\\Recording (2).mp3"

# if not os.path.exists(reference_audio):
#     print(f"Reference audio file '{reference_audio}' not found.")

# user_text = input("\nEnter Text for the clone")

# clone_audio = "clone_output.mp3"

# tts.tts_to_file(
#     text = user_text,
#     speaker_wav = reference_audio,
#     file_path = clone_audio,
#     language = "en"
# )

# print(f"Clone audio saved to '{clone_audio}'")

from TTS.api import TTS
import os

print("Loading model …")
tts = TTS(model_name="coqui/xtts-v2", progress_bar=False, gpu=False)   # set gpu=True if you have GPU environment

# reference audio: your recorded sample (WAV preferred)
reference_audio = "C:\\Users\\HP\\Downloads\\Recording (2).mp3"
if not os.path.exists(reference_audio):
    raise FileNotFoundError(f"Reference audio file '{reference_audio}' not found.")

user_text = input("Enter your text (Hinglish): \n")

output_file = "clone_output.wav"

tts.tts_to_file(
    text = user_text,
    speaker_wav = reference_audio,
    file_path = output_file,
    language = "hi"   # you can set "hi" for Hindi or "en" for English; for Hinglish you might set “hi” and include English words in your text
)

print(f"Clone audio saved to '{output_file}'")