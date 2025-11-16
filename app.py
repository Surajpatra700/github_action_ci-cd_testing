# import torch
# from torch.serialization import add_safe_globals

# # Import ALL XTTS classes that torch.load needs
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import (
#     XttsArgs,
#     XttsAudioConfig,
#     XttsPretrainedArgs,
#     XttsTokenizer,
#     XTTSSynthesizer
# )

# # Allowlist EVERYTHING for PyTorch 2.6+
# add_safe_globals([
#     XttsConfig,
#     XttsArgs,
#     XttsAudioConfig,
#     XttsPretrainedArgs,
#     XttsTokenizer,
#     XTTSSynthesizer
# ])

# from TTS.api import TTS
# from scipy.io.wavfile import write

# # Load FREE XTTS-v2 model
# model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
# tts = TTS(model_name=model_name, gpu=False)

# # Your reference voice
# voice_samples = [
#     "input/Standard recording 69.wav",
#     "input/Standard recording 70.wav",
#     "input/Standard recording 72.wav",
#     "input/Standard recording 73.wav"
# ]

# text = "Main aaj thoda busy hoon. Kal milte hain, theek hai?"

# output_path = "output_my_voice.wav"

# # Generate
# wav = tts.tts(
#     text=text,
#     speaker_wav=voice_samples,
#     language="hi",
#     split_sentences=False
# )

# # Save
# write(output_path, 24000, wav)

# print("Done! Saved:", output_path)


from TTS.api import TTS
from scipy.io.wavfile import write

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

voice_samples = [
    "input/Standard recording 69.wav",
    "input/Standard recording 70.wav",
    "input/Standard recording 72.wav",
    "input/Standard recording 73.wav"
]

text = "Aaj main tumse baat kar raha hoon. Yeh awaaz mera hi clone hai."

audio = tts.tts(
    text=text,
    speaker_wav=voice_samples,
    language="hi"
)

write("output.wav", 24000, audio)

print("Done!")