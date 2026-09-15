import torch
import sounddevice as sd
from transformers import pipeline
import numpy as np
# import pyttsx3
# import os
# import speech_recognition as sr
# import pyttsx3
# import ollama
# import json
# from vosk import Model, KaldiRecognizer
# import pyaudio
# import espeakng_loader
#
# # Force the environment to see the espeak binaries inside your python folder
# os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeakng_loader.get_library_path()
# os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.dirname(espeakng_loader.get_library_path())
#
#
# # --- FORCE PYTORCH TO TRUST XTTS CHECKPOINTS ---
# # This forces weights_only=False dynamically when loading the model layers
# original_torch_load = torch.load
# def safe_torch_load(*args, **kwargs):
#     kwargs['weights_only'] = False
#     return original_torch_load(*args, **kwargs)
# torch.load = safe_torch_load
#
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# REF_TEXT = "Hey! My name is josh and i will be chatting with you today. Hope you're excited"
# REF_AUDIO = os.path.join(SCRIPT_DIR, "josh_REF.wav")