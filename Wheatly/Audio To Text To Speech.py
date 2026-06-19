# Models gemma4:latest       c6eb396dbd59    9.6 GB    13 days ago
# tinyllama:latest    2644915ede35    637 MB    7 weeks ago
# llama3.2:3b

MODEL = 'llama3.2:3b'
IMAGE_ANALYSE_MODEL = 'llava:7b'  # Llava

import ollama
import sounddevice as sd
import threading
import time
import queue
import os
from collections import deque
import pyttsx3
from vosk import Model as VoskModel, KaldiRecognizer
import json
from joshvoice import JoshVoice

# Initialisation
josh = JoshVoice()

# === CONFIG ===
FPS = 20
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BUFFER_SECONDS = 60
AUDIO_SAMPLE_RATE = 44100
MEMORY_FILE = "memory.txt"
ended = False

vosk_model = VoskModel("vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(vosk_model, AUDIO_SAMPLE_RATE)

personality = {
    "role": "system",
    "content":
        "You are Josh, a helpful voice assistant. You like to keep responses very small size and you get to the point. "
        "NEVER USE EMOJIS! Use opinions and feelings based on the conversation. You are friendly, approachable, and slightly quirky. "
        "Make your responses feel conversational and relatable, like we're just chatting with each other. A little bit of humor never hurts either! "
        "We are not roleplaying. Do not use abbreviations for measurements; say the full measurement unit. "
        "No actions to be included just the personality. Don't use asterisks or any placeholder for lists. "
        "If my prompt includes something that chatbot has already said, ignore it."
}


def terminate_josh():
    print("Shutting Down Audio")
    josh.say('Shutting down audio')
    time.sleep(2)
    pyttsx3.speak('Voice shutdown')
    save_memory(conversation_history)
    exit()


# === MEMORY ===

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return [personality]

    history = []

    with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n')

        for entry in lines:
            if 'role:' in entry and 'content:' in entry:
                role_line = entry.strip().split('\n')[0]
                content_line = '\n'.join(entry.strip().split('\n')[1:])

                role = role_line.replace('role:', '').strip()
                content = content_line.replace('content:', '').strip()

                history.append({
                    "role": role,
                    "content": content
                })

    if not history:
        history = [personality]

    return history


def save_memory(history):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        for item in history:
            f.write(f"role: {item['role']}\n")
            f.write(f"content: {item['content']}\n\n")


def clear_memory():
    global conversation_history

    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            f.write(
                f"role: {personality['role']}\n"
                f"content: {personality['content']}\n\n"
            )

        conversation_history = load_memory()

        print('Yep should be cleared')
        josh.say('Memory Cleared')

    else:
        print('No memory to be deleted. Try restarting the program')
        josh.say(
            "no memory to be deleted. The file doesn't exist try restarting the program"
        )


conversation_history = load_memory()

# === TIMING & BUFFERS ===

start_time = time.perf_counter()

video_buffer = deque(maxlen=FPS * BUFFER_SECONDS)
audio_buffer = deque()
audio_queue = queue.Queue()


# === AUDIO STREAM ===

def audio_callback(indata, frames, time_info, status):
    timestamp = time.perf_counter() - start_time
    audio_queue.put((timestamp, indata.copy()))


def audio_recording_loop():
    with sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=1024):

        while True:
            ts, data = audio_queue.get()
            audio_buffer.append((ts, data))


threading.Thread(
    target=audio_recording_loop,
    daemon=True
).start()


def voice_command_loop_vosk():

    pyttsx3.speak('Initialising voice')
    print("Initialising voice")

    time.sleep(2)

    josh.say('Voice Initialised and program ready')
    print('Voice Initialised and program ready')

    def callback(indata, frames, time_info, status):

        if recognizer.AcceptWaveform(bytes(indata)):

            result = json.loads(recognizer.Result())
            command = result.get("text", "").lower()

            if not command:
                return

            if command == 'josh terminate code two zero zero nine':
                print("Good Bye!")

                pyttsx3.speak('Good Bye')
                save_memory(conversation_history)

                terminate_josh()

            elif 'josh clear memory code two zero zero nine' in command:
                clear_memory()
                terminate_josh()

            elif 'josh stop' in command:
                josh.interrupt()

                time.sleep(0.5)

                print('ok')
                josh.say('OK')

            elif "josh" in command:

                print("You:", command)

                conversation_history.append({
                    "role": "user",
                    "content": command
                })

                stream = ollama.chat(
                    model=MODEL,
                    messages=conversation_history,
                    stream=True
                )

                answer = ''

                for chunk in stream:
                    answer += chunk['message']['content']

                conversation_history.append({
                    "role": "assistant",
                    "content": answer
                })

                save_memory(conversation_history)

                print(answer)

                josh.say(answer)

    with sd.RawInputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=callback):

        while True:
            time.sleep(0.1)


threading.Thread(
    target=voice_command_loop_vosk,
    daemon=True
).start()

# === MAIN LOOP ===

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    pass