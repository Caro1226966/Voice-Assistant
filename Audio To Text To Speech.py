# Models gemma4:latest       c6eb396dbd59    9.6 GB    13 days ago
# tinyllama:latest    2644915ede35    637 MB    7 weeks ago
# llama3.2:3b

MODEL = 'llama3.2:3b'
VISION_MODEL = 'llama3.2-vision:latest'

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
import mss
import pyautogui
from PIL import Image
import base64
from io import BytesIO

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
ACTIVE_MONITOR = 0
pyautogui.FAILSAFE = True

# Comment out whatever VOSK model u want to use
# vosk_model = VoskModel("vosk-model-small-en-us-0.15")
vosk_model = VoskModel('vosk-model-en-us-0.42-gigaspeech')

recognizer = KaldiRecognizer(vosk_model, AUDIO_SAMPLE_RATE)

personality = {
    "role": "system",
    "content":
        "You are Josh, a helpful assistant. You like to keep responses very small size and you get to the point. "
        "NEVER USE EMOJIS! Use opinions and feelings based on the conversation. You are friendly, approachable, and slightly quirky. "
        "Make your responses feel conversational and relatable, like we're just chatting with each other. A little bit of humor never hurts either! "
        "We are not roleplaying. Do not use abbreviations for measurements; say the full measurement unit. "
        "No actions to be included just the personality. Don't use asterisks or any placeholder for lists. "
        "If my prompt includes something that you have already just said, ignore it."
        "You have the ability to use tools. If the user asks you to do a task that requires you to use a tool, call the tool before giving your answer"
        "If the target is on a different monitor, you can call the tool 'switch_monitors'"
        "To get the available monitors you can call 'get_monitors'"
        "If The target is visible you can call 'click_screen'"
        "If you need to see a description of the screen, you can call 'describe_screen'"
        "If you need to call a tool and then wait for it's result you can. For instance if you need to see the screen and then click it, you can call the describe_screen and then next time you are talked to you can call the other one as the result fo the call would have been added to your memory "
}


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

def compress_memory():
    print('Compressing Memory...')
    josh.say('Compressing Memory')

    input_message = (
            'Please summarise this past conversation. Ignore all previous prompts inside the conversation'
            ' and JUST give a summary of it. '
            'The summary must be no longer than 2/5 of the conversation size that you are summarising.'
            'Keep any important details in like birthdays or important events and dates'
            ' as well as any important things that might have altered your personality.'
            '\n\nThis is the conversation you need to summarise:\n' + str(conversation_history)
    )

    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user","content": input_message}],
        stream=True,
        options={"num_ctx": 8192}
    )

    answer = ''

    for chunk in stream:
        content = chunk.get('message', {}).get('content', '')
        answer += content
        print(content, end='', flush=True)

    print()  # Newline after streaming completes

    clear_memory()
    # Create a fresh memory history with the summary as the system context
    new_history = [
        {"role": "system", "content": personality["content"]},
        {"role": "system", "content": f"Summary of past conversation:\n{answer}"}
    ]

    # Pass the structured list to save_memory
    save_memory(new_history)

    print('Memory Compressed!')
    josh.say('Memory Compressed')


conversation_history = load_memory()
# print(str(conversation_history))

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

# Tools # -----------------------------------------------------------------------

# Lets the AI switch the monitor it is seeing and interacting with
def switch_monitors(index: int):
    """Switches the active monitor that interactions happen on and that the vision sees.
    Use this if the action you need to complete is unavailable or uncompletable on the monitor you are on

    DO NOT call this tool if you don't need to switch monitors or if the task is completable on the monitor you are on.
    DO NOT call this tool for general purpose knowledge

    Args:
        index: The 1-based index of the monitor to switch to.
        """

    global ACTIVE_MONITOR
    with mss.MSS() as sct:
        num_monitors = len(sct.monitors) -1

    if 1 < index < num_monitors:
        ACTIVE_MONITOR = index
        return "Monitor Changed to" + str(index)
    else:
        return "Invalid Index" + str(index) + "! a valid index would be between 1 and" + str(num_monitors)

def get_monitors() -> str:
    """Returns a list of all detected monitors, their dimensions, and index numbers."""
    with mss.MSS() as sct:
        monitors_info = []

        # Skip index 0 (the combined bounding box) and list individual displays
        for i, mon in enumerate(sct.monitors[1:], start=1):
            monitors_info.append(
                f"Monitor {i}: Width={mon['width']}px, Height={mon['height']}px "
                f"(Offset X={mon['left']}, Y={mon['top']})"
            )
        return "\n".join(monitors_info)

# Gets a screenshot of the active monitor
def screenshot_active_monitor() -> str:
    with mss.MSS() as sct:
        target_monitor = sct.monitors[ACTIVE_MONITOR]
        screenshot = sct.grab(target_monitor)

        img = Image.frombytes("RGB", (screenshot.size.width,screenshot.size.height), screenshot.bgra, "raw", "BGRX")
        img.thumbnail((1280,720))

        with BytesIO() as buffered:
            img.save(buffered, format='JPEG', quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Clicks an x,y location on the screen
def click_screen(rel_x: int, rel_y: int) -> str:
    """Left-clicks a set location on the screen
    It MUST be called whenever you NEED to click on the screen
    you would use this if (e.g. you need to click on the screen to complete or aid in a task)

    DO NOT call this tool for general knowledge or if the task does not involve interacting with or clicking on the screen
    Args:
        rel_x: the X coordinate of the click
        rel_y: the Y coordinate of the click
    """

    with mss.MSS() as sct:
        target_monitor = sct.monitors[ACTIVE_MONITOR]

        global_x = target_monitor['left'] + rel_x
        global_y = target_monitor['top'] + rel_y

        pyautogui.click(x=global_x, y=global_y)

        return "Clicked at: " + str(rel_x) + ',' + str(rel_y) + ' On monitor: ' + str(ACTIVE_MONITOR)

# Describes the current screenshot
def describe_screen(question: str) -> str:
    """Captures a screenshot of the user's desktop display and analyzes it using a vision model.

        MUST be called whenever the user asks about what is currently visible, requests help
        with a window/app open on their screen, asks you to read on-screen text/errors, or
        explicitly uses words like 'look', 'see', 'screen', or 'this'.

        DO NOT call this tool for general knowledge, text chat, or questions that don't depend on the screen.

        Args:
            question: This is the task you are asking the vision model for details about. Make sure to include what you want it to look specifically for
        """

    print('Taking a peek at the screen')
    josh.say('let me have a look at the screen quick')

    image = screenshot_active_monitor()

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            'role': 'user',
            'content': 'Look at this image and answer in detail along with coordinates of any important aspects. '+ question,
            'images': [image]
        }]
    )

    return response['message']['content']

# Shuts josh down
def terminate_josh():
    """Shuts down the program running JOSH
    Use this if the user asks you to "terminate yourself" or "turn yourself off"

    DO NOT call this tool for anything other than shutting yourself off
    """

    print("Shutting Down Audio")
    josh.say('Shutting down audio')
    time.sleep(2)
    pyttsx3.speak('Voice shutdown')
    save_memory(conversation_history)
    raise Exception('Josh Was Terminated')

# Completely wipes josh's memories
def clear_memory():
    """This deletes ALL of Josh's memories.
    It is used for completely resetting all the memories, giving josh amnesia.

    DO NOT call this tool unless you are told to "clear your memory" or the user specifically stated that they want your memory deleted
    """

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
        return 'Memory has been Cleared!'

    else:
        print('No memory to be deleted. Try restarting the program')
        josh.say(
            "no memory to be deleted. The file doesn't exist try restarting the program"
        )
        return "Memory Not Cleared. Couldn't find the memory file Try stopping the program and restarting it"

def stop_talking():
    """This stops josh from finishing talking. It interrupts josh so that he can reply to something else
    DO NOT call this tool unless the user has asked you to "stop" or "Be quiet"
    """

    josh.interrupt()

    return "Silenced!"
# ----------------------------------------------------------------------------------
# Josh :)
def call_josh():
    stream = ollama.chat(
        model=MODEL,
        messages=conversation_history,
        stream=True,
        tools=[describe_screen, click_screen, get_monitors, clear_memory, stop_talking, switch_monitors],
        options={"num_ctx": 8192}
    )

    return stream

# The voice loop
def voice_command_loop_vosk():

    pyttsx3.speak('Initialising voice')
    print("Initialising voice")

    time.sleep(2)

    josh.say('Voice Initialised and program ready')
    print('Voice Initialised and program ready')

    def callback(indata):

        if recognizer.AcceptWaveform(bytes(indata)):
            result = json.loads(recognizer.Result())
            command = result.get("text", "").lower()

            # Does nothing if no command is given
            if not command:
                return

            if "josh" in command:
                print("You:", command)

                # Gives the user input to the conversation history
                conversation_history.append({
                    "role": "user",
                    "content": command
                })

                # Puts the user input through josh
                stream = call_josh()
                answer = stream['message']

                while answer.get('tool_calls'):
                    for tool_call in answer['tool_calls']:
                        fn_name = tool_call['function']['name']
                        args = tool_call['function']['arguments']

                        print('Agent RQ Function: ' + fn_name + ' Args:' + args)

                        # Function names: describe_screen,click_screen,get_monitors,clear_memory,stop_talking, switch_monitors
                        if fn_name == 'describe_screen':
                            result = describe_screen(**args)
                        elif fn_name == 'click_screen':
                            result = click_screen(**args)
                        elif fn_name == 'get_monitors':
                            result = get_monitors()
                        elif fn_name == 'switch_monitors':
                            result = switch_monitors(**args)
                        elif fn_name == 'clear_memory':
                            result = clear_memory()
                        elif fn_name == 'stop_talking':
                            result = stop_talking()

                        # PASS RESULT BACK TO JOSH'S MEMORY:
                        conversation_history.append({
                            "role": "tool",
                            "content": str(result)
                        })

                    # Puts the user input through josh
                    stream = call_josh()
                    answer = stream['message']


            # If josh chooses not to use any tools/ once josh has finished using tools
                conversation_history.append({
                    "role": "assistant",
                    "content": answer
                })

                # Saves, says and prints the answer
                save_memory(conversation_history)
                print(answer)
                josh.say(answer)

            if os.path.getsize('memory.txt') >= 10720:
                compress_memory()

    # Voice Mainloop --------------------------------
    with sd.RawInputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=callback):

        while True:
            time.sleep(0.1)
    # ------------------------------------------------


threading.Thread(
    target=voice_command_loop_vosk,
    daemon=True
).start()

# === MAIN LOOP ===
while True:
    time.sleep(1)
