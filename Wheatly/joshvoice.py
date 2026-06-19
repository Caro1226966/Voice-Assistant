# -----------------------------------------------
import os
import torch
import threading
from queue import Queue
import sounddevice as sd
from TTS.api import TTS

class JoshVoice:
    def __init__(self, speaker="p234"):
        gpu_available = torch.cuda.is_available()
        self.tts = TTS("tts_models/en/vctk/vits", progress_bar=False, gpu=gpu_available)
        self.speaker = speaker
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._current_playing = None

    def say(self, text):
        self._queue.put(text)

    def runAndWait(self):
        self._queue.join()

    def stop(self):
        self._stop_event.set()
        self.interrupt()
        while not self._queue.empty():
            self._queue.get_nowait()
        self._queue.join()

    def interrupt(self):
        if self._current_playing:
            sd.stop()
        while not self._queue.empty():
            self._queue.get_nowait()

    def _run_loop(self):
        while not self._stop_event.is_set():
            text = self._queue.get()
            try:
                wav = self.tts.tts(text, speaker=self.speaker)
                self._current_playing = True
                sd.play(wav, samplerate=self.tts.synthesizer.output_sample_rate)
                sd.wait()
                self._current_playing = False
            except Exception as e:
                print(f"[JoshVoice Error] {e}")
            self._queue.task_done()