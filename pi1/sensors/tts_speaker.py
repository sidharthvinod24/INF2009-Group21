"""
speaker/tts_speaker.py
=======================
Text-to-speech announcements via USB speaker for Pi 1.

Uses pyttsx3 with the espeak backend — fully offline, no cloud dependency.
Runs in a background thread with a bounded queue so the inference loop
is never blocked by audio playback.

Speech is queued and played in order. If the queue is full (e.g. multiple
events fire rapidly), the oldest message is dropped to keep alerts current.

Install:
    pip install pyttsx3
    sudo apt-get install espeak espeak-data libespeak1

Usage:
    speaker = TTSSpeaker()
    speaker.start()
    speaker.say('Warning, fall detected for person 1')
    speaker.stop()
"""

import threading
import time
from collections import deque

from config.settings import (
    SPEAKER_ENABLED, SPEAKER_RATE, SPEAKER_VOLUME, SPEAKER_QUEUE_SIZE
)


# ─── Optional pyttsx3 import ─────────────────────────────────────────────────

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False
    print('[Speaker] pyttsx3 not found — TTS disabled. '
          'Install: pip install pyttsx3 && sudo apt-get install espeak')


class TTSSpeaker(threading.Thread):
    """
    Background TTS speaker thread.

    Drains a bounded message queue and speaks each message via espeak.
    All speech runs off the main thread so inference is never blocked.

    Usage:
        speaker = TTSSpeaker()
        speaker.start()
        speaker.say('Alert — fall detected')
        speaker.stop()
    """

    def __init__(self):
        super().__init__(daemon=True, name='TTSSpeaker')
        self._queue = deque(maxlen=SPEAKER_QUEUE_SIZE)
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._engine = None

        if _TTS_AVAILABLE and SPEAKER_ENABLED:
            try:
                self._engine = pyttsx3.init('espeak')
                # Set English voice explicitly for espeak-ng on Pi
                voices = self._engine.getProperty('voices')
                for v in voices:
                    if 'en' in v.id.lower() or 'english' in v.name.lower():
                        self._engine.setProperty('voice', v.id)
                        break
                self._engine.setProperty('rate',   SPEAKER_RATE)
                self._engine.setProperty('volume', SPEAKER_VOLUME)
                print(f'[Speaker] TTS ready — rate={SPEAKER_RATE}, '
                      f'volume={SPEAKER_VOLUME}')
            except Exception as e:
                print(f'[Speaker] TTS init failed: {e}')
                self._engine = None

    # ── Public API ────────────────────────────────────────────────────────────

    def say(self, text: str):
        """
        Queue a message for TTS playback.
        If queue is full the oldest message is dropped (bounded deque behaviour).
        No-op if speaker is disabled or unavailable.
        """
        if not SPEAKER_ENABLED or self._engine is None:
            return
        with self._lock:
            self._queue.append(text)

    # ── Convenience alert methods ─────────────────────────────────────────────

    def announce_fall(self, person_id: int):
        self.say(f'Warning. Fall detected for person {person_id}. '
                 f'Please check immediately.')

    def announce_fight(self, pair_id: int = None):
        if pair_id is not None:
            self.say(f'Warning. Fight detected between two people. '
                     f'Please check immediately.')
        else:
            self.say('Warning. Fight detected. Please check immediately.')

    def announce_motionless(self, person_id: int):
        self.say(f'Alert. Person {person_id} is motionless. '
                 f'Urgent attention required.')

    def announce_child_missing(self):
        self.say('Alert. The child has not been seen by either camera '
                 'for over one minute. Please check on the child immediately.')

    def announce_impact(self):
        self.say('Impact sound detected. Please check the area.')

    def announce_temp(self, temp_c: float, humidity: float):
        temp_f = (temp_c * 9 / 5) + 32
        self.say(f'Current temperature is {temp_c:.0f} degrees Celsius, '
                 f'{temp_f:.0f} degrees Fahrenheit. '
                 f'Humidity is {humidity:.0f} percent.')

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self):
        if not _TTS_AVAILABLE or not SPEAKER_ENABLED or self._engine is None:
            return
        while not self._stopped.is_set():
            msg = None
            with self._lock:
                if self._queue:
                    msg = self._queue.popleft()
            if msg:
                self._speak(msg)
            else:
                time.sleep(0.1)

    def _speak(self, text: str):
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print(f'[Speaker] TTS error: {e}')
            # Reinitialise engine on failure
            try:
                self._engine = pyttsx3.init('espeak')
                # Set English voice explicitly for espeak-ng on Pi
                voices = self._engine.getProperty('voices')
                for v in voices:
                    if 'en' in v.id.lower() or 'english' in v.name.lower():
                        self._engine.setProperty('voice', v.id)
                        break
                self._engine.setProperty('rate',   SPEAKER_RATE)
                self._engine.setProperty('volume', SPEAKER_VOLUME)
            except Exception:
                pass

    def stop(self):
        self._stopped.set()
        try:
            if self._engine:
                self._engine.stop()
        except Exception:
            pass
        print('[Speaker] Stopped.')
