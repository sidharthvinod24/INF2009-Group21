"""
audio/audio_state.py
=====================
Shared, lock-protected store of the last detection time per audio label.

This is the ONLY object that crosses the thread boundary between
AudioDetectorThread (writer) and InferenceThread (reader).

InferenceThread reads it once per YOLO frame with a single lock acquisition —
no copies, no allocations, no blocking.

Labels: 'scream', 'cry', 'laugh', 'impact'
"""

import time
import threading


class AudioState:
    """
    Lightweight shared state for audio detection results.

    Usage:
        state = AudioState()

        # Writer (AudioDetectorThread):
        state.record('scream')

        # Reader (InferenceThread):
        if state.recent('scream', window=6.0):
            ...  # enrich alert message

        seconds = state.seconds_since('cry')
    """

    _ALL_LABELS = ('scream', 'cry', 'laugh', 'impact')

    def __init__(self):
        self._lock = threading.Lock()
        self._last: dict = {label: 0.0 for label in self._ALL_LABELS}

    def record(self, label: str):
        """Mark label as detected right now."""
        with self._lock:
            self._last[label] = time.time()

    def seconds_since(self, label: str) -> float:
        """Seconds since label was last detected. Returns inf if never detected."""
        with self._lock:
            t = self._last.get(label, 0.0)
        return (time.time() - t) if t > 0 else float('inf')

    def recent(self, label: str, window: float) -> bool:
        """True if label was detected within the last `window` seconds."""
        return self.seconds_since(label) <= window


# Module-level singleton — imported by InferenceThread and AudioDetectorThread
audio_state = AudioState()
