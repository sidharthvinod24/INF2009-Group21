"""
audio/yamnet_thread.py
=======================
YAMNetThread — runs Google YAMNet neural inference in a dedicated thread.

Why a separate thread?
    YAMNet inference on a Pi takes ~100-300ms per call. Running it directly
    inside the mic read loop would stall audio capture and cause buffer
    overflows. This thread decouples inference from capture entirely.

Queue design:
    Bounded deque (maxlen=2). If YAMNet can't keep up, old chunks are
    silently dropped rather than building a backlog. The heuristic detectors
    in AudioDetectorThread provide real-time coverage in the meantime.

YAMNet class map:
    Downloaded once from GitHub on startup in a daemon thread.
    Falls back to hardcoded index ranges if the download fails.

Requires:
    pip install tensorflow tensorflow-hub
"""

import threading
import time
import csv
import io
import urllib.request
import numpy as np
from collections import deque

from config.settings import YAMNET_CONF_THRESH, YAMNET_EVERY_N_CHUNKS


# ─── Optional TF import ───────────────────────────────────────────────────────

try:
    import tensorflow_hub as hub
    _YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
    YAMNET_OK     = True
    print('[Audio] YAMNet loaded.')
except Exception as e:
    YAMNET_OK     = False
    _YAMNET_MODEL = None
    print(f'[Audio] YAMNet unavailable ({e}) — heuristic-only mode.')


# ─── Keyword → class index mapping ───────────────────────────────────────────

_YAMNET_KW = {
    'scream': {'scream', 'shriek', 'shout', 'yell', 'squeal', 'wail'},
    'cry':    {'crying', 'cry', 'whimper', 'sob', 'wail', 'infant', 'baby'},
    'laugh':  {'laugh', 'giggle', 'chuckle', 'chortle'},
    'impact': {'bang', 'thud', 'crash', 'impact', 'thump', 'knock', 'slam',
               'explosion', 'smash', 'clatter', 'crack'},
}

# Populated by _build_yamnet_indices() at startup
YAMNET_IDX: dict = {k: set() for k in _YAMNET_KW}


def _build_yamnet_indices():
    """
    Download the YAMNet class map CSV and map label keywords to class indices.
    Runs in a daemon thread at startup so it doesn't block the main thread.
    Falls back to hardcoded ranges if the download fails.
    """
    global YAMNET_IDX
    try:
        url = ('https://raw.githubusercontent.com/tensorflow/models/master'
               '/research/audioset/yamnet/yamnet_class_map.csv')
        with urllib.request.urlopen(url, timeout=8) as r:
            text = r.read().decode()
        for row in csv.DictReader(io.StringIO(text)):
            idx  = int(row['index'])
            name = row['display_name'].lower()
            for label, kws in _YAMNET_KW.items():
                if any(kw in name for kw in kws):
                    YAMNET_IDX[label].add(idx)
        print('[Audio] YAMNet indices: ' +
              ', '.join(f'{k}={len(v)}' for k, v in YAMNET_IDX.items()))
    except Exception as e:
        print(f'[Audio] YAMNet class map download failed ({e}) — using fallback ranges.')
        YAMNET_IDX = {
            'scream': set(range(20, 40)),
            'cry':    set(range(20, 35)),
            'laugh':  set(range(56, 65)),
            'impact': set(range(461, 480)),
        }


# Start index build in background immediately on import
if YAMNET_OK:
    threading.Thread(target=_build_yamnet_indices, daemon=True).start()


# ─── YAMNet thread ────────────────────────────────────────────────────────────

class YAMNetThread(threading.Thread):
    """
    Consumes batched audio chunks from a bounded queue and runs YAMNet.
    Confirmed detections are written to AudioState.

    Usage:
        yt = YAMNetThread(state=audio_state)
        yt.start()
        yt.enqueue(np.array(...))   # called by AudioDetectorThread
        yt.stop()
    """

    def __init__(self, state):
        super().__init__(daemon=True, name='YAMNet')
        self.state    = state
        self._queue   = deque(maxlen=2)   # drop old if YAMNet falls behind
        self._lock    = threading.Lock()
        self._stopped = threading.Event()

    def enqueue(self, audio: np.ndarray):
        """Called by AudioDetectorThread every YAMNET_EVERY_N_CHUNKS chunks."""
        with self._lock:
            self._queue.append(audio)

    def run(self):
        while not self._stopped.is_set():
            job = None
            with self._lock:
                if self._queue:
                    job = self._queue.popleft()
            if job is None:
                time.sleep(0.02)
                continue
            self._infer(job)

    def _infer(self, audio: np.ndarray):
        if not YAMNET_OK or _YAMNET_MODEL is None:
            return
        try:
            scores, _, _ = _YAMNET_MODEL(audio)
            mean_scores  = np.mean(scores.numpy(), axis=0)
            for label, idx_set in YAMNET_IDX.items():
                if not idx_set:
                    continue
                top = max(
                    (mean_scores[i] for i in idx_set if i < len(mean_scores)),
                    default=0.0)
                if top >= YAMNET_CONF_THRESH:
                    self.state.record(label)
                    print(f'[Audio/YAMNet] {label.upper()} score={top:.2f}')
        except Exception:
            pass   # YAMNet errors are non-fatal

    def stop(self):
        self._stopped.set()
