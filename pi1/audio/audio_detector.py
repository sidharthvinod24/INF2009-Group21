"""
audio/audio_detector.py
========================
AudioDetectorThread — continuous mic capture + fast heuristic detectors.

Runs independently of the camera — Pi 1 has no PIR so both mic and camera
run continuously from startup.

Detection pipeline (per chunk ~64ms):
    1. Impact   — short RMS spike that drops quickly
    2. Laugh    — sustained spectral flux in 500-2000 Hz band
    3. Cry      — sustained spectral flux in 350-900 Hz band
    4. Scream   — loud + sustained high-frequency flux 900-4000 Hz

Every YAMNET_EVERY_N_CHUNKS chunks, the accumulated audio is forwarded to
YAMNetThread for neural-net verification (runs off the hot mic path).

Cross-modal interactions:
    - Laugh suppresses cry detection (laughing ≠ crying)
    - Laugh/cry/impact suppress scream detection (avoid double-counting)

Standalone Telegram alerts:
    - impact  → sends alert (AUDIO_ALERT_IMPACT = True)
    - scream  → enriches video alerts only (AUDIO_ALERT_SCREAM = False)
    - cry     → enriches video alerts only (AUDIO_ALERT_CRY = False)
    - laugh   → never alerts (used only for fight suppression)

Requires:
    pip install pyaudio
"""

import io
import math
import threading
import time
import numpy as np
from collections import deque

from config.settings import (
    AUDIO_DEVICE_INDEX, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE, AUDIO_CHANNELS,
    AUDIO_COOLDOWN_SEC, AUDIO_ALERT_COOLDOWN,
    AUDIO_ALERT_SCREAM, AUDIO_ALERT_CRY, AUDIO_ALERT_LAUGH, AUDIO_ALERT_IMPACT,
    IMPACT_RMS_THRESH, IMPACT_DURATION_MS, IMPACT_DROP_MS,
    SCREAM_RMS_THRESH, SCREAM_LOW_HZ, SCREAM_HIGH_HZ,
    SCREAM_FLUX_THRESH, SCREAM_DURATION_MS,
    CRY_LOW_HZ, CRY_HIGH_HZ, CRY_FLUX_THRESH, CRY_DURATION_MS,
    LAUGH_LOW_HZ, LAUGH_HIGH_HZ, LAUGH_RMS_THRESH, LAUGH_FLUX_THRESH, LAUGH_DURATION_MS,
    YAMNET_EVERY_N_CHUNKS,
)
from audio.yamnet_thread import YAMNET_OK


# ─── Optional pyaudio import ─────────────────────────────────────────────────

try:
    import pyaudio
    PYAUDIO_OK = True
except ImportError:
    PYAUDIO_OK = False
    print('[Audio] pyaudio not found — audio detection disabled. '
          'Install: pip install pyaudio')


# ─── Spectral flux helper ─────────────────────────────────────────────────────

def _bandpass_flux(chunk: np.ndarray, sr: int,
                   low_hz: float, high_hz: float,
                   prev_spec) -> tuple:
    """
    Compute one-sided spectral flux in the given frequency band.

    Returns (flux, new_spectrum).
    flux > threshold means energy in this band is rising — indicates
    a transient event (cry, laugh, scream onset).
    """
    n        = len(chunk)
    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(n)))
    freqs    = np.fft.rfftfreq(n, d=1.0 / sr)
    band     = spectrum[(freqs >= low_hz) & (freqs <= high_hz)]

    if prev_spec is None or len(prev_spec) != len(band):
        return 0.0, band

    diff = band - prev_spec
    flux = float(np.sum(diff[diff > 0])) / (float(np.sum(band)) + 1e-6)
    return flux, band


# ─── Audio detector thread ────────────────────────────────────────────────────

class AudioDetectorThread(threading.Thread):
    """
    Opens the system microphone and processes audio continuously.

    Heuristic detectors run on every chunk (~64ms).
    YAMNet inference is triggered every YAMNET_EVERY_N_CHUNKS chunks.
    Detections are written to AudioState (shared with InferenceThread).
    Standalone Telegram alerts fire for impact events.

    Usage:
        adt = AudioDetectorThread(telegram=t, yamnet_thread=yt, state=s)
        adt.start()
        adt.stop()
    """

    _ALERT_FLAGS = {
        'scream': AUDIO_ALERT_SCREAM,
        'cry':    AUDIO_ALERT_CRY,
        'laugh':  AUDIO_ALERT_LAUGH,
        'impact': AUDIO_ALERT_IMPACT,
    }
    _EMOJI = {
        'scream': '😱',
        'cry':    '😢',
        'laugh':  '😄',
        'impact': '💥',
    }

    def __init__(self, telegram, yamnet_thread, state,
                 device_index=None):
        super().__init__(daemon=True, name='AudioDetector')
        self.telegram     = telegram
        self.yamnet       = yamnet_thread
        self.state        = state
        self.device_index = device_index if device_index is not None else AUDIO_DEVICE_INDEX

        self._running     = False
        self._chunk_ms    = (AUDIO_CHUNK_SIZE / AUDIO_SAMPLE_RATE) * 1000.0
        self._chunk_n     = 0
        self._yamnet_buf  = deque(maxlen=YAMNET_EVERY_N_CHUNKS)

        # Heuristic accumulators
        self._scream_count    = 0
        self._cry_count       = 0
        self._laugh_count     = 0
        self._impact_count    = 0
        self._impact_peak_t   = 0.0

        # Previous spectra for flux computation
        self._prev_cry_spec    = None
        self._prev_laugh_spec  = None
        self._prev_scream_spec = None

        # Per-label standalone Telegram alert cooldowns
        self._last_alerted = {l: 0.0 for l in ('scream', 'cry', 'laugh', 'impact')}
        self._alert_lock   = threading.Lock()

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self):
        if not PYAUDIO_OK:
            print('[Audio] pyaudio not available — AudioDetectorThread exiting.')
            return

        self._running = True
        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=AUDIO_CHUNK_SIZE,
            )
            print(f'[Audio] Mic open — {AUDIO_SAMPLE_RATE} Hz mono. Listening...')
            while self._running:
                raw     = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                samples = (np.frombuffer(raw, dtype=np.int16)
                           .astype(np.float32) / 32768.0)
                self._process(samples)
        except Exception as exc:
            print(f'[Audio] Stream error: {exc}')
        finally:
            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            pa.terminate()
            print('[Audio] Mic closed.')

    def stop(self):
        self._running = False

    # ── Per-chunk processing ──────────────────────────────────────────────────

    def _process(self, samples: np.ndarray):
        self._chunk_n += 1
        rms = float(np.sqrt(np.mean(samples ** 2)))
        now = time.time()

        # ── 1. Impact ─────────────────────────────────────────────────────────
        # Short RMS spike (high energy) that drops off quickly
        if rms > IMPACT_RMS_THRESH:
            self._impact_count += 1
            if self._impact_peak_t == 0.0:
                self._impact_peak_t = now
        else:
            if (self._impact_count >= math.ceil(IMPACT_DURATION_MS / self._chunk_ms)
                    and self._impact_peak_t > 0.0
                    and (now - self._impact_peak_t) < (IMPACT_DROP_MS / 1000.0)):
                self._confirm('impact', 'heuristic')
            self._impact_count  = 0
            self._impact_peak_t = 0.0

        # ── 2. Laugh ──────────────────────────────────────────────────────────
        # Sustained rising flux in 500-2000 Hz
        flux_l, self._prev_laugh_spec = _bandpass_flux(
            samples, AUDIO_SAMPLE_RATE,
            LAUGH_LOW_HZ, LAUGH_HIGH_HZ,
            self._prev_laugh_spec)
        self._laugh_count = (
            self._laugh_count + 1
            if (flux_l > LAUGH_FLUX_THRESH and rms > LAUGH_RMS_THRESH)
            else max(0, self._laugh_count - 1))
        if self._laugh_count >= math.ceil(LAUGH_DURATION_MS / self._chunk_ms):
            self._confirm('laugh', 'heuristic')
            self._laugh_count = 0

        # ── 3. Cry ────────────────────────────────────────────────────────────
        # Sustained rising flux in 350-900 Hz, suppressed if laugh recent
        flux_c, self._prev_cry_spec = _bandpass_flux(
            samples, AUDIO_SAMPLE_RATE,
            CRY_LOW_HZ, CRY_HIGH_HZ,
            self._prev_cry_spec)
        laugh_recent      = self.state.recent('laugh', 1.0)
        self._cry_count   = (
            self._cry_count + 1
            if (flux_c > CRY_FLUX_THRESH and not laugh_recent)
            else max(0, self._cry_count - 1))
        if self._cry_count >= math.ceil(CRY_DURATION_MS / self._chunk_ms):
            self._confirm('cry', 'heuristic')
            self._cry_count = 0

        # ── 4. Scream ─────────────────────────────────────────────────────────
        # Loud + sustained rising flux in 900-4000 Hz
        # Suppressed if cry/laugh/impact recently detected (avoid double-counting)
        flux_s, self._prev_scream_spec = _bandpass_flux(
            samples, AUDIO_SAMPLE_RATE,
            SCREAM_LOW_HZ, SCREAM_HIGH_HZ,
            self._prev_scream_spec)
        suppressed = (
            self.state.recent('cry',    2.0) or
            self.state.recent('laugh',  2.0) or
            self.state.recent('impact', 2.0))
        if rms > SCREAM_RMS_THRESH and flux_s > SCREAM_FLUX_THRESH and not suppressed:
            self._scream_count += 1
        else:
            self._scream_count  = 0
        if self._scream_count >= math.ceil(SCREAM_DURATION_MS / self._chunk_ms):
            self._confirm('scream', 'heuristic')
            self._scream_count  = 0

        # ── YAMNet ────────────────────────────────────────────────────────────
        # Accumulate chunks and forward to YAMNetThread every N chunks
        self._yamnet_buf.append(samples)
        if YAMNET_OK and self._chunk_n % YAMNET_EVERY_N_CHUNKS == 0:
            self.yamnet.enqueue(np.concatenate(list(self._yamnet_buf)))

    # ── Confirm detection ─────────────────────────────────────────────────────

    def _confirm(self, label: str, source: str):
        """
        Record detection in AudioState (with per-label cooldown).
        Optionally fire a standalone Telegram alert for impact events.
        Scream/cry only enrich video alerts — they never send alone.
        """
        # Per-label detection cooldown — avoids flooding AudioState
        if not self.state.recent(label, AUDIO_COOLDOWN_SEC):
            self.state.record(label)
            print(f'[Audio/{source}] {label.upper()} detected')

        # Standalone Telegram alert — only for labels where flag is True
        if not self._ALERT_FLAGS.get(label, False):
            return

        now = time.time()
        with self._alert_lock:
            if now - self._last_alerted[label] < AUDIO_ALERT_COOLDOWN:
                return
            self._last_alerted[label] = now

        msg = (f'{self._EMOJI.get(label, "🔊")} {label.upper()} DETECTED (audio)\n'
               f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        self.telegram.send_photo(msg, _blank_jpeg())
        print(f'[Audio] Telegram alert: {label.upper()}')


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _blank_jpeg() -> bytes:
    """
    1×1 black JPEG — used as placeholder for audio-only Telegram alerts
    where there is no camera frame to attach.
    """
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new('RGB', (1, 1)).save(buf, format='JPEG')
        return buf.getvalue()
    except Exception:
        # Minimal valid JPEG bytes as fallback
        return (b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
                b'\x00\x00\xff\xd9')
