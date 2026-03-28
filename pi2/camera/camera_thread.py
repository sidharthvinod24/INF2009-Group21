"""
camera/camera_thread.py
========================
USB camera capture thread for Pi 2.

Pi 2 supports PIR-driven pause/resume:
  - Camera starts PAUSED when PIR is enabled
  - PIRWatchThread calls resume() when motion is detected
  - InferenceThread calls pause() when no person seen for NO_PERSON_IDLE_SEC

Unlike Pi 1, this thread raises RuntimeError immediately if the camera
cannot be opened (no auto-reconnect) since Pi 2 is expected to be a
stable indoor deployment.
"""

import cv2
import time
import threading

from config.settings import (
    CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
)


class CameraThread(threading.Thread):
    """
    Reads frames from a USB camera.
    Supports pause() / resume() for PIR-driven power management.

    Usage:
        cam = CameraThread()
        cam.start()
        ok, frame = cam.read()
        cam.pause()   # PIR sleep
        cam.resume()  # PIR wake
        cam.stop()
    """

    def __init__(self, src=None):
        super().__init__(daemon=True)
        self._src     = src if src is not None else CAMERA_SOURCE
        self._lock    = threading.Lock()
        self._frame   = None
        self._stopped = threading.Event()
        self._paused  = threading.Event()

        print(f'[Camera] Opening USB camera index {self._src}')
        self.cap = cv2.VideoCapture(self._src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not self.cap.isOpened():
            raise RuntimeError(f'[Camera] Cannot open USB camera (index {self._src})')

        actual_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f'[Camera] Opened — {actual_w}x{actual_h} @ {actual_fps:.0f} fps')

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self):
        while not self._stopped.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self._stopped.set()
                break
            with self._lock:
                self._frame = frame

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self):
        """Return (True, frame_copy) or (False, None) if no frame yet."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def pause(self):
        """Pause frame capture (PIR sleep mode)."""
        self._paused.set()

    def resume(self):
        """Resume frame capture (PIR wake)."""
        with self._lock:
            self._frame = None   # discard stale frame
        self._paused.clear()

    def stop(self):
        self._stopped.set()
        self.cap.release()
        print('[Camera] Stopped.')

    @property
    def is_paused(self):
        return self._paused.is_set()
