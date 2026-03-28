"""
camera/camera_thread.py
========================
USB camera capture thread for Pi 1.

Pi 1 uses a directly connected USB camera (cv2.VideoCapture index).
The camera runs CONTINUOUSLY — there is no PIR sensor on Pi 1 and no
pause/resume logic.  The thread writes the latest frame into a shared
slot protected by a lock; the InferenceThread reads from it.

Auto-reconnect: if the USB camera disconnects or returns a bad read,
the thread releases and re-opens the capture after a short back-off.
"""

import cv2
import time
import threading

from config.settings import (
    CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
)


class CameraThread(threading.Thread):
    """
    Continuously reads frames from a USB camera.

    Usage:
        cam = CameraThread()
        cam.start()
        ok, frame = cam.read()
        cam.stop()
    """

    def __init__(self, src=None):
        super().__init__(daemon=True)
        self._src     = src if src is not None else CAMERA_SOURCE
        self._lock    = threading.Lock()
        self._frame   = None
        self._stopped = threading.Event()
        self.cap      = self._open()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _open(self):
        """Open the USB camera and apply resolution / FPS hints."""
        print(f'[Camera] Opening USB camera index {self._src}')
        cap = cv2.VideoCapture(self._src)          # default backend (V4L2 on Linux)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)      # always get the latest frame

        if cap.isOpened():
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f'[Camera] Opened — {actual_w}x{actual_h} @ {actual_fps:.0f} fps')
        else:
            print('[Camera] WARNING: could not open USB camera — will retry')
        return cap

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self):
        consecutive_fails = 0
        while not self._stopped.is_set():
            ret, frame = self.cap.read()
            if not ret:
                consecutive_fails += 1
                wait = min(5, consecutive_fails)        # back-off up to 5 s
                print(f'[Camera] Bad read (attempt {consecutive_fails}) '
                      f'— reconnecting in {wait}s...')
                self.cap.release()
                time.sleep(wait)
                self.cap = self._open()
                continue

            consecutive_fails = 0
            with self._lock:
                self._frame = frame

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self):
        """Return (True, frame_copy) or (False, None) if no frame yet."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._stopped.set()
        self.cap.release()
        print('[Camera] Stopped.')

    @property
    def is_paused(self):
        """Always False — Pi 1 camera never pauses."""
        return False
