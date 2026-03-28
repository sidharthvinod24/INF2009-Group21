"""
camera/pir_thread.py
=====================
PIRWatchThread — monitors the PIR motion sensor on GPIO 26.

When motion is detected:
  - Records the trigger time in shared state (_pir_last_trigger_time)
  - Sets _wake_requested = True if the camera is currently paused
    and the rearm cooldown has elapsed

InferenceThread checks _wake_requested every second and calls
camera.resume() when it is True.

Graceful fallback:
  If gpiozero is not installed or not running on a Pi,
  MotionSensor will be None and PIRWatchThread becomes a no-op.
  This lets the code run on a dev machine for testing.

Shared state (all protected by _pir_lock):
  _pir_last_trigger_time  — epoch of last PIR trigger
  _wake_requested         — InferenceThread should resume camera
  _camera_active          — True when camera is running
  _last_camera_off_time   — epoch when camera was last paused
  _camera_last_person_time — epoch when YOLO last saw a person
"""

import threading
import time

from config.settings import (
    ENABLE_PIR, PIR_GPIO_PIN, PIR_REARM_COOLDOWN_SEC
)

try:
    from gpiozero import MotionSensor
except Exception:
    MotionSensor = None


# ─── Shared PIR state ─────────────────────────────────────────────────────────
# Written by PIRWatchThread, read/written by InferenceThread.

pir_lock               = threading.Lock()
pir_last_trigger_time  = 0.0
wake_requested         = False
camera_active          = False
last_camera_off_time   = 0.0
camera_last_person_time = 0.0


class PIRWatchThread(threading.Thread):
    """
    Watches the PIR sensor and sets wake_requested when motion is detected.
    Becomes a no-op if ENABLE_PIR is False or gpiozero is unavailable.

    Usage:
        pir = PIRWatchThread()
        pir.start()
        pir.stop()
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._stopped = threading.Event()

    def run(self):
        global pir_last_trigger_time, wake_requested

        if not ENABLE_PIR or MotionSensor is None:
            if not ENABLE_PIR:
                print('[PIR] PIR disabled in config — camera will run continuously.')
            else:
                print('[PIR] gpiozero not available — PIR disabled.')
            return

        print(f'[PIR] Watching GPIO pin {PIR_GPIO_PIN}')
        pir = MotionSensor(PIR_GPIO_PIN)

        while not self._stopped.is_set():
            pir.wait_for_motion()
            now = time.time()

            with pir_lock:
                pir_last_trigger_time = now
                active   = camera_active
                cool_ok  = (now - last_camera_off_time) >= PIR_REARM_COOLDOWN_SEC

            if not active and cool_ok:
                with pir_lock:
                    wake_requested = True
                print('[PIR] Motion detected — requesting camera wake')

            pir.wait_for_no_motion()

    def stop(self):
        self._stopped.set()
