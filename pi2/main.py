"""
main.py — Pi 2 Monitor Entry Point
====================================
Multi-Pi Home Monitoring System — Pi 2 Node

Responsibilities of Pi 2:
  - PIR sensor on GPIO 26 — camera wakes on motion, sleeps after idle
  - USB camera — paused by default, PIR wakes it
  - YOLOv8n-pose inference for fight + fall detection
  - Audio detection (YAMNet + heuristics) for scream/cry/laugh/impact
  - Audio cross-modal logic:
      - Fight + laugh  → alert suppressed (kids playing)
      - Fight/Fall + cry/scream → alert enriched
  - ChildWatchdogThread — polls Pi 1 every 10s
      If BOTH cameras see nobody for 60s → Telegram child-missing alert
  - Serve live MJPEG stream at http://<pi2-ip>:5000/
  - Send Telegram alerts

Usage:
    export TELEGRAM_TOKEN=<token>
    export TELEGRAM_CHAT_ID=<chat_id>
    export PI1_STATUS_URL=http://10.108.206.188:5001/person_status
    python main.py
"""

import signal
import sys
import threading

from ultralytics import YOLO

from config.settings           import MODEL_PATH, WEB_PORT, ENABLE_PIR
from camera.camera_thread      import CameraThread
from camera.pir_thread         import PIRWatchThread, pir_lock, camera_active
import camera.pir_thread as pir_state
from detection.person_registry import PersonRegistry
from alerting.telegram_alert   import TelegramThread
from audio.audio_state         import audio_state
from audio.yamnet_thread       import YAMNetThread, YAMNET_OK
from audio.audio_detector      import AudioDetectorThread, PYAUDIO_OK
from inference_thread          import InferenceThread
from watchdog.child_watchdog   import ChildWatchdogThread
from server.web_server         import start_server


# ─── Graceful shutdown ────────────────────────────────────────────────────────

_shutdown = threading.Event()


def _handle_signal(sig, frame):
    print(f'\n[Main] Signal {sig} received — shutting down...')
    _shutdown.set()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # 1. Load model
    print(f'[Main] Loading pose model: {MODEL_PATH}')
    model = YOLO(MODEL_PATH)

    # 2. Person registry
    registry = PersonRegistry()

    # 3. Telegram alert thread
    telegram = TelegramThread()
    telegram.start()

    # 4. Audio threads
    yamnet_thread = YAMNetThread(state=audio_state)
    yamnet_thread.start()

    audio_thread = AudioDetectorThread(
        telegram=telegram,
        yamnet_thread=yamnet_thread,
        state=audio_state,
    )
    if PYAUDIO_OK:
        audio_thread.start()
        print('[Main] Audio detection started.')
    else:
        print('[Main] pyaudio not installed — audio detection disabled.')

    if not YAMNET_OK:
        print('[Main] YAMNet unavailable — heuristic-only audio mode.')

    # 5. Camera thread
    cam = CameraThread()
    cam.start()

    # 6. PIR thread — starts paused if PIR is enabled
    pir = PIRWatchThread()
    pir.start()

    if ENABLE_PIR:
        cam.pause()
        with pir_state.pir_lock:
            pir_state.camera_active = False
        print('[Main] Camera started PAUSED — waiting for PIR trigger.')
    else:
        with pir_state.pir_lock:
            pir_state.camera_active = True
        print('[Main] PIR disabled — camera running continuously.')

    # 7. Inference thread
    inference = InferenceThread(
        camera=cam,
        model=model,
        registry=registry,
        telegram=telegram,
        audio_state=audio_state,
    )
    inference.start()

    # 8. Child watchdog thread
    watchdog = ChildWatchdogThread(telegram=telegram)
    watchdog.start()

    # 9. Flask server
    start_server(WEB_PORT)

    print(f'\n[Main] Pi 2 Monitor running.')
    print(f'       Stream:   http://0.0.0.0:{WEB_PORT}/')
    print(f'       PIR:      {"Enabled (GPIO 26)" if ENABLE_PIR else "Disabled"}')
    print(f'       Audio:    YAMNet={YAMNET_OK}  pyaudio={PYAUDIO_OK}')
    print(f'       Watchdog: polling Pi 1 for child presence')
    print('       Press Ctrl+C to quit.\n')

    _shutdown.wait()

    print('[Main] Stopping threads...')
    inference.stop()
    pir.stop()
    watchdog.stop()
    audio_thread.stop()
    yamnet_thread.stop()
    telegram.stop()
    cam.stop()
    print('[Main] Done.')
    sys.exit(0)


if __name__ == '__main__':
    main()
