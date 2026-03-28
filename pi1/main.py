"""
main.py — Pi 1 Monitor Entry Point
====================================
Multi-Pi Home Monitoring System — Pi 1 Node

Responsibilities:
  - USB camera (always on, no PIR)
  - YOLO26n-pose fall + fight detection
  - Audio detection (YAMNet + heuristics)
  - DHT22 temperature sensor on GPIO 26
  - USB speaker TTS announcements
  - Telegram bot commands (/temp /status /snapshot /arm /disarm /uptime /help)
  - MJPEG stream + /person_status endpoint for Pi 2 watchdog
  - Telegram alerts

Usage:
    export TELEGRAM_TOKEN=<token>
    export TELEGRAM_CHAT_ID=<chat_id>
    python main.py
"""

import signal
import sys
import threading

from ultralytics import YOLO

from config.settings import MODEL_PATH, WEB_PORT
from camera.camera_thread import CameraThread
from detection.person_registry import PersonRegistry
from alerting.telegram_alert import TelegramThread
from alerting.telegram_bot import TelegramBotThread
from audio.audio_state import audio_state
from audio.yamnet_thread import YAMNetThread, YAMNET_OK
from audio.audio_detector import AudioDetectorThread, PYAUDIO_OK
from sensors.temp_sensor import TempSensor
from sensors.tts_speaker import TTSSpeaker
from inference_thread import InferenceThread, get_live_status
from server.web_server import start_server, get_latest_frame


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

    # 5. Temperature sensor (DHT22 on GPIO 26)
    temp_sensor = TempSensor()
    temp_sensor.start()

    # 6. USB speaker TTS
    speaker = TTSSpeaker()
    speaker.start()

    # 7. USB camera thread
    cam = CameraThread()
    cam.start()

    # 8. Inference thread
    inference = InferenceThread(
        camera=cam,
        model=model,
        registry=registry,
        telegram=telegram,
        audio_state=audio_state,
        speaker=speaker,
    )
    inference.start()

    # 9. Flask server
    start_server(WEB_PORT)

    # 10. Telegram bot command thread
    bot = TelegramBotThread(
        temp_sensor=temp_sensor,
        speaker=speaker,
        get_status_fn=get_live_status,
        get_frame_fn=get_latest_frame,
        get_clip_fn=inference.get_clip_frames_for_bot,
    )
    bot.start()

    print(f'\n[Main] Pi 1 Monitor running.')
    print(f'       Stream:        http://0.0.0.0:{WEB_PORT}/')
    print(f'       Person status: http://0.0.0.0:{WEB_PORT}/person_status')
    print(f'       Audio:         YAMNet={YAMNET_OK}  pyaudio={PYAUDIO_OK}')
    print(f'       Temp sensor:   GPIO 26 (DHT22)')
    print(f'       Speaker:       USB TTS')
    print(f'       Bot commands:  /help /temp /status /snapshot /arm /disarm /uptime')
    print('       Press Ctrl+C to quit.\n')

    _shutdown.wait()

    print('[Main] Stopping threads...')
    inference.stop()
    bot.stop()
    audio_thread.stop()
    yamnet_thread.stop()
    speaker.stop()
    temp_sensor.stop()
    telegram.stop()
    cam.stop()
    print('[Main] Done.')
    sys.exit(0)


if __name__ == '__main__':
    main()
