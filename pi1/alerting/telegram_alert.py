"""
alerting/telegram_alert.py
===========================
TelegramThread — sends alerts via plain requests.post calls.
No Session, no HTTPAdapter, no connection pooling.
Python 3.13 compatible.
"""

import time
import threading
import io
import requests
from collections import deque

from config.settings import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    ALERT_CLIP_SECONDS, ALERT_CLIP_FPS, ALERT_CLIP_SCALE,
)

BASE_URL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}'


def _tg_post(endpoint, data=None, files=None, timeout=10):
    """Plain requests.post with no session — Python 3.13 safe."""
    try:
        requests.post(f'{BASE_URL}/{endpoint}', data=data, files=files, timeout=timeout)
    except Exception as e:
        print(f'[Telegram] Post failed ({endpoint}): {e}')


def build_gif_from_jpegs(jpeg_list):
    """Stitch JPEG bytes into an animated GIF. Returns GIF bytes or None."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if not jpeg_list or len(jpeg_list) < 2:
        return None
    MAX_FRAMES = int(ALERT_CLIP_SECONDS * ALERT_CLIP_FPS)
    if len(jpeg_list) > MAX_FRAMES:
        step = len(jpeg_list) / MAX_FRAMES
        jpeg_list = [jpeg_list[int(i * step)] for i in range(MAX_FRAMES)]
    frames = []
    for jb in jpeg_list:
        try:
            img = Image.open(io.BytesIO(jb))
            if ALERT_CLIP_SCALE != 1.0:
                nw = int(img.width  * ALERT_CLIP_SCALE)
                nh = int(img.height * ALERT_CLIP_SCALE)
                img = img.resize((nw, nh), Image.NEAREST)
            frames.append(img.convert('RGB').quantize(colors=64, method=0))
        except Exception:
            continue
    if len(frames) < 2:
        return None
    buf = io.BytesIO()
    frames[0].save(buf, format='GIF', save_all=True,
                   append_images=frames[1:],
                   duration=int(1000 / ALERT_CLIP_FPS), loop=0)
    buf.seek(0)
    result = buf.getvalue()
    for f in frames:
        f.close()
    return result


class TelegramThread(threading.Thread):
    """
    Background thread that drains a bounded alert queue and sends
    messages to Telegram using plain requests.post calls.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._lock          = threading.Lock()
        self._queue         = deque(maxlen=5)
        self._stopped       = threading.Event()
        self._gif_sem       = threading.Semaphore(1)

    def send_photo(self, msg, jpeg_bytes):
        with self._lock:
            self._queue.append(('photo', msg, jpeg_bytes))

    def send_gif(self, msg, jpeg_bytes, raw_frames):
        if not self._gif_sem.acquire(blocking=False):
            self.send_photo(msg, jpeg_bytes)
            return

        def _build():
            try:
                gif = build_gif_from_jpegs(raw_frames)
                with self._lock:
                    if gif:
                        self._queue.append(('gif', msg, gif))
                    else:
                        self._queue.append(('photo', msg, jpeg_bytes))
            finally:
                self._gif_sem.release()

        threading.Thread(target=_build, daemon=True).start()

    def send(self, msg, jpeg_bytes, gif_frames=None):
        if gif_frames:
            self.send_gif(msg, jpeg_bytes, gif_frames)
        else:
            self.send_photo(msg, jpeg_bytes)

    def run(self):
        while not self._stopped.is_set():
            job = None
            with self._lock:
                if self._queue:
                    job = self._queue.popleft()
            if job:
                self._dispatch(job)
            else:
                time.sleep(0.1)

    def _dispatch(self, job):
        kind, msg, data = job
        _tg_post('sendMessage',
                 data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg})
        if kind == 'gif':
            _tg_post('sendAnimation',
                     data={'chat_id': TELEGRAM_CHAT_ID},
                     files={'animation': ('alert.gif', data, 'image/gif')},
                     timeout=20)
        else:
            _tg_post('sendPhoto',
                     data={'chat_id': TELEGRAM_CHAT_ID},
                     files={'photo': ('snap.jpg', data, 'image/jpeg')})

    def stop(self):
        self._stopped.set()
