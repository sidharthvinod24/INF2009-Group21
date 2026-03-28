"""
alerting/telegram_alert.py
===========================
Background Telegram alert thread for Pi 1.

Responsibilities:
  - Sends text + photo or text + GIF (animation) alerts to a Telegram chat.
  - All network calls happen in a background thread — inference is never blocked.
  - GIF building is offloaded to a throwaway thread with a semaphore (max 1 at a time).
  - Bounded send queue (maxlen=3) drops old alerts if the queue backs up.
  - Exponential back-off after consecutive failures; session is recycled on error.

Public API:
    t = TelegramThread()
    t.start()
    t.send(message_text, jpeg_bytes)               # photo alert
    t.send(message_text, jpeg_bytes, gif_frames)   # GIF alert
    t.stop()
"""

import time
import threading
import requests
from collections import deque

from config.settings import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    ALERT_CLIP_SECONDS, ALERT_CLIP_FPS, ALERT_CLIP_SCALE,
)


# ─── GIF builder ─────────────────────────────────────────────────────────────

def build_gif_from_jpegs(jpeg_list):
    """
    Stitch a list of raw JPEG bytes into an animated GIF.
    Runs in a throwaway thread to keep the main send queue non-blocking.
    Returns GIF bytes, or None on failure.
    """
    import io
    try:
        from PIL import Image
    except ImportError:
        return None

    if not jpeg_list or len(jpeg_list) < 2:
        return None

    MAX_GIF_FRAMES = int(ALERT_CLIP_SECONDS * ALERT_CLIP_FPS)
    if len(jpeg_list) > MAX_GIF_FRAMES:
        step      = len(jpeg_list) / MAX_GIF_FRAMES
        jpeg_list = [jpeg_list[int(i * step)] for i in range(MAX_GIF_FRAMES)]

    pil_frames = []
    for jb in jpeg_list:
        try:
            img = Image.open(io.BytesIO(jb))
            if ALERT_CLIP_SCALE != 1.0:
                nw  = int(img.width  * ALERT_CLIP_SCALE)
                nh  = int(img.height * ALERT_CLIP_SCALE)
                img = img.resize((nw, nh), Image.NEAREST)   # fastest resize
            pil_frames.append(img.convert('RGB').quantize(colors=64, method=0))
        except Exception:
            continue

    if len(pil_frames) < 2:
        return None

    import io as _io
    buf = _io.BytesIO()
    dur = int(1000 / ALERT_CLIP_FPS)
    pil_frames[0].save(
        buf, format='GIF', save_all=True,
        append_images=pil_frames[1:],
        duration=dur, loop=0,
    )
    buf.seek(0)
    result = buf.getvalue()
    for f in pil_frames:
        f.close()
    del pil_frames
    return result


# ─── Telegram send thread ─────────────────────────────────────────────────────

class TelegramThread(threading.Thread):
    """
    Background thread that drains a bounded alert queue and sends
    messages to the configured Telegram chat.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._lock           = threading.Lock()
        self._queue          = deque(maxlen=3)          # bounded — drop old on overflow
        self._stopped        = threading.Event()
        self._consec_fails   = 0
        self._last_fail_time = 0.0
        self._session        = None
        self._gif_build_sem  = threading.Semaphore(1)   # max 1 GIF build at a time

    # ── Public send API ───────────────────────────────────────────────────────

    def send(self, msg, jpeg_bytes, gif_frames=None):
        """Queue an alert.  Pass gif_frames to send an animated GIF instead of a photo."""
        if gif_frames:
            self.send_gif(msg, jpeg_bytes, gif_frames)
        else:
            self.send_photo(msg, jpeg_bytes)

    def send_photo(self, msg, jpeg_bytes):
        with self._lock:
            self._queue.append(('photo', msg, jpeg_bytes))

    def send_gif(self, msg, jpeg_bytes, raw_jpeg_frames):
        """Build GIF in a throwaway thread (non-blocking); fall back to photo if busy."""
        if not self._gif_build_sem.acquire(blocking=False):
            self.send_photo(msg, jpeg_bytes)
            return

        def _build_and_queue():
            try:
                gif_data = build_gif_from_jpegs(raw_jpeg_frames)
                with self._lock:
                    if gif_data:
                        self._queue.append(('gif', msg, gif_data))
                    else:
                        self._queue.append(('photo', msg, jpeg_bytes))
            finally:
                self._gif_build_sem.release()

        threading.Thread(target=_build_and_queue, daemon=True).start()

    # ── Thread body ───────────────────────────────────────────────────────────

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
        base = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}'

        # Back off after repeated failures
        if self._consec_fails >= 3:
            backoff = min(60, 10 * self._consec_fails)
            if time.time() - self._last_fail_time < backoff:
                return
            print(f'[Telegram] Retrying after {backoff}s back-off...')

        session = self._get_session()
        ok      = True

        try:
            session.post(f'{base}/sendMessage',
                         data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg},
                         timeout=8)
        except Exception:
            pass  # text failure is non-critical

        try:
            if kind == 'gif':
                session.post(f'{base}/sendAnimation',
                             data={'chat_id': TELEGRAM_CHAT_ID},
                             files={'animation': ('alert.gif', data, 'image/gif')},
                             timeout=15)
            else:
                session.post(f'{base}/sendPhoto',
                             data={'chat_id': TELEGRAM_CHAT_ID},
                             files={'photo': ('snap.jpg', data, 'image/jpeg')},
                             timeout=10)
        except Exception as e:
            ok = False
            print(f'[Telegram] Send failed: {type(e).__name__}')

        if ok:
            self._consec_fails = 0
        else:
            self._consec_fails += 1
            self._last_fail_time = time.time()
            if self._consec_fails >= 3:
                print(f'[Telegram] {self._consec_fails} consecutive failures — recycling session')
                self._close_session()

    def _get_session(self):
        if self._session is None:
            s       = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=1, pool_maxsize=2, max_retries=1)
            s.mount('https://', adapter)
            self._session = s
        return self._session

    def _close_session(self):
        try:
            if self._session:
                self._session.close()
        except Exception:
            pass
        self._session = None

    def stop(self):
        self._stopped.set()
        self._close_session()
        print('[Telegram] Thread stopped.')
