"""
watchdog/child_watchdog.py
===========================
ChildWatchdogThread — cross-camera child-missing alert.

Every CHILD_WATCHDOG_POLL_SEC seconds, polls Pi 1's /person_status endpoint.
Compares Pi 1's last-seen time with Pi 2's own last-seen time.
If BOTH cameras have seen nobody for >= CHILD_MISSING_SECONDS, sends a
Telegram warning.

Offline tolerance:
  If Pi 1 is unreachable for CHILD_PI1_OFFLINE_TOLERANCE consecutive polls,
  the watchdog suppresses the alert to avoid false alarms during Pi 1 reboots.

Pi 2 person-seen time:
  _pi2_last_seen is a module-level float updated by InferenceThread via
  update_pi2_seen(). Protected by a threading.Lock.
"""

import threading
import time
import requests
import io

from config.settings import (
    PI1_STATUS_URL,
    CHILD_MISSING_SECONDS,
    CHILD_WATCHDOG_POLL_SEC,
    CHILD_ALERT_COOLDOWN,
    CHILD_PI1_OFFLINE_TOLERANCE,
)


# ─── Pi 2 shared person-seen state ───────────────────────────────────────────
# Written by InferenceThread via update_pi2_seen(), read by ChildWatchdogThread.

_pi2_lock      = threading.Lock()
_pi2_last_seen = 0.0   # epoch; 0 = no person seen since startup


def update_pi2_seen():
    """Called by InferenceThread when YOLO detects at least one person."""
    global _pi2_last_seen
    with _pi2_lock:
        _pi2_last_seen = time.time()


# ─── Watchdog thread ──────────────────────────────────────────────────────────

class ChildWatchdogThread(threading.Thread):
    """
    Polls Pi 1 and compares person-seen timestamps from both cameras.
    Fires a Telegram alert if neither camera has seen anyone for
    CHILD_MISSING_SECONDS seconds.

    Usage:
        wd = ChildWatchdogThread(telegram=telegram_thread)
        wd.start()
        wd.stop()
    """

    def __init__(self, telegram):
        super().__init__(daemon=True)
        self.telegram        = telegram
        self._stopped        = threading.Event()
        self._last_alerted   = 0.0
        self._pi1_fail_count = 0

    def run(self):
        print(f'[Watchdog] Started — polling Pi 1 at {PI1_STATUS_URL} '
              f'every {CHILD_WATCHDOG_POLL_SEC}s')
        while not self._stopped.is_set():
            self._stopped.wait(timeout=CHILD_WATCHDOG_POLL_SEC)
            if self._stopped.is_set():
                break
            self._check()

    def _check(self):
        # ── Poll Pi 1 ─────────────────────────────────────────────────────────
        pi1_seconds_ago = None
        try:
            resp = requests.get(PI1_STATUS_URL, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            pi1_seconds_ago      = float(data.get('seconds_ago', float('inf')))
            self._pi1_fail_count = 0
        except Exception as e:
            self._pi1_fail_count += 1
            print(f'[Watchdog] Pi 1 poll failed ({self._pi1_fail_count}): {e}')
            if self._pi1_fail_count >= CHILD_PI1_OFFLINE_TOLERANCE:
                print('[Watchdog] Pi 1 unreachable — suppressing child-missing alert.')
            return

        # ── Pi 2 own age ──────────────────────────────────────────────────────
        with _pi2_lock:
            pi2_t = _pi2_last_seen
        now             = time.time()
        pi2_seconds_ago = (now - pi2_t) if pi2_t > 0 else float('inf')

        # ── Both cameras must exceed threshold ────────────────────────────────
        both_missing = (pi1_seconds_ago >= CHILD_MISSING_SECONDS and
                        pi2_seconds_ago >= CHILD_MISSING_SECONDS)
        cooldown_ok  = (now - self._last_alerted) >= CHILD_ALERT_COOLDOWN

        if both_missing and cooldown_ok:
            ts  = time.strftime('%Y-%m-%d %H:%M:%S')
            msg = (
                f'⚠️ CHILD NOT SEEN — BOTH CAMERAS\n'
                f'Time           : {ts}\n'
                f'Pi 1 last seen : {self._fmt(pi1_seconds_ago)} ago\n'
                f'Pi 2 last seen : {self._fmt(pi2_seconds_ago)} ago\n'
                f'Threshold      : {CHILD_MISSING_SECONDS}s\n'
                f'Please check on the child immediately.'
            )
            self.telegram.send_photo(msg, _blank_jpeg())
            self._last_alerted = now
            print(f'[Watchdog] CHILD MISSING alert sent at {ts}')
        else:
            print(f'[Watchdog] Pi1={self._fmt(pi1_seconds_ago)} ago  '
                  f'Pi2={self._fmt(pi2_seconds_ago)} ago  '
                  f'threshold={CHILD_MISSING_SECONDS}s')

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(seconds):
        if seconds == float('inf'):
            return 'never'
        return f'{seconds:.0f}s'

    def stop(self):
        self._stopped.set()


def _blank_jpeg() -> bytes:
    """1×1 black JPEG placeholder for watchdog alerts that have no frame."""
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new('RGB', (1, 1)).save(buf, format='JPEG')
        return buf.getvalue()
    except Exception:
        return (b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
                b'\x00\x00\xff\xd9')
