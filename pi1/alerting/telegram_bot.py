"""
alerting/telegram_bot.py
=========================
TelegramBotThread — polls for commands using plain requests.get/post.
No Session, no HTTPAdapter. Python 3.13 compatible.

Commands:
    /help /temp /status /snapshot /clip /arm /disarm /uptime
"""

import time
import threading
import requests

from config.settings import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

BASE_URL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}'


def _tg_get(endpoint, params=None):
    return requests.get(f'{BASE_URL}/{endpoint}', params=params, timeout=6).json()


def _tg_post(endpoint, data=None, files=None, timeout=12):
    return requests.post(f'{BASE_URL}/{endpoint}', data=data, files=files, timeout=timeout).json()


# ─── Shared arm/disarm state ──────────────────────────────────────────────────

_alerts_lock   = threading.Lock()
alerts_enabled = True


def is_alerts_enabled() -> bool:
    with _alerts_lock:
        return alerts_enabled


# ─── Bot thread ───────────────────────────────────────────────────────────────

class TelegramBotThread(threading.Thread):

    _COMMANDS = {
        '/help':     'List all commands',
        '/temp':     'Current temperature and humidity',
        '/status':   'System status (FPS, detections, audio, uptime)',
        '/snapshot': 'Send a live photo from the camera',
        '/clip':     'Send a GIF clip from the last few seconds',
        '/arm':      'Enable Telegram alerts',
        '/disarm':   'Disable Telegram alerts (e.g. when you are home)',
        '/uptime':   'How long Pi 1 has been running',
    }

    def __init__(self, temp_sensor=None, speaker=None, get_status_fn=None,
                 get_frame_fn=None, get_clip_fn=None):
        super().__init__(daemon=True, name='TelegramBot')
        self.temp_sensor   = temp_sensor
        self.speaker       = speaker
        self.get_status_fn = get_status_fn
        self.get_frame_fn  = get_frame_fn
        self.get_clip_fn   = get_clip_fn
        self._stopped      = threading.Event()
        self._offset       = 0
        self._start_time   = time.time()

    def run(self):
        print('[Bot] Telegram command bot started.')
        while not self._stopped.is_set():
            try:
                data = _tg_get('getUpdates', params={
                    'offset':  self._offset,
                    'timeout': 1,
                    'limit':   10,
                })
                if data.get('ok'):
                    updates = data.get('result', [])
                    if updates:
                        self._offset = updates[-1]['update_id'] + 1
                    for update in updates:
                        self._dispatch(update)
            except Exception as e:
                print(f'[Bot] Poll error: {e}')
                time.sleep(3)

    def _dispatch(self, update):
        msg     = update.get('message', {})
        text    = msg.get('text', '').strip()
        chat_id = str(msg.get('chat', {}).get('id', ''))

        if chat_id != str(TELEGRAM_CHAT_ID):
            return

        cmd = text.split()[0].lower() if text else ''
        print(f'[Bot] Command: {cmd} from {chat_id}')

        if   cmd == '/help':      self._cmd_help(chat_id)
        elif cmd == '/temp':      self._cmd_temp(chat_id)
        elif cmd == '/status':    self._cmd_status(chat_id)
        elif cmd == '/snapshot':  self._cmd_snapshot(chat_id)
        elif cmd == '/clip':      self._cmd_clip(chat_id)
        elif cmd == '/arm':       self._cmd_arm(chat_id)
        elif cmd == '/disarm':    self._cmd_disarm(chat_id)
        elif cmd == '/uptime':    self._cmd_uptime(chat_id)
        elif cmd.startswith('/'): self._send(chat_id,
            f'Unknown command: {cmd}\nSend /help for a list.')

    def _send(self, chat_id, text):
        try:
            _tg_post('sendMessage', data={
                'chat_id':    chat_id,
                'text':       text,
                'parse_mode': 'Markdown',
            })
        except Exception as e:
            print(f'[Bot] Send failed: {e}')

    def _fmt_uptime(self):
        s     = int(time.time() - self._start_time)
        return f'{s//3600}h {(s%3600)//60}m {s%60}s'

    def _cmd_help(self, chat_id):
        lines = ['🤖 *Pi 1 Monitor Commands*\n']
        for cmd, desc in self._COMMANDS.items():
            lines.append(f'{cmd} — {desc}')
        self._send(chat_id, '\n'.join(lines))

    def _cmd_temp(self, chat_id):
        if self.temp_sensor is None:
            self._send(chat_id, '❌ Temperature sensor not available.')
            return
        msg = self.temp_sensor.read_formatted()
        age = self.temp_sensor.last_read_age()
        if age < 9999:
            msg += f'\n🕐 Reading taken {age:.0f}s ago'
        self._send(chat_id, msg)
        if self.speaker:
            temp, hum = self.temp_sensor.read()
            if temp is not None:
                self.speaker.announce_temp(temp, hum)

    def _cmd_status(self, chat_id):
        s   = self.get_status_fn() if self.get_status_fn else {}
        arm = '✅ Armed' if is_alerts_enabled() else '🔕 Disarmed'
        msg = (
            f'📊 *Pi 1 System Status*\n\n'
            f'🎥 FPS          : {s.get("fps", 0):.1f}\n'
            f'👤 People seen  : {s.get("person_count", 0)}\n'
            f'🔊 Audio        : {"Active" if s.get("audio_ok") else "Disabled"}\n'
            f'🧠 YAMNet       : {"Active" if s.get("yamnet_ok") else "Disabled"}\n'
            f'🔔 Alerts       : {arm}\n'
            f'⏱ Uptime       : {self._fmt_uptime()}\n'
            f'🕐 Last person  : {s.get("last_person_ago", "never")}\n'
        )
        if self.temp_sensor:
            temp, hum = self.temp_sensor.read()
            if temp is not None:
                msg += f'🌡 Temp         : {temp:.1f}°C  💧 {hum:.1f}%\n'
        self._send(chat_id, msg)

    def _cmd_snapshot(self, chat_id):
        if self.get_frame_fn is None:
            self._send(chat_id, '❌ Camera not available.')
            return
        jpeg = self.get_frame_fn()
        if jpeg is None:
            self._send(chat_id, '⚠️ No frame available yet.')
            return
        try:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            _tg_post('sendPhoto',
                     data={'chat_id': chat_id, 'caption': f'📸 {ts}'},
                     files={'photo': ('snap.jpg', jpeg, 'image/jpeg')})
        except Exception as e:
            print(f'[Bot] Snapshot failed: {e}')

    def _cmd_clip(self, chat_id):
        if self.get_clip_fn is None:
            self._send(chat_id, '❌ Clip buffer not available.')
            return
        frames = self.get_clip_fn()
        if not frames or len(frames) < 2:
            self._send(chat_id, '⚠️ Not enough frames yet — try again shortly.')
            return
        self._send(chat_id, '🎞 Building clip...')

        def _build():
            from alerting.telegram_alert import build_gif_from_jpegs
            gif = build_gif_from_jpegs(frames)
            if gif:
                try:
                    ts = time.strftime('%Y-%m-%d %H:%M:%S')
                    _tg_post('sendAnimation',
                             data={'chat_id': chat_id, 'caption': f'🎞 {ts}'},
                             files={'animation': ('clip.gif', gif, 'image/gif')},
                             timeout=25)
                except Exception as e:
                    print(f'[Bot] Clip send failed: {e}')
            else:
                self._send(chat_id, '❌ Failed to build GIF.')

        threading.Thread(target=_build, daemon=True).start()

    def _cmd_arm(self, chat_id):
        global alerts_enabled
        with _alerts_lock:
            alerts_enabled = True
        self._send(chat_id, '✅ Alerts ARMED.')
        if self.speaker:
            self.speaker.say('Alerts have been enabled.')

    def _cmd_disarm(self, chat_id):
        global alerts_enabled
        with _alerts_lock:
            alerts_enabled = False
        self._send(chat_id, '🔕 Alerts DISARMED.')
        if self.speaker:
            self.speaker.say('Alerts have been disabled.')

    def _cmd_uptime(self, chat_id):
        self._send(chat_id, f'⏱ Uptime: {self._fmt_uptime()}')

    def stop(self):
        self._stopped.set()
