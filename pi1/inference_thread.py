"""
inference_thread.py
====================
InferenceThread — the central processing loop for Pi 1.

Single YOLO call per (possibly skipped) frame drives:
  - Fall analysis  (per person via FallTracker)
  - Fight analysis (per pair  via FightTracker)
  - Audio cross-modal logic (reads AudioState):
      - Fight + laugh recent  → alert SUPPRESSED  (kids playing)
      - Fight/Fall + cry/scream → alert enriched
      - Fall alert delayed FALL_AUDIO_WAIT seconds for audio context
  - Speaker announcements on fall/fight events
  - Respects arm/disarm state from TelegramBotThread
  - HUD overlay + MJPEG frame push
  - Telegram alert dispatch with cooldowns

Shared state exposed for TelegramBotThread /status command:
  - _status dict updated every inference frame
"""

import cv2
import time
import threading
import numpy as np
from collections import deque

from config.settings import (
    CONF_THRESHOLD, INFERENCE_SKIP_FRAMES,
    ALERT_CLIP_SECONDS, ALERT_CLIP_FPS,
    TELEGRAM_GLOBAL_COOLDOWN, TELEGRAM_FIGHT_COOLDOWN,
    FIGHT_GIF_DELAY, FALL_AUDIO_WAIT,
    LAUGH_SUPPRESS_WINDOW, CRY_ENRICH_WINDOW, SCREAM_ENRICH_WINDOW,
)
from detection.pose_utils     import draw_skeleton, state_color, keypoints_snapshot
from detection.fall_detector  import smart_fall_check
from server.web_server        import update_mjpeg_frame, update_person_seen
from alerting.telegram_bot    import is_alerts_enabled


# ─── Shared live status (read by TelegramBotThread for /status command) ──────

_status_lock = threading.Lock()
_status = {
    'fps':             0.0,
    'person_count':    0,
    'audio_ok':        False,
    'yamnet_ok':       False,
    'last_person_ago': 'never',
}


def get_clip_frames_snapshot() -> list:
    """Called by TelegramBotThread for /clip command — returns current clip buffer."""
    # This is replaced at runtime by InferenceThread._get_clip_frames
    return []


def get_live_status() -> dict:
    with _status_lock:
        return dict(_status)


class InferenceThread(threading.Thread):
    """
    Main inference loop. Wires YOLO → fall/fight detectors → speaker → Telegram.
    Speaker and bot are optional — pass None to disable.
    """

    def __init__(self, camera, model, registry, telegram,
                 audio_state=None, speaker=None):
        super().__init__(daemon=True)
        self.camera   = camera
        self.model    = model
        self.registry = registry
        self.telegram = telegram
        self.audio    = audio_state
        self.speaker  = speaker
        self._stopped = threading.Event()

        # FPS counter
        self._fc        = 0
        self._fps_start = time.time()
        self._fps       = 0.0

        # Rolling clip buffer
        self._clip_buf      = deque(maxlen=int(ALERT_CLIP_FPS * ALERT_CLIP_SECONDS) + 5)
        self._clip_interval = 0

        # Frame skip
        self._skip_counter = 0
        self._last_overlay = None

        # Alert timing
        self._last_any_alert_time = 0.0
        self._pending_fight_gifs  = []
        self._pending_fall_audio  = []

        # Last person seen epoch (for /status)
        self._last_person_time = 0.0

    # ── Clip helpers ──────────────────────────────────────────────────────────

    def _get_clip_frames(self):
        frames = list(self._clip_buf)
        target = int(ALERT_CLIP_FPS * ALERT_CLIP_SECONDS)
        if len(frames) > target:
            step   = max(1, len(frames) // target)
            frames = frames[::step][:target]
        return frames

    def _maybe_buffer_clip(self, frame):
        self._clip_interval += 1
        _clip_every = max(1, int((self._fps if self._fps > 1 else 10) / ALERT_CLIP_FPS))
        if self._clip_interval >= _clip_every:
            self._clip_interval = 0
            _, raw_jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
            self._clip_buf.append(raw_jpg.tobytes())

    # ── Alert helpers ─────────────────────────────────────────────────────────

    def _global_ok(self):
        """True if global cooldown has elapsed AND alerts are armed."""
        cooldown_ok = (time.time() - self._last_any_alert_time) >= TELEGRAM_GLOBAL_COOLDOWN
        return cooldown_ok and is_alerts_enabled()

    def _audio_suffix(self):
        if self.audio is None:
            return ''
        parts = []
        if self.audio.recent('cry',    CRY_ENRICH_WINDOW):    parts.append('CRY')
        if self.audio.recent('scream', SCREAM_ENRICH_WINDOW): parts.append('SCREAM')
        return (' + ' + ' + '.join(parts)) if parts else ''

    # ── Status update (for /status command) ───────────────────────────────────

    def _update_status(self, person_count):
        from audio.yamnet_thread  import YAMNET_OK
        from audio.audio_detector import PYAUDIO_OK
        now = time.time()
        if person_count > 0:
            self._last_person_time = now
        if self._last_person_time > 0:
            ago = f'{now - self._last_person_time:.0f}s ago'
        else:
            ago = 'never'
        with _status_lock:
            _status['fps']             = round(self._fps, 1)
            _status['person_count']    = person_count
            _status['audio_ok']        = PYAUDIO_OK
            _status['yamnet_ok']       = YAMNET_OK
            _status['last_person_ago'] = ago

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        while not self._stopped.is_set():
            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]

            # ── Frame skip ────────────────────────────────────────────────────
            if self._skip_counter > 0:
                self._skip_counter -= 1
                if self._last_overlay is not None:
                    update_mjpeg_frame(self._last_overlay)
                continue
            self._skip_counter = INFERENCE_SKIP_FRAMES

            # ── Single YOLO inference ─────────────────────────────────────────
            results  = self.model(frame, conf=CONF_THRESHOLD, imgsz=320, verbose=False)
            result   = results[0]
            kps_list = []
            if result.keypoints is not None:
                kps_list = [
                    k.cpu().numpy() if hasattr(k, 'cpu') else np.array(k)
                    for k in result.keypoints.xy
                ]

            if kps_list:
                update_person_seen()

            self._maybe_buffer_clip(frame)
            self._update_status(len(kps_list))
            annotated = frame.copy()

            # ── Fall detection ────────────────────────────────────────────────
            assignments = self.registry.match(kps_list) if kps_list else []
            hud_lines   = []

            for pid, ft, kps in assignments:
                prev_st    = ft.state
                is_h, cur_angle, debug = smart_fall_check(kps, h)
                mode       = debug.get('mode', '?')
                snap       = keypoints_snapshot(kps)
                fall_state = ft.update(is_h, cur_angle, mode, snap)

                color = state_color(fall_state)
                draw_skeleton(annotated, kps, color, f'P{pid} {fall_state}')
                hud_lines.append(f'P{pid}: {fall_state}')

                _now = time.time()

                if (fall_state == 'FALL DETECTED'
                        and prev_st != 'FALL DETECTED'
                        and ft.fall_alert_ready()
                        and self._global_ok()):
                    # Speaker — immediate announcement
                    if self.speaker:
                        self.speaker.announce_fall(pid)
                    # Telegram — delayed to wait for audio context
                    _, sj = cv2.imencode('.jpg', annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self._pending_fall_audio.append({
                        'send_at': _now + FALL_AUDIO_WAIT,
                        'pid':     pid,
                        'ft':      ft,
                        'snap':    sj.tobytes(),
                        'clip':    self._get_clip_frames(),
                        'time':    time.strftime('%Y-%m-%d %H:%M:%S'),
                    })
                    ft.mark_fall_alerted()
                    self._last_any_alert_time = _now

                elif (fall_state == 'MOTIONLESS AFTER FALL'
                        and ft.motionless_alert_ready()
                        and is_alerts_enabled()):
                    ts     = time.strftime('%Y-%m-%d %H:%M:%S')
                    suffix = self._audio_suffix()
                    if self.speaker:
                        self.speaker.announce_motionless(pid)
                    _, sj = cv2.imencode('.jpg', annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self.telegram.send(
                        f'MOTIONLESS AFTER FALL{suffix} — P{pid}\n'
                        f'Time: {ts}\nStill: {ft.motionless_still_seconds():.0f}s',
                        sj.tobytes(), self._get_clip_frames())
                    ft.mark_motionless_alerted()
                    self._last_any_alert_time = _now

                elif (fall_state == 'Motionless'
                        and prev_st == 'Lying Down'
                        and ft.motionless_alert_ready()
                        and is_alerts_enabled()):
                    ts     = time.strftime('%Y-%m-%d %H:%M:%S')
                    suffix = self._audio_suffix()
                    if self.speaker:
                        self.speaker.announce_motionless(pid)
                    _, sj  = cv2.imencode('.jpg', annotated,
                                          [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self.telegram.send(
                        f'P{pid} motionless (lying){suffix}\nTime: {ts}',
                        sj.tobytes())
                    ft.mark_motionless_alerted()
                    self._last_any_alert_time = _now

            # ── Fight detection ───────────────────────────────────────────────
            fight_pairs = self.registry.get_fight_pairs(assignments)
            worst_fight = 'Normal'

            for pair_kps, ft in fight_pairs:
                prev_st = ft.state
                f_state, f_sig = ft.update(pair_kps)

                if 'FIGHT' in f_state:
                    worst_fight = f_state
                elif 'Possible' in f_state and 'FIGHT' not in worst_fight:
                    worst_fight = f_state
                elif 'Interaction' in f_state and worst_fight == 'Normal':
                    worst_fight = f_state

                if f_state != 'Normal':
                    fc = state_color(f_state)
                    for pk in pair_kps:
                        draw_skeleton(annotated, pk, fc, f_state)

                _now_f    = time.time()
                _fight_ok = (_now_f - ft.last_alerted) >= TELEGRAM_FIGHT_COOLDOWN

                if (f_state == 'FIGHT DETECTED'
                        and prev_st != 'FIGHT DETECTED'
                        and _fight_ok
                        and self._global_ok()):

                    if (self.audio is not None
                            and self.audio.recent('laugh', LAUGH_SUPPRESS_WINDOW)):
                        print(f'[Fight] Alert SUPPRESSED — laugh within '
                              f'{LAUGH_SUPPRESS_WINDOW}s (likely playing)')
                        ft.mark_alerted()
                        self._last_any_alert_time = _now_f
                    else:
                        # Speaker announcement
                        if self.speaker:
                            self.speaker.announce_fight(ft.pair_id)
                        suffix = self._audio_suffix()
                        msg = (f'FIGHT DETECTED{suffix} (pair #{ft.pair_id})\n'
                               f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
                               f'Score: {f_sig.get("score", "?")}')
                        self._pending_fight_gifs.append(
                            (_now_f + FIGHT_GIF_DELAY, msg, ft))
                        ft.mark_alerted()
                        self._last_any_alert_time = _now_f

            if not assignments:
                self.registry.reset_idle()
                for kps in kps_list:
                    draw_skeleton(annotated, kps, (120, 120, 120))

            # ── Delayed fight GIF sends ───────────────────────────────────────
            _now_pg       = time.time()
            still_pending = []
            for send_at, msg, ftrk in self._pending_fight_gifs:
                if _now_pg >= send_at:
                    _, sj = cv2.imencode('.jpg', annotated,
                                          [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self.telegram.send(msg, sj.tobytes(), self._get_clip_frames())
                else:
                    still_pending.append((send_at, msg, ftrk))
            self._pending_fight_gifs = still_pending

            # ── Delayed fall alerts ───────────────────────────────────────────
            _now_pfa           = time.time()
            still_pending_fall = []
            for pf in self._pending_fall_audio:
                if _now_pfa < pf['send_at']:
                    still_pending_fall.append(pf)
                    continue
                parts = []
                if self.audio is not None:
                    if self.audio.recent('cry',    FALL_AUDIO_WAIT + 1.0): parts.append('CRY')
                    if self.audio.recent('scream', FALL_AUDIO_WAIT + 1.0): parts.append('SCREAM')
                suffix = (' + ' + ' + '.join(parts)) if parts else ''
                msg = f'FALL DETECTED{suffix} — P{pf["pid"]}\nTime: {pf["time"]}'
                self.telegram.send(msg, pf['snap'], pf['clip'])
                print(f'[Fall] Alert sent: P{pf["pid"]}{suffix}')
            self._pending_fall_audio = still_pending_fall

            # ── Alert border ──────────────────────────────────────────────────
            any_fall_alert = any('FALL' in ft.state for _, ft, _ in assignments)
            if 'FIGHT' in worst_fight or any_fall_alert:
                cv2.rectangle(annotated, (2, 2),   (w-3, h-3), (0, 0, 180), 6,  cv2.LINE_AA)
                cv2.rectangle(annotated, (0, 0),   (w-1, h-1), (0, 0, 255), 2,  cv2.LINE_AA)
            elif 'Possible' in worst_fight:
                cv2.rectangle(annotated, (1, 1),   (w-2, h-2), (0, 60, 200), 3, cv2.LINE_AA)

            # ── HUD ───────────────────────────────────────────────────────────
            if worst_fight != 'Normal':
                hud_lines.append(f'Fight: {worst_fight}')

            if self.audio is not None:
                audio_hud = []
                for lbl, win in [
                    ('scream', SCREAM_ENRICH_WINDOW),
                    ('cry',    CRY_ENRICH_WINDOW),
                    ('laugh',  LAUGH_SUPPRESS_WINDOW),
                    ('impact', 4.0),
                ]:
                    if self.audio.recent(lbl, win):
                        audio_hud.append(lbl.upper())
                if audio_hud:
                    hud_lines.append(f'Audio: {" ".join(audio_hud)}')

            # Arm/disarm status on HUD
            if not is_alerts_enabled():
                hud_lines.append('ALERTS: DISARMED')

            n = len(hud_lines)
            if n > 0:
                box_h   = 12 + n * 20
                overlay = annotated[0:box_h, 0:360].copy()
                cv2.rectangle(overlay, (0, 0), (360, box_h), (0, 0, 0), -1)
                annotated[0:box_h, 0:360] = cv2.addWeighted(
                    overlay, 0.65, annotated[0:box_h, 0:360], 0.35, 0)
                for i, line in enumerate(hud_lines):
                    c = state_color(line.split(': ', 1)[-1] if ': ' in line else line)
                    cv2.putText(annotated, line, (8, 16 + i*20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

            # FPS counter
            self._fc += 1
            el = time.time() - self._fps_start
            if el >= 1.0:
                self._fps       = self._fc / el
                self._fc        = 0
                self._fps_start = time.time()
            cv2.putText(annotated, f'FPS: {self._fps:.1f}', (w-110, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            # Push to MJPEG server
            _, jbuf   = cv2.imencode('.jpg', annotated,
                                      [cv2.IMWRITE_JPEG_QUALITY, 50])
            jpg_bytes = jbuf.tobytes()
            update_mjpeg_frame(jpg_bytes)
            self._last_overlay = jpg_bytes

    def stop(self):
        self._stopped.set()

    def get_clip_frames_for_bot(self) -> list:
        """Public accessor for TelegramBotThread /clip command."""
        return self._get_clip_frames()
