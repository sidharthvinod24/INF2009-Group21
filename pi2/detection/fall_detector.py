"""
detection/fall_detector.py
===========================
Fall detection logic for Pi 1.

Exports:
  smart_fall_check(kps, frame_h) -> (is_horizontal, cur_angle, debug_dict)
  MotionlessTracker              — tracks whether a person has stopped moving
  FallTracker                    — per-person state machine (Safe → FALL DETECTED)
"""

import time
import numpy as np
from collections import deque

from config.settings import (
    SPINE_HORIZONTAL_RATIO, TORSO_ANGLE_THRESHOLD, HIP_GROUND_RATIO,
    FALL_CONFIRM_SECONDS, LYING_DOWN_SECONDS,
    ANGLE_BUFFER_SIZE, FALL_ENTRY_DELTA, FALL_ANGLE_DELTA_THRESHOLD,
    MOTIONLESS_SAMPLE_FRAMES,
    MOTIONLESS_THRESHOLD_LYING, MOTIONLESS_THRESHOLD_FALL,
    MOTIONLESS_SECONDS_LYING, MOTIONLESS_SECONDS_FALL,
    MOTIONLESS_ALERT_REPEAT, TELEGRAM_COOLDOWN,
)
from detection.pose_utils import get_kp, mean_displacement


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def spine_angle_deg(kps):
    """
    Return (mean_angle_deg, mean_dx_dy_ratio) across visible shoulder-hip pairs.
    Both values are None if no valid pair found.
    """
    pairs = []
    for s, h in [(5, 11), (6, 12)]:
        sp, hp = get_kp(kps, s), get_kp(kps, h)
        if sp and hp:
            pairs.append((sp, hp))
    if not pairs:
        return None, None

    ratios, angles = [], []
    for shoulder, hip in pairs:
        dx, dy = abs(shoulder[0]-hip[0]), abs(shoulder[1]-hip[1])
        if dy > 0:
            ratios.append(dx / dy)
        total = (dx**2 + dy**2) ** 0.5
        if total > 0:
            angles.append(np.degrees(np.arctan2(dx, dy)))

    if not angles:
        return None, None
    return (float(np.mean(angles)),
            round(float(np.mean(ratios)), 2) if ratios else None)


def spine_is_horizontal(kps):
    angle, ratio = spine_angle_deg(kps)
    if angle is None:
        return False, None
    return (ratio is not None and ratio > SPINE_HORIZONTAL_RATIO), ratio


def hips_near_ground(kps, frame_h):
    hip = get_kp(kps, 11) or get_kp(kps, 12)
    if hip is None:
        return None
    return hip[1] > frame_h * HIP_GROUND_RATIO


def torso_angle_check(kps):
    """
    Compute shoulder-hip-knee angle.
    Returns (angle_below_threshold, angle_value) or (None, None).
    """
    from detection.pose_utils import calculate_angle
    shoulder = get_kp(kps, 5) or get_kp(kps, 6)
    hip      = get_kp(kps, 11) or get_kp(kps, 12)
    knee     = get_kp(kps, 13) or get_kp(kps, 14)
    if not (shoulder and hip and knee):
        return None, None
    angle = calculate_angle(shoulder, hip, knee)
    if angle is None:
        return None, None
    return angle < TORSO_ANGLE_THRESHOLD, round(angle, 1)


def head_beside_shoulder(kps):
    """
    Returns (head_beside_shoulder_bool, dx/dy_ratio).
    Used as fallback when only upper body is visible.
    """
    nose     = get_kp(kps, 0)
    shoulder = get_kp(kps, 5) or get_kp(kps, 6)
    if not nose or not shoulder:
        return None, None
    dx, dy = abs(nose[0]-shoulder[0]), abs(nose[1]-shoulder[1])
    if dy == 0:
        return None, None
    ratio = dx / dy
    return ratio > 0.8, round(ratio, 2)


# ─── Multi-mode fall check ────────────────────────────────────────────────────

def smart_fall_check(kps, frame_h):
    """
    Decide whether a person is in a horizontal/fallen pose using the richest
    available keypoints.  Falls back gracefully as body parts go out of frame.

    Returns:
        is_horizontal (bool), cur_angle (float|None), debug (dict)
    """
    debug = {}
    has_hip      = get_kp(kps, 11) or get_kp(kps, 12)
    has_knee     = get_kp(kps, 13) or get_kp(kps, 14)
    has_shoulder = get_kp(kps, 5)  or get_kp(kps, 6)
    has_nose     = get_kp(kps, 0)
    cur_angle, _ = spine_angle_deg(kps)

    if has_shoulder and has_hip and has_knee:
        debug['mode'] = 'Full body'
        spine_fallen, spine_ratio   = spine_is_horizontal(kps)
        angle_fallen, torso_angle   = torso_angle_check(kps)
        hip_low                     = hips_near_ground(kps, frame_h)
        debug.update(spine_ratio=spine_ratio, torso_angle=torso_angle, hip_low=hip_low)
        is_h = spine_fallen and (angle_fallen or hip_low)
        return is_h, cur_angle, debug

    elif has_shoulder and has_hip:
        debug['mode'] = 'Torso only'
        spine_fallen, spine_ratio = spine_is_horizontal(kps)
        hip_low                   = hips_near_ground(kps, frame_h)
        debug.update(spine_ratio=spine_ratio, hip_low=hip_low)
        if hip_low is True:
            is_h = spine_fallen
        elif hip_low is False:
            is_h = (spine_ratio is not None and
                    spine_ratio > SPINE_HORIZONTAL_RATIO * 1.5)
        else:
            is_h = spine_fallen
        return is_h, cur_angle, debug

    elif has_shoulder and has_nose:
        debug['mode'] = 'Upper body'
        head_fallen, head_ratio = head_beside_shoulder(kps)
        debug['head_ratio'] = head_ratio
        return head_fallen is True, cur_angle, debug

    else:
        debug['mode'] = 'Insufficient'
        return False, None, debug


# ─── Motionless tracker ───────────────────────────────────────────────────────

class MotionlessTracker:
    """
    Confirms a person has stopped moving by comparing keypoint snapshots
    sampled MOTIONLESS_SAMPLE_FRAMES apart.  Confirmed only if displacement
    stays below threshold_px for at least required_seconds.
    """

    def __init__(self, threshold_px, required_seconds):
        self.threshold_px    = threshold_px
        self.required_seconds = required_seconds
        self._snap_buffer    = deque(maxlen=MOTIONLESS_SAMPLE_FRAMES + 1)
        self._still_since    = None
        self.current_disp    = 0.0
        self.confirmed       = False

    def update(self, snap):
        self._snap_buffer.append(snap)
        if len(self._snap_buffer) < MOTIONLESS_SAMPLE_FRAMES + 1:
            return False
        self.current_disp = mean_displacement(
            self._snap_buffer[0], self._snap_buffer[-1])
        now = time.time()
        if self.current_disp <= self.threshold_px:
            if self._still_since is None:
                self._still_since = now
            if now - self._still_since >= self.required_seconds:
                self.confirmed = True
        else:
            self._still_since = None
            self.confirmed    = False
        return self.confirmed

    def reset(self):
        self._snap_buffer.clear()
        self._still_since = None
        self.confirmed    = False
        self.current_disp = 0.0

    @property
    def still_seconds(self):
        return (time.time() - self._still_since) if self._still_since else 0.0


# ─── Per-person fall state machine ────────────────────────────────────────────

class FallTracker:
    """
    Maintains the fall/lying state for a single tracked person.

    States:
        Safe → Observing... → Horizontal... → Lying Down → Motionless
                                            → Possible fall... → FALL DETECTED
                                                               → MOTIONLESS AFTER FALL
    """

    def __init__(self, pid):
        self.pid                     = pid
        self.angle_buffer            = deque(maxlen=ANGLE_BUFFER_SIZE)
        self.was_horizontal          = False
        self.buffer_frozen           = False
        self.entry_delta             = None
        self.fall_path               = None
        self.fall_start_time         = None
        self.lying_start_time        = None
        self.confirmed_state         = None
        self.state                   = 'Safe'
        self.last_seen               = time.time()
        self.last_alerted            = 0.0
        self.last_motionless_alerted = 0.0
        self._mt_lying = MotionlessTracker(
            MOTIONLESS_THRESHOLD_LYING, MOTIONLESS_SECONDS_LYING)
        self._mt_fall  = MotionlessTracker(
            MOTIONLESS_THRESHOLD_FALL,  MOTIONLESS_SECONDS_FALL)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _active_mt(self):
        if self.confirmed_state == 'fall':   return self._mt_fall
        if self.confirmed_state == 'lying':  return self._mt_lying
        return None

    def _buffer_delta(self):
        valid = [a for a in self.angle_buffer if a is not None]
        return float(max(valid) - min(valid)) if len(valid) >= 3 else 0.0

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, is_horizontal, cur_angle, mode, kps_snap):
        """Call once per inference frame with the result of smart_fall_check."""
        if not self.buffer_frozen:
            self.angle_buffer.append(cur_angle)

        if not is_horizontal:
            self._reset_to_safe()
            return self.state

        mt = self._active_mt()
        if mt is not None:
            mt.update(kps_snap)

        if self.confirmed_state == 'fall':
            self.state = ('MOTIONLESS AFTER FALL'
                          if self._mt_fall.confirmed else 'FALL DETECTED')
            return self.state

        if self.confirmed_state == 'lying':
            self.state = 'Motionless' if self._mt_lying.confirmed else 'Lying Down'
            return self.state

        # First frame going horizontal
        if not self.was_horizontal and is_horizontal:
            self.was_horizontal = True
            self.buffer_frozen  = True
            self.entry_delta    = self._buffer_delta()
            if self.entry_delta >= FALL_ENTRY_DELTA:
                self.fall_path      = True
                self.fall_start_time = time.time()
            else:
                self.fall_path       = False
                self.lying_start_time = time.time()

        if not self.was_horizontal and len(self.angle_buffer) < ANGLE_BUFFER_SIZE:
            self.state = 'Observing...'
            return self.state

        if self.fall_path is None:
            delta          = self._buffer_delta()
            self.fall_path = delta >= FALL_ANGLE_DELTA_THRESHOLD
            now            = time.time()
            self.fall_start_time  = now if self.fall_path  else None
            self.lying_start_time = now if not self.fall_path else None

        now  = time.time()
        csec = FALL_CONFIRM_SECONDS + (2 if 'Upper body' in mode else 0)

        if self.fall_path:
            elapsed = now - self.fall_start_time
            if elapsed >= csec:
                self.confirmed_state = 'fall'
                self.state           = 'FALL DETECTED'
            else:
                self.state = f'Possible fall... ({elapsed:.1f}s/{csec}s)'
        else:
            elapsed = now - self.lying_start_time
            if elapsed >= LYING_DOWN_SECONDS:
                self.confirmed_state = 'lying'
                self.state           = 'Lying Down'
            else:
                self.state = f'Horizontal... ({elapsed:.1f}s/{LYING_DOWN_SECONDS}s)'

        return self.state

    def _reset_to_safe(self):
        self.was_horizontal  = False
        self.buffer_frozen   = False
        self.entry_delta     = None
        self.fall_path       = None
        self.fall_start_time = None
        self.lying_start_time = None
        self.confirmed_state = None
        self.state           = 'Safe'
        self._mt_lying.reset()
        self._mt_fall.reset()

    # ── Alert readiness ───────────────────────────────────────────────────────

    def fall_alert_ready(self):
        return time.time() - self.last_alerted >= TELEGRAM_COOLDOWN

    def mark_fall_alerted(self):
        self.last_alerted = time.time()

    def motionless_alert_ready(self):
        if MOTIONLESS_ALERT_REPEAT <= 0:
            return self.last_motionless_alerted == 0.0
        return time.time() - self.last_motionless_alerted >= MOTIONLESS_ALERT_REPEAT

    def mark_motionless_alerted(self):
        self.last_motionless_alerted = time.time()

    def motionless_disp(self):
        mt = self._active_mt()
        return mt.current_disp if mt else None

    def motionless_still_seconds(self):
        mt = self._active_mt()
        return mt.still_seconds if mt else 0.0
