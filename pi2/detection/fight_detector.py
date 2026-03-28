"""
detection/fight_detector.py
============================
Fight detection logic for Pi 1.

Architecture:
    FightScorer       — computes a raw score (0-14) from 11 signals per frame pair
    RollingFightWindow — confirms score stays high over a time window
    FightTracker      — per-pair state machine (Normal → Possible → FIGHT DETECTED)

Signal list (S1-S11):
    S1  Wrist/elbow velocity          S7  Strike burst (acceleration)
    S2  Wrist invasion                S8  Torso push
    S3  Facing each other             S9  Mutual aggression
    S4  Pose instability              S10 Grapple (IoU delta)
    S5  Head proximity                S11 Asymmetric aggression (attacker/victim)
    S6  Kick + ankle invasion
"""

import time
import numpy as np
from collections import deque

from config.settings import (
    DEPTH_CONFIDENCE_THRESHOLD, MAX_HEIGHT_RATIO,
    MIN_BBOX_IOU, MAX_BBOX_IOU,
    MIN_LATERAL_SEPARATION, MAX_LATERAL_SEPARATION,
    MIN_SHOULDER_WIDTH_RATIO, PROXIMITY_THRESHOLD,
    WRIST_VELOCITY_THRESHOLD, ELBOW_VELOCITY_THRESHOLD,
    KNEE_VELOCITY_THRESHOLD, MOTION_BUFFER_SIZE,
    POSE_INSTABILITY_THRESH,
    POSSIBLE_FIGHT_SCORE, FIGHT_SCORE_THRESHOLD,
    SCORE_SMOOTHING_ALPHA,
    FIGHT_CONFIRM_SECONDS, FIGHT_WINDOW_SECONDS, FIGHT_WINDOW_RATIO,
    HEAD_PROXIMITY_RATIO, ANKLE_INVASION_ENABLED,
    ACCELERATION_MULTIPLIER, TORSO_VELOCITY_THRESHOLD,
    TORSO_ACCEL_THRESHOLD, MUTUAL_AGGRESSION_BONUS,
    GRAPPLE_IOU_THRESHOLD, GRAPPLE_IOU_DELTA,
    VICTIM_STATIONARY_THRESHOLD, TELEGRAM_COOLDOWN,
)
from detection.pose_utils import (
    get_kp, bbox_from_kps, bbox_height, bbox_width, bbox_centre,
    compute_iou, point_in_bbox, normalised_distance, shoulder_width_px,
    limb_velocity,
)


# ─── Depth / proximity gate ───────────────────────────────────────────────────

def depth_confidence(ba, bb, kps_a, kps_b):
    """
    Estimate whether two people are roughly at the same depth in the scene.
    Returns (confidence 0..1, debug_dict).
    Below DEPTH_CONFIDENCE_THRESHOLD the pair is skipped for fight scoring.
    """
    if ba is None or bb is None:
        return 0.0, {}

    ha, hb = bbox_height(ba), bbox_height(bb)
    wa, wb = bbox_width(ba),  bbox_width(bb)
    debug  = {}

    # Height similarity score
    h_ratio = max(ha, hb) / (min(ha, hb) + 1e-6)
    hs = max(0.0, 1.0 - (h_ratio - 1.0) / (MAX_HEIGHT_RATIO - 1.0 + 1e-6))
    debug['h_ratio'] = round(h_ratio, 2)

    # IoU score (some overlap expected for a confrontation)
    iou     = compute_iou(ba, bb)
    iou_mid = (MIN_BBOX_IOU + MAX_BBOX_IOU) / 2
    iou_half = (MAX_BBOX_IOU - MIN_BBOX_IOU) / 2
    is_ = (0.0 if iou < MIN_BBOX_IOU or iou > MAX_BBOX_IOU
           else max(0.0, 1.0 - abs(iou - iou_mid) / (iou_half + 1e-6)))
    debug['iou'] = round(iou, 2)

    # Lateral separation score
    cxa, _ = bbox_centre(ba)
    cxb, _ = bbox_centre(bb)
    avg_w  = (wa + wb) / 2
    lat    = abs(cxa - cxb) / (avg_w + 1e-6)
    if lat < MIN_LATERAL_SEPARATION:
        ls = lat / (MIN_LATERAL_SEPARATION + 1e-6)
    elif lat > MAX_LATERAL_SEPARATION:
        ls = 0.0
    else:
        ls = 1.0 - (lat - MIN_LATERAL_SEPARATION) / (
            MAX_LATERAL_SEPARATION - MIN_LATERAL_SEPARATION + 1e-6)
    ls = max(0.0, min(1.0, ls))
    debug['lat_sep'] = round(lat, 2)

    # Shoulder-width ratio score
    swa, swb = shoulder_width_px(kps_a), shoulder_width_px(kps_b)
    if swa and swb and swa > 0 and swb > 0:
        nswa  = swa / (ha + 1e-6)
        nswb  = swb / (hb + 1e-6)
        swr   = min(nswa, nswb) / (max(nswa, nswb) + 1e-6)
        ss    = max(0.0, (swr - MIN_SHOULDER_WIDTH_RATIO) /
                         (1.0 - MIN_SHOULDER_WIDTH_RATIO + 1e-6))
        debug['sw_ratio'] = round(swr, 2)
    else:
        ss = 0.5
        debug['sw_ratio'] = None

    conf = round(0.25*hs + 0.30*is_ + 0.30*ls + 0.15*ss, 3)
    debug['confidence'] = conf
    return conf, debug


# ─── Orientation helpers ──────────────────────────────────────────────────────

def facing_each_other(kps_a, kps_b):
    """True when shoulder forward-vectors point roughly toward each other."""
    def fvec(kps):
        l, r = get_kp(kps, 5), get_kp(kps, 6)
        if l is None or r is None:
            return None
        dx, dy = r[0]-l[0], r[1]-l[1]
        return (-dy, dx)
    fa, fb = fvec(kps_a), fvec(kps_b)
    if fa is None or fb is None:
        return False
    na = (fa[0]**2 + fa[1]**2)**0.5 + 1e-6
    nb = (fb[0]**2 + fb[1]**2)**0.5 + 1e-6
    dot = (fa[0]/na)*(fb[0]/nb) + (fa[1]/na)*(fb[1]/nb)
    return dot < -0.3


def joint_angle(a, b, c):
    bax, bay = a[0]-b[0], a[1]-b[1]
    bcx, bcy = c[0]-b[0], c[1]-b[1]
    na = (bax**2 + bay**2)**0.5
    nb = (bcx**2 + bcy**2)**0.5
    if na == 0 or nb == 0:
        return None
    cos_val = max(-1.0, min(1.0, (bax*bcx + bay*bcy) / (na * nb)))
    return float(np.degrees(np.arccos(cos_val)))


def arm_angle_variance(kps, angle_buf):
    """Variance of arm angles added to a rolling buffer — detects flailing."""
    angles = []
    for sh, el, wr in [(5, 7, 9), (6, 8, 10)]:
        s, e, w = get_kp(kps, sh), get_kp(kps, el), get_kp(kps, wr)
        if s and e and w:
            a = joint_angle(s, e, w)
            if a is not None:
                angles.append(a)
    if angles:
        angle_buf.append(np.mean(angles))
    if len(angle_buf) < 3:
        return 0.0
    return float(np.var(list(angle_buf)))


# ─── Fight scorer ─────────────────────────────────────────────────────────────

class FightScorer:
    """
    Stateful scorer for a single person-pair.
    Returns an integer score each frame; higher = more fight-like behaviour.
    Keypoints are EMA-smoothed before scoring to reduce jitter.
    """

    def __init__(self):
        self.angle_buf       = [deque(maxlen=MOTION_BUFFER_SIZE),
                                deque(maxlen=MOTION_BUFFER_SIZE)]
        self.prev_kps        = [None, None]
        self.prev_wrist_vel  = [0.0, 0.0]
        self.prev_iou        = 0.0
        self.prev_torso      = [None, None]
        self.smoothed_score  = 0.0
        self._kp_smooth      = [None, None]
        self._KP_ALPHA       = 0.5

    def _smooth_kps(self, kps, idx):
        """EMA smooth keypoints — reduces pose jitter without torch overhead."""
        if hasattr(kps, 'cpu'):
            raw = kps.cpu().numpy()
        elif not isinstance(kps, np.ndarray):
            raw = np.array(kps, dtype=np.float32)
        else:
            raw = kps
        raw = np.ascontiguousarray(raw, dtype=np.float32)
        if self._kp_smooth[idx] is None:
            self._kp_smooth[idx] = raw.copy()
            return raw
        prev = self._kp_smooth[idx]
        mask  = (raw[:, 0] != 0) | (raw[:, 1] != 0)
        pmask = (prev[:, 0] != 0) | (prev[:, 1] != 0)
        both  = mask & pmask
        out   = raw.copy()
        out[both] = self._KP_ALPHA * prev[both] + (1 - self._KP_ALPHA) * raw[both]
        self._kp_smooth[idx] = out
        return out

    def _hip_mid(self, kps):
        lh, rh = get_kp(kps, 11), get_kp(kps, 12)
        if lh is None or rh is None:
            return None
        return [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2]

    def _lower_bbox(self, kps):
        pts = [get_kp(kps, i) for i in range(11, 17)]
        pts = [p for p in pts if p is not None]
        if len(pts) < 2:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def update(self, kps_list):
        kps_a = self._smooth_kps(kps_list[0], 0)
        kps_b = self._smooth_kps(kps_list[1], 1)
        ba, bb = bbox_from_kps(kps_a), bbox_from_kps(kps_b)
        sig    = {}
        score  = 0

        # ── Depth gate ────────────────────────────────────────────────────────
        conf, dd = depth_confidence(ba, bb, kps_a, kps_b)
        sig['depth_conf']  = conf
        sig['depth_debug'] = dd
        if conf < DEPTH_CONFIDENCE_THRESHOLD:
            sig['skip_reason'] = f'depth {conf:.2f}'
            self.prev_kps = [kps_a, kps_b]
            return 0, sig

        # ── Proximity gate ────────────────────────────────────────────────────
        nd = normalised_distance(ba, bb)
        sig['norm_dist'] = round(nd, 2)
        if nd > PROXIMITY_THRESHOLD:
            sig['skip_reason'] = f'far {nd:.2f}'
            self.prev_kps = [kps_a, kps_b]
            return 0, sig

        sig['skip_reason'] = None
        ha, hb = bbox_height(ba), bbox_height(bb)

        # S1 — wrist / elbow velocity
        wva = wvb = eva = evb = 0.0
        if self.prev_kps[0] is not None:
            wva = limb_velocity(kps_a, self.prev_kps[0], [9, 10])
            eva = limb_velocity(kps_a, self.prev_kps[0], [7, 8])
        if self.prev_kps[1] is not None:
            wvb = limb_velocity(kps_b, self.prev_kps[1], [9, 10])
            evb = limb_velocity(kps_b, self.prev_kps[1], [7, 8])
        fwa = wva > WRIST_VELOCITY_THRESHOLD
        fwb = wvb > WRIST_VELOCITY_THRESHOLD
        sig['wrist_vel'] = (round(wva, 1), round(wvb, 1))
        if fwa or fwb:
            score += 2
        elif eva > ELBOW_VELOCITY_THRESHOLD or evb > ELBOW_VELOCITY_THRESHOLD:
            score += 1

        # S2 — wrist invasion
        size_ok = max(ha, hb) / (min(ha, hb) + 1e-6) < 1.5
        inv_a   = size_ok and any(point_in_bbox(get_kp(kps_a, i), bb) for i in [9, 10])
        inv_b   = size_ok and any(point_in_bbox(get_kp(kps_b, i), ba) for i in [9, 10])
        sig['wrist_invasion'] = (inv_a, inv_b)
        if inv_a or inv_b:
            score += 2

        # S3 — facing
        if facing_each_other(kps_a, kps_b):
            score += 1

        # S4 — pose instability (arm angle variance)
        va = arm_angle_variance(kps_a, self.angle_buf[0])
        vb = arm_angle_variance(kps_b, self.angle_buf[1])
        if va > POSE_INSTABILITY_THRESH or vb > POSE_INSTABILITY_THRESH:
            score += 1

        # S5 — head proximity
        na, nb = get_kp(kps_a, 0), get_kp(kps_b, 0)
        if na and nb:
            hd = ((na[0]-nb[0])**2 + (na[1]-nb[1])**2)**0.5
            hr = hd / ((ha + hb) / 2 + 1e-6)
            if hr < HEAD_PROXIMITY_RATIO:
                score += 1

        # S6 — kick + ankle invasion
        kva = kvb = 0.0
        if self.prev_kps[0] is not None:
            kva = limb_velocity(kps_a, self.prev_kps[0], [13, 14, 15, 16])
        if self.prev_kps[1] is not None:
            kvb = limb_velocity(kps_b, self.prev_kps[1], [13, 14, 15, 16])
        fk         = kva > KNEE_VELOCITY_THRESHOLD or kvb > KNEE_VELOCITY_THRESHOLD
        ankle_inv  = False
        if ANKLE_INVASION_ENABLED and size_ok:
            lba, lbb = self._lower_bbox(kps_a), self._lower_bbox(kps_b)
            for idx in [15, 16]:
                if point_in_bbox(get_kp(kps_a, idx), lbb): ankle_inv = True
                if point_in_bbox(get_kp(kps_b, idx), lba): ankle_inv = True
        if fk and ankle_inv: score += 2
        elif fk:             score += 1

        # S7 — strike burst (wrist acceleration)
        burst = False
        for vn, pi in [(wva, 0), (wvb, 1)]:
            pv = self.prev_wrist_vel[pi]
            if (vn > WRIST_VELOCITY_THRESHOLD and pv > 0
                    and vn > pv * ACCELERATION_MULTIPLIER):
                burst = True
        if burst:
            score += 1

        # S8 — torso push
        ta, tb = self._hip_mid(kps_a), self._hip_mid(kps_b)
        tva = tvb = 0.0
        if ta and self.prev_torso[0]:
            tva = ((ta[0]-self.prev_torso[0][0])**2 +
                   (ta[1]-self.prev_torso[0][1])**2)**0.5
        if tb and self.prev_torso[1]:
            tvb = ((tb[0]-self.prev_torso[1][0])**2 +
                   (tb[1]-self.prev_torso[1][1])**2)**0.5
        pushed = ((tva > TORSO_VELOCITY_THRESHOLD and tvb < TORSO_ACCEL_THRESHOLD) or
                  (tvb > TORSO_VELOCITY_THRESHOLD and tva < TORSO_ACCEL_THRESHOLD))
        if pushed:
            score += 1

        # S9 — mutual aggression
        if fwa and fwb:
            score += MUTUAL_AGGRESSION_BONUS

        # S10 — grapple (IoU delta)
        cur_iou = compute_iou(ba, bb)
        iou_d   = cur_iou - self.prev_iou
        if cur_iou >= GRAPPLE_IOU_THRESHOLD and iou_d >= GRAPPLE_IOU_DELTA:
            score += 1

        # S11 — asymmetric aggression
        if ((inv_a and wvb < VICTIM_STATIONARY_THRESHOLD) or
                (inv_b and wva < VICTIM_STATIONARY_THRESHOLD)):
            score += 1

        # EMA smooth
        self.smoothed_score = (SCORE_SMOOTHING_ALPHA * self.smoothed_score +
                               (1 - SCORE_SMOOTHING_ALPHA) * score)
        eff = int(round(self.smoothed_score))
        sig['raw_score'] = score
        sig['score']     = eff

        self.prev_kps       = [kps_a, kps_b]
        self.prev_wrist_vel = [wva, wvb]
        self.prev_iou       = cur_iou
        self.prev_torso     = [ta, tb]
        return eff, sig


# ─── Rolling time-window ──────────────────────────────────────────────────────

class RollingFightWindow:
    """
    Tracks what fraction of frames in the last FIGHT_WINDOW_SECONDS had a
    score >= FIGHT_SCORE_THRESHOLD.  Returns True when ratio >= FIGHT_WINDOW_RATIO.
    """

    def __init__(self):
        self._entries    = deque()
        self.window_ratio = 0.0
        self.window_size  = 0

    def update(self, score):
        now = time.time()
        self._entries.append((now, score >= FIGHT_SCORE_THRESHOLD))
        cutoff = now - FIGHT_WINDOW_SECONDS
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()
        total = len(self._entries)
        if total == 0:
            self.window_ratio = 0.0
            self.window_size  = 0
            return False
        high              = sum(1 for _, h in self._entries if h)
        self.window_ratio = high / total
        self.window_size  = total
        return self.window_ratio >= FIGHT_WINDOW_RATIO

    def reset(self):
        self._entries.clear()
        self.window_ratio = 0.0
        self.window_size  = 0


# ─── Per-pair fight state machine ─────────────────────────────────────────────

class FightTracker:
    """
    Wraps FightScorer + RollingFightWindow into a state machine per person pair.

    States:  Normal → Interaction → Possible fight → FIGHT DETECTED
    """

    GATE_FAIL_TOLERANCE = 3   # frames where depth/proximity gate fails before reset
    CALM_STREAK_REQUIRED = 4  # consecutive low-score frames before returning to Normal

    def __init__(self, pair_id):
        self.pair_id          = pair_id
        self.scorer           = FightScorer()
        self.window           = RollingFightWindow()
        self.state            = 'Normal'
        self.confirmed        = False
        self.fight_start_time = None
        self.last_seen        = time.time()
        self.last_alerted     = 0.0
        self._gate_fail_streak = 0
        self._calm_streak      = 0

    def update(self, kps_list):
        self.last_seen = time.time()
        score, sig     = self.scorer.update(kps_list)

        if sig.get('skip_reason') is not None:
            self._gate_fail_streak += 1
            if self._gate_fail_streak >= self.GATE_FAIL_TOLERANCE:
                self._reset()
            return self.state, sig

        self._gate_fail_streak = 0
        window_met = self.window.update(score)
        sig['window_ratio'] = round(self.window.window_ratio, 2)

        if self.confirmed:
            self.state        = 'FIGHT DETECTED'
            self._calm_streak = 0
            return self.state, sig

        now = time.time()
        if score >= FIGHT_SCORE_THRESHOLD:
            self._calm_streak = 0
            if self.fight_start_time is None:
                self.fight_start_time = now
            elapsed = now - self.fight_start_time
            if elapsed >= FIGHT_CONFIRM_SECONDS and window_met:
                self.confirmed = True
                self.state     = 'FIGHT DETECTED'
            else:
                self.state = f'Possible fight ({elapsed:.1f}s)'
        elif score >= POSSIBLE_FIGHT_SCORE:
            self._calm_streak = 0
            self.state        = 'Interaction'
        else:
            self._calm_streak += 1
            if self._calm_streak >= self.CALM_STREAK_REQUIRED:
                self.fight_start_time = None
                self.state            = 'Normal'

        return self.state, sig

    def _reset(self):
        self.window.reset()
        self.confirmed        = False
        self.fight_start_time = None
        self.state            = 'Normal'
        self._gate_fail_streak = 0
        self._calm_streak      = 0
        self.scorer.smoothed_score  = 0.0
        self.scorer._kp_smooth      = [None, None]

    def alert_ready(self):
        return time.time() - self.last_alerted >= TELEGRAM_COOLDOWN

    def mark_alerted(self):
        self.last_alerted = time.time()

    @property
    def timer_elapsed(self):
        return (time.time() - self.fight_start_time) if self.fight_start_time else 0.0
