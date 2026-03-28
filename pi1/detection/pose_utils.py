"""
detection/pose_utils.py
========================
Shared low-level helpers for working with YOLO pose keypoints.

COCO keypoint index reference (17 points):
  0  Nose           1  L-Eye        2  R-Eye
  3  L-Ear          4  R-Ear
  5  L-Shoulder     6  R-Shoulder
  7  L-Elbow        8  R-Elbow
  9  L-Wrist       10  R-Wrist
 11  L-Hip         12  R-Hip
 13  L-Knee        14  R-Knee
 15  L-Ankle       16  R-Ankle
"""

import cv2
import numpy as np


# ─── Keypoint access ─────────────────────────────────────────────────────────

def get_kp(kps, idx):
    """Return [x, y] for keypoint idx, or None if missing/zero."""
    if idx >= len(kps):
        return None
    x, y = float(kps[idx][0]), float(kps[idx][1])
    return None if (x == 0 and y == 0) else [x, y]


def keypoints_snapshot(kps):
    """Return {idx: (x, y)} dict for all visible keypoints."""
    snap = {}
    for i in range(17):
        pt = get_kp(kps, i)
        if pt:
            snap[i] = (pt[0], pt[1])
    return snap


def person_centroid(kps):
    """Best-effort centroid: hips > shoulders > any visible keypoint."""
    for pair in [(11, 12), (5, 6)]:
        a, b = get_kp(kps, pair[0]), get_kp(kps, pair[1])
        if a and b:
            return ((a[0]+b[0])/2, (a[1]+b[1])/2)
        if a: return tuple(a)
        if b: return tuple(b)
    for i in range(17):
        pt = get_kp(kps, i)
        if pt:
            return tuple(pt)
    return None


# ─── Bounding box helpers ─────────────────────────────────────────────────────

def bbox_from_kps(kps):
    """Axis-aligned bounding box from all visible keypoints."""
    pts = [get_kp(kps, i) for i in range(len(kps))]
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_height(b):
    return b[3] - b[1] if b else 1


def bbox_width(b):
    return b[2] - b[0] if b else 1


def bbox_centre(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)


def compute_iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def point_in_bbox(pt, bbox):
    if pt is None or bbox is None:
        return False
    return bbox[0] <= pt[0] <= bbox[2] and bbox[1] <= pt[1] <= bbox[3]


def normalised_distance(ba, bb):
    """Euclidean centre-to-centre distance normalised by average bbox height."""
    cxa, cya = bbox_centre(ba)
    cxb, cyb = bbox_centre(bb)
    dist = ((cxa-cxb)**2 + (cya-cyb)**2) ** 0.5
    avg_h = (bbox_height(ba) + bbox_height(bb)) / 2
    return dist / (avg_h + 1e-6)


def shoulder_width_px(kps):
    ls, rs = get_kp(kps, 5), get_kp(kps, 6)
    if ls is None or rs is None:
        return None
    return abs(rs[0] - ls[0])


# ─── Motion helpers ───────────────────────────────────────────────────────────

def limb_velocity(kps_now, kps_prev, indices):
    """Mean pixel displacement of the given keypoint indices between two frames."""
    vels = []
    for idx in indices:
        a, b = get_kp(kps_now, idx), get_kp(kps_prev, idx)
        if a and b:
            vels.append(((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5)
    return float(np.mean(vels)) if vels else 0.0


def mean_displacement(snap_old, snap_new):
    """Mean pixel displacement of common keypoints between two snapshots."""
    disps = []
    for idx in snap_old:
        if idx in snap_new:
            ox, oy = snap_old[idx]
            nx, ny = snap_new[idx]
            disps.append(((nx-ox)**2 + (ny-oy)**2) ** 0.5)
    return float(np.mean(disps)) if disps else 0.0


# ─── Angle helpers ────────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    """Angle at vertex b formed by points a-b-c, in degrees."""
    bax, bay = a[0]-b[0], a[1]-b[1]
    bcx, bcy = c[0]-b[0], c[1]-b[1]
    n_ba = (bax**2 + bay**2)**0.5
    n_bc = (bcx**2 + bcy**2)**0.5
    if n_ba == 0 or n_bc == 0:
        return None
    cos_v = (bax*bcx + bay*bcy) / (n_ba * n_bc)
    cos_v = max(-1.0, min(1.0, cos_v))
    return float(np.degrees(np.arccos(cos_v)))


# ─── Drawing ──────────────────────────────────────────────────────────────────

_SKEL_CONNS = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (0, 5), (0, 6)
]


def state_color(state):
    """Map a detector state string to a BGR colour."""
    if 'MOTIONLESS AFTER FALL' in state: return (0, 0, 180)
    if 'FIGHT'       in state:           return (0, 0, 255)
    if 'FALL'        in state:           return (0, 0, 255)
    if 'Possible'    in state:           return (0, 100, 255)
    if 'Motionless'  in state:           return (180, 0, 180)
    if 'Lying'       in state or 'Horizontal' in state: return (0, 165, 255)
    if 'Interaction' in state:           return (0, 165, 255)
    return (0, 220, 0)


def draw_skeleton(frame, kps, color, label=None):
    """Draw skeleton with glow joints, AA lines, and an optional label pill."""
    pts_cache = {}
    for idx in range(min(len(kps), 17)):
        pt = get_kp(kps, idx)
        if pt:
            pts_cache[idx] = (int(pt[0]), int(pt[1]))

    for i, j in _SKEL_CONNS:
        if i in pts_cache and j in pts_cache:
            cv2.line(frame, pts_cache[i], pts_cache[j], color, 2, cv2.LINE_AA)

    for idx, pt in pts_cache.items():
        cv2.circle(frame, pt, 5, (0, 0, 0),  -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 3, color,       -1, cv2.LINE_AA)

    if len(pts_cache) >= 3:
        xs = [p[0] for p in pts_cache.values()]
        ys = [p[1] for p in pts_cache.values()]
        cv2.rectangle(frame,
                      (min(xs)-8, min(ys)-8),
                      (max(xs)+8, max(ys)+8),
                      color, 1, cv2.LINE_AA)

    if label and pts_cache:
        top_pt = min(pts_cache.values(), key=lambda p: p[1])
        lx, ly = top_pt[0] - 30, max(top_pt[1] - 18, 14)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(frame, (lx-2, ly-th-4), (lx+tw+4, ly+4), (0, 0, 0), -1)
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
