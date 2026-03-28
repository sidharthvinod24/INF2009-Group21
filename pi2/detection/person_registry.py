"""
detection/person_registry.py
=============================
Assigns stable person IDs across frames via centroid matching.
Manages per-person FallTrackers and per-pair FightTrackers.
"""

import time
from itertools import combinations

from config.settings import PERSON_TIMEOUT, CENTROID_MATCH_DIST, PROXIMITY_THRESHOLD
from detection.pose_utils import person_centroid, bbox_from_kps, normalised_distance
from detection.fall_detector import FallTracker
from detection.fight_detector import FightTracker


class PersonRegistry:
    """
    Central registry that maps detected pose keypoints to stable person IDs.

    On each inference frame:
        assignments = registry.match(kps_list)
            -> [(pid, FallTracker, kps), ...]

        fight_pairs = registry.get_fight_pairs(assignments)
            -> [([kps_a, kps_b], FightTracker), ...]
    """

    def __init__(self):
        self._fall_trackers  = {}   # pid -> FallTracker
        self._fight_trackers = {}   # frozenset({pid_a, pid_b}) -> FightTracker
        self._centroids      = {}   # pid -> (cx, cy)
        self._next_id        = 1
        self._last_seen      = 0.0

    # ── Person tracking ───────────────────────────────────────────────────────

    def match(self, kps_list):
        """
        Match detected people to existing tracked IDs via centroid distance.
        Creates new FallTrackers for new IDs.
        Prunes tracks that have not been seen for PERSON_TIMEOUT seconds.

        Returns: [(pid, FallTracker, kps), ...]
        """
        now = time.time()
        self._last_seen = now
        new_centroids   = [person_centroid(k) for k in kps_list]
        existing        = list(self._fall_trackers.keys())

        # Build distance edges and sort — greedy optimal assignment for small N
        edges = []
        for ni, nc in enumerate(new_centroids):
            if nc is None:
                continue
            for pid in existing:
                ec = self._centroids.get(pid)
                if ec is None:
                    continue
                d = ((nc[0]-ec[0])**2 + (nc[1]-ec[1])**2)**0.5
                if d < CENTROID_MATCH_DIST:
                    edges.append((d, ni, pid))
        edges.sort()

        matched_new, matched_old = set(), set()
        matched = {}
        for d, ni, pid in edges:
            if ni in matched_new or pid in matched_old:
                continue
            matched[ni] = pid
            matched_new.add(ni)
            matched_old.add(pid)

        results = []
        for ni, kps in enumerate(kps_list):
            nc = new_centroids[ni]
            if ni in matched:
                pid = matched[ni]
            else:
                pid = self._next_id
                self._next_id += 1
                self._fall_trackers[pid] = FallTracker(pid)
            if nc is not None:
                self._centroids[pid] = nc
            self._fall_trackers[pid].last_seen = now
            results.append((pid, self._fall_trackers[pid], kps))

        # Prune stale person tracks
        stale = [p for p, t in self._fall_trackers.items()
                 if now - t.last_seen > PERSON_TIMEOUT]
        for p in stale:
            del self._fall_trackers[p]
            self._centroids.pop(p, None)

        return results

    # ── Fight pair management ─────────────────────────────────────────────────

    def get_fight_pairs(self, assignments):
        """
        Create / retrieve FightTrackers for every nearby person pair.
        Pairs that are too far apart (> 1.5× PROXIMITY_THRESHOLD) are skipped.

        Returns: [([kps_a, kps_b], FightTracker), ...]
        """
        if len(assignments) < 2:
            return []

        pairs = []
        for (pa, _, ka), (pb, _, kb) in combinations(assignments, 2):
            ba, bb = bbox_from_kps(ka), bbox_from_kps(kb)
            if ba is None or bb is None:
                continue
            if normalised_distance(ba, bb) > PROXIMITY_THRESHOLD * 1.5:
                continue
            key = frozenset([pa, pb])
            if key not in self._fight_trackers:
                self._fight_trackers[key] = FightTracker(
                    pair_id=hash(key) % 10000)
            pairs.append(([ka, kb], self._fight_trackers[key]))

        # Prune stale fight trackers
        now   = time.time()
        stale = [k for k, t in self._fight_trackers.items()
                 if now - t.last_seen > PERSON_TIMEOUT]
        for k in stale:
            del self._fight_trackers[k]

        return pairs

    # ── Idle reset ────────────────────────────────────────────────────────────

    def reset_idle(self):
        """Reset fight trackers if no person has been seen recently."""
        if time.time() - self._last_seen > PERSON_TIMEOUT:
            for t in self._fight_trackers.values():
                t._reset()
            self._fight_trackers.clear()
