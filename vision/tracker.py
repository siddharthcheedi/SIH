"""
Person Tracker — gender-prediction smoothing on top of YOLO ByteTrack IDs.

YOLO's built-in ByteTrack already gives each person a persistent ID.
This module maintains a short history of gender predictions per tracked
person so that the displayed label does not flicker frame-to-frame.
"""

import logging
from collections import deque
from typing import Dict, List, Tuple

import config

logger = logging.getLogger(__name__)


class TrackedPerson:
    """State for one tracked person (keyed by ByteTrack ID)."""

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.gender_history: deque = deque(maxlen=config.GENDER_SMOOTHING_FRAMES)
        self.last_seen_frame: int = 0
        self.smoothed_gender: str = "Unknown"
        self.smoothed_confidence: float = 0.0

    def update_gender(self, gender: str, confidence: float):
        """Push a new per-frame prediction into the history buffer."""
        self.gender_history.append((gender, confidence))
        self._recompute()

    def _recompute(self):
        """Majority-vote across the history window."""
        if not self.gender_history:
            self.smoothed_gender = "Unknown"
            self.smoothed_confidence = 0.0
            return

        votes: Dict[str, int] = {"Male": 0, "Female": 0, "Unknown": 0}
        conf_sum: Dict[str, float] = {"Male": 0.0, "Female": 0.0, "Unknown": 0.0}

        for g, c in self.gender_history:
            votes[g] += 1
            conf_sum[g] += c

        male_v, female_v = votes["Male"], votes["Female"]

        if male_v == 0 and female_v == 0:
            self.smoothed_gender = "Unknown"
            self.smoothed_confidence = 0.0
            return

        if male_v > female_v:
            self.smoothed_gender = "Male"
            self.smoothed_confidence = conf_sum["Male"] / male_v
        elif female_v > male_v:
            self.smoothed_gender = "Female"
            self.smoothed_confidence = conf_sum["Female"] / female_v
        else:
            # Tie — pick by higher average confidence
            m_avg = conf_sum["Male"] / male_v if male_v else 0
            f_avg = conf_sum["Female"] / female_v if female_v else 0
            if m_avg >= f_avg:
                self.smoothed_gender = "Male"
                self.smoothed_confidence = m_avg
            else:
                self.smoothed_gender = "Female"
                self.smoothed_confidence = f_avg


class PersonTracker:
    """
    Maintains a dict of :class:`TrackedPerson` keyed by YOLO track ID.

    Call :meth:`update` once per detection per frame, then :meth:`tick`
    at the end of each frame to purge stale tracks.
    """

    def __init__(self, max_disappeared_frames: int = 90):
        self.tracks: Dict[int, TrackedPerson] = {}
        self.max_disappeared = max_disappeared_frames
        self.current_frame: int = 0

    def update(
        self,
        track_id: int,
        gender: str | None = None,
        confidence: float = 0.0,
    ) -> TrackedPerson:
        """
        Create or refresh a tracked person entry.

        If *gender* is not ``None``, it is appended to the person's
        prediction-history buffer for temporal smoothing.
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackedPerson(track_id)

        person = self.tracks[track_id]
        person.last_seen_frame = self.current_frame

        if gender is not None:
            person.update_gender(gender, confidence)

        return person

    def get_smoothed_gender(self, track_id: int) -> Tuple[str, float]:
        """Return the temporally-smoothed gender label and confidence."""
        if track_id in self.tracks:
            p = self.tracks[track_id]
            return p.smoothed_gender, p.smoothed_confidence
        return "Unknown", 0.0

    def tick(self):
        """Advance the frame counter and remove stale tracks."""
        self.current_frame += 1
        stale = [
            tid
            for tid, p in self.tracks.items()
            if (self.current_frame - p.last_seen_frame) > self.max_disappeared
        ]
        for tid in stale:
            del self.tracks[tid]

    def get_counts(self, active_ids: List[int]) -> Dict[str, int]:
        """
        Return ``{"total": …, "male": …, "female": …, "unknown": …}``
        for the persons currently visible (by their track IDs).
        """
        counts = {"total": 0, "male": 0, "female": 0, "unknown": 0}
        for tid in active_ids:
            counts["total"] += 1
            if tid in self.tracks:
                g = self.tracks[tid].smoothed_gender.lower()
                counts[g] = counts.get(g, 0) + 1
            else:
                counts["unknown"] += 1
        return counts
