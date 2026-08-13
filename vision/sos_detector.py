"""
SOS / Distress-Gesture Detection — hand open/close pattern.

Gesture
-------
**Open and close hand repeatedly** (default: 3 cycles within 5 seconds).

The detector tracks hand state transitions:
  OPEN → CLOSED → OPEN → CLOSED → OPEN → CLOSED  =  3 cycles

State machine
-------------
::

    NORMAL  ──pattern started──►  WARNING  ──cycles complete──►  CRITICAL
       ▲                              │                              │
       └──────timeout / reset─────────┘                              │
       └──────────────────────cooldown expires────────────────────────┘

When the state reaches CRITICAL:
  1. ``just_triggered`` is set to ``True`` for exactly one frame.
  2. A cooldown timer starts (default 15 s).
  3. No new alert can fire until the cooldown expires.
"""

import logging
import time
from collections import deque
from enum import Enum
from typing import Optional, List

import config
from vision.hand_detector import HandDetector

logger = logging.getLogger(__name__)


class SOSState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class SOSDetector:
    """SOS gesture detector: hand open/close pattern with temporal tracking."""

    def __init__(self):
        self.state: SOSState = SOSState.NORMAL
        self.warning_start: Optional[float] = None
        self.last_alert_time: float = 0.0
        self.just_triggered: bool = False

        # Hand state tracking
        self._prev_hand_open: Optional[bool] = None  # True=open, False=closed
        self._transitions: deque = deque()  # timestamps of open↔closed transitions
        self._cycle_count: int = 0

    # -----------------------------------------------------------------
    # Hand gesture detection
    # -----------------------------------------------------------------
    def _check_hand_pattern(self, hand_landmarks_list: List) -> bool:
        """
        Check if any detected hand is performing the open/close pattern.

        A "cycle" is one complete OPEN → CLOSED transition.
        The gesture is confirmed when the required number of cycles
        occur within the time window.

        Parameters
        ----------
        hand_landmarks_list : list
            List of hand landmark lists from HandDetector.detect().

        Returns
        -------
        bool
            True if the pattern is complete (enough cycles in time window).
        """
        if not hand_landmarks_list:
            return False

        now = time.time()

        # Check the first detected hand
        hand_lm = hand_landmarks_list[0]
        hand_open = HandDetector.is_hand_open(hand_lm)
        hand_closed = HandDetector.is_hand_closed(hand_lm)

        # Determine current state
        if hand_open:
            current_open = True
        elif hand_closed:
            current_open = False
        else:
            # Hand is in ambiguous state — ignore
            return False

        # Detect state transition
        if self._prev_hand_open is not None and current_open != self._prev_hand_open:
            self._transitions.append(now)
            logger.debug(
                "Hand transition: %s → %s  (total transitions: %d)",
                "OPEN" if self._prev_hand_open else "CLOSED",
                "OPEN" if current_open else "CLOSED",
                len(self._transitions),
            )

        self._prev_hand_open = current_open

        # Prune old transitions outside the time window
        window_start = now - config.SOS_HAND_WINDOW_SECONDS
        while self._transitions and self._transitions[0] < window_start:
            self._transitions.popleft()

        # Each cycle = 2 transitions (open→closed + closed→open, or vice versa)
        # But we count open→closed as one cycle for simplicity
        cycles = len(self._transitions) // 2
        self._cycle_count = cycles

        return cycles >= config.SOS_HAND_CYCLES_REQUIRED

    # -----------------------------------------------------------------
    # State-machine update (call once per frame)
    # -----------------------------------------------------------------
    def update(self, hand_landmarks_list: List) -> SOSState:
        """
        Advance the SOS state machine by one frame.

        Sets ``self.just_triggered = True`` on the exact frame that the
        state first transitions to CRITICAL.

        Parameters
        ----------
        hand_landmarks_list : list
            Output of :meth:`HandDetector.detect`.

        Returns
        -------
        SOSState
        """
        self.just_triggered = False
        now = time.time()
        in_cooldown = (now - self.last_alert_time) < config.SOS_COOLDOWN_SECONDS
        pattern_complete = self._check_hand_pattern(hand_landmarks_list)

        if self.state == SOSState.NORMAL:
            if not in_cooldown and self._cycle_count > 0:
                # Pattern started — enter warning
                self.state = SOSState.WARNING
                self.warning_start = now

            if pattern_complete and not in_cooldown:
                # Immediate trigger if pattern already complete
                self.state = SOSState.CRITICAL
                self.just_triggered = True
                self.last_alert_time = now
                self._reset_tracking()

        elif self.state == SOSState.WARNING:
            if pattern_complete and not in_cooldown:
                self.state = SOSState.CRITICAL
                self.just_triggered = True
                self.last_alert_time = now
                self._reset_tracking()
            elif self._cycle_count == 0:
                # No recent transitions — go back to normal
                elapsed = now - (self.warning_start or now)
                if elapsed > config.SOS_HAND_WINDOW_SECONDS:
                    self.state = SOSState.NORMAL
                    self.warning_start = None

        elif self.state == SOSState.CRITICAL:
            # Stay critical briefly, then reset
            elapsed = now - self.last_alert_time
            if elapsed > 3.0:  # Show critical state for 3 seconds
                self.state = SOSState.NORMAL
                self.warning_start = None

        return self.state

    def _reset_tracking(self):
        """Clear the transition history after a successful alert."""
        self._transitions.clear()
        self._cycle_count = 0
        self._prev_hand_open = None

    @property
    def warning_progress(self) -> float:
        """Fraction of required cycles completed (0.0 – 1.0)."""
        if config.SOS_HAND_CYCLES_REQUIRED == 0:
            return 0.0
        return min(1.0, self._cycle_count / config.SOS_HAND_CYCLES_REQUIRED)
