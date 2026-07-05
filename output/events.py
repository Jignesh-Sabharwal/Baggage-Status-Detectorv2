"""Event log (CSV) + session ledger with defined t_end (REQ-06/16/17/31)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from activity.activity_fsm import ActivityState


@dataclass
class Session:
    session_id: int
    type: str  # "LOADING" | "UNLOADING"
    t_start: float
    frame_start: int
    truncated_start: bool = False
    t_last_track: float = 0.0
    bag_count: int = 0
    _counted_track_ids: set = field(default_factory=set)
    # Set while a reposition holds this session open (REQ-16); cleared on resume or close.
    holding_since_reposition: bool = False


class EventLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []
        self._sessions: list[dict] = []
        self._event_id = 0

    def log_event(self, frame_idx: int, t_video: float, event_type: str, scene_state: str,
                  activity_state: str, n_tracks: int) -> int:
        self._event_id += 1
        self._events.append({
            "event_id": self._event_id, "event_type": event_type, "video_time": round(t_video, 3),
            "frame": frame_idx, "scene_state": scene_state, "activity_state": activity_state,
            "n_tracks": n_tracks,
        })
        return self._event_id

    def log_session_close(self, session: Session, t_closed: float) -> None:
        self._sessions.append({
            "session_id": session.session_id, "type": session.type,
            "t_start": round(session.t_start, 3), "t_end": round(session.t_last_track, 3),
            "t_closed": round(t_closed, 3), "duration_s": round(session.t_last_track - session.t_start, 3),
            "bag_count": session.bag_count, "truncated_start": session.truncated_start,
        })

    def write(self) -> None:
        events_path = self.run_dir / "events.csv"
        with open(events_path, "w", newline="") as f:
            fieldnames = ["event_id", "event_type", "video_time", "frame", "scene_state", "activity_state", "n_tracks"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self._events)

        sessions_path = self.run_dir / "sessions.csv"
        with open(sessions_path, "w", newline="") as f:
            fieldnames = ["session_id", "type", "t_start", "t_end", "t_closed", "duration_s",
                          "bag_count", "truncated_start"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self._sessions)


class SessionManager:
    def __init__(self, reposition_grace_s: float):
        self.reposition_grace_s = reposition_grace_s
        self.current: Session | None = None
        self._next_id = 1
        self._held_open_deadline: float | None = None

    def on_activity_transition(self, new_state: ActivityState, t_video: float, frame_idx: int,
                                logger: EventLogger, scene_state: str, n_tracks: int,
                                truncated_start: bool = False) -> None:
        if new_state == ActivityState.IDLE:
            if self.current is not None:
                logger.log_session_close(self.current, t_video)
                self.current = None
                self._held_open_deadline = None
            return

        session_type = "LOADING" if new_state == ActivityState.LOADING else "UNLOADING"
        if self.current is None:
            self.current = Session(session_id=self._next_id, type=session_type, t_start=t_video,
                                    frame_start=frame_idx, truncated_start=truncated_start,
                                    t_last_track=t_video)
            self._next_id += 1
        elif self.current.type != session_type:
            # Direct LOADING<->UNLOADING flip: close the old, open a new (REQ-31 t_end via
            # last-track time, not this transition time).
            logger.log_session_close(self.current, t_video)
            self.current = Session(session_id=self._next_id, type=session_type, t_start=t_video,
                                    frame_start=frame_idx, t_last_track=t_video)
            self._next_id += 1
        self._held_open_deadline = None

    def on_confirmed_track_sighting(self, t_video: float, track_id: int, vote: int) -> None:
        if self.current is None:
            return
        self.current.t_last_track = t_video
        matches_direction = (vote > 0) == (self.current.type == "LOADING")
        if matches_direction and track_id not in self.current._counted_track_ids:
            self.current._counted_track_ids.add(track_id)
            self.current.bag_count += 1

    def on_reposition(self, t_video: float) -> None:
        """REQ-16: hold any open session open, bookmarked at last confirmed-track sighting."""
        if self.current is not None:
            self.current.holding_since_reposition = True
            self._held_open_deadline = self.current.t_last_track + self.reposition_grace_s

    def check_grace_and_maybe_close(self, t_video: float, logger: EventLogger) -> None:
        if self.current is not None and self.current.holding_since_reposition and self._held_open_deadline is not None:
            if t_video > self._held_open_deadline:
                logger.log_session_close(self.current, t_video)
                self.current = None
                self._held_open_deadline = None

    def on_resume_same_direction(self, new_state: ActivityState) -> bool:
        """Call right before on_activity_transition when resuming from a reposition hold —
        returns True if the held-open session continues seamlessly (REQ-16)."""
        if self.current is None or not self.current.holding_since_reposition:
            return False
        session_type = "LOADING" if new_state == ActivityState.LOADING else "UNLOADING"
        if session_type == self.current.type:
            self.current.holding_since_reposition = False
            self._held_open_deadline = None
            return True
        return False

    def close_at_teardown(self, t_video: float, logger: EventLogger) -> None:
        """REQ-17: teardown from SETTLING (or MONITORING) closes any held-open session with
        t_end = t_last_track — never the teardown time."""
        if self.current is not None:
            logger.log_session_close(self.current, t_video)
            self.current = None
            self._held_open_deadline = None
