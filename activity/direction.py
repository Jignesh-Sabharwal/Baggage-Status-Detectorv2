"""Axis projection, signed velocity, per-track vote (feeds the Activity FSM's M-of-N)."""
from __future__ import annotations

from config import DirConfig
from track.tracker import Track


def track_vote(track: Track, cfg: DirConfig, axis_unit: tuple[float, float], px_scale: float = 1.0) -> int:
    """+1 = LOADING (toward aircraft), -1 = UNLOADING, 0 = not eligible to vote this frame."""
    if len(track.history) < 2 or track.velocity_ema is None:
        return 0
    ax, ay = axis_unit
    net = (track.history[-1][0] - track.history[0][0], track.history[-1][1] - track.history[0][1])
    net_axis_disp = net[0] * ax + net[1] * ay
    if abs(net_axis_disp) < cfg.min_disp_px * px_scale:
        return 0
    if abs(track.velocity_ema) < cfg.min_speed:
        return 0
    return 1 if track.velocity_ema > 0 else -1


def frame_vote(tracks: list[Track], cfg: DirConfig, axis_unit: tuple[float, float], px_scale: float = 1.0) -> int | None:
    """Majority of eligible per-track votes; None if no track is eligible this frame."""
    votes = [track_vote(t, cfg, axis_unit, px_scale) for t in tracks]
    votes = [v for v in votes if v != 0]
    if not votes:
        return None
    total = sum(votes)
    if total == 0:  # exact tie
        return None
    return 1 if total > 0 else -1
