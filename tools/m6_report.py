"""M6 validation report: aggregate the latest run per clip into a summary table.

This reports what the pipeline itself observes (presence/dock/lock rates, session and bag
counts, event counts). It does NOT compute ground-truth boundary error or precision/recall —
README §7 step 1 requires manually scrubbing each clip to build a true session timeline, which
is a human annotation task outside what this automated session performed. Producing invented
numbers for those metrics would misrepresent the system's validated accuracy, so they are
left explicitly marked "not computed" rather than filled in.
"""
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLIPS = ["D01", "D02", "D03", "D04", "N01", "N02", "N03", "N04"]


def latest_run_dir(clip: str) -> Path | None:
    candidates = sorted(glob.glob(f"runs/{clip}_*"))
    return Path(candidates[-1]) if candidates else None


def summarize(run_dir: Path) -> dict:
    events_path = run_dir / "events.csv"
    sessions_path = run_dir / "sessions.csv"

    event_counts: dict[str, int] = {}
    last_frame = 0
    with open(events_path) as f:
        for row in csv.DictReader(f):
            event_counts[row["event_type"]] = event_counts.get(row["event_type"], 0) + 1
            last_frame = max(last_frame, int(row["frame"]))

    n_sessions = 0
    total_bags = 0
    with open(sessions_path) as f:
        for row in csv.DictReader(f):
            n_sessions += 1
            total_bags += int(row["bag_count"])

    roi_locks = event_counts.get("ROI_LOCKED", 0)
    first_lock_event = None
    with open(events_path) as f:
        for row in csv.DictReader(f):
            if row["event_type"] == "ROI_LOCKED":
                first_lock_event = float(row["video_time"])
                break

    return {
        "belt_arrived": event_counts.get("BELT_ARRIVED", 0),
        "belt_docked": event_counts.get("BELT_DOCKED", 0),
        "roi_locked": roi_locks,
        "roi_repositioned": event_counts.get("ROI_REPOSITIONED", 0),
        "roi_health_reset": event_counts.get("ROI_HEALTH_RESET", 0),
        "first_lock_t": first_lock_event,
        "n_sessions": n_sessions,
        "total_bags": total_bags,
        "loading_started": event_counts.get("LOADING_STARTED", 0),
        "unloading_started": event_counts.get("UNLOADING_STARTED", 0),
        "belt_covered": event_counts.get("BELT_COVERED", 0),
    }


def main():
    print("Zero-shot structural-config table (D01-tuned config, PER-CAMERA values only per clip):")
    print(f"{'clip':6s} {'belt_arr':>8s} {'docked':>7s} {'roi_lock':>9s} {'first_lock_t':>13s} "
          f"{'reposit':>8s} {'health_rst':>10s} {'sessions':>9s} {'bags':>5s} {'load':>5s} {'unload':>7s}")
    for clip in CLIPS:
        rd = latest_run_dir(clip)
        if rd is None:
            print(f"{clip:6s} (no run found)")
            continue
        s = summarize(rd)
        flt = f"{s['first_lock_t']:.1f}s" if s["first_lock_t"] is not None else "never"
        print(f"{clip:6s} {s['belt_arrived']:8d} {s['belt_docked']:7d} {s['roi_locked']:9d} {flt:>13s} "
              f"{s['roi_repositioned']:8d} {s['roi_health_reset']:10d} {s['n_sessions']:9d} "
              f"{s['total_bags']:5d} {s['loading_started']:5d} {s['unloading_started']:7d}")

    print()
    print("NOTE: session boundary error, precision/recall, and bag-count error against ground")
    print("truth are NOT computed here — README Sec7 step 1 requires a manual scrub of each")
    print("clip to build a true session timeline, which was not performed in this session.")
    print("These numbers describe pipeline *self-consistency* (did it detect, lock, and log")
    print("something plausible), not validated accuracy.")


if __name__ == "__main__":
    main()
