"""Metric logic on tiny hand-checkable cases."""
from __future__ import annotations

from testbed.metrics import RunTracker, clip_events


def test_spillback_needs_min_duration():
    # dt = 1 s; condition true for steps t=5..14 (10 s) on lane a, t=20..28 (9 s) on lane b
    tr = RunTracker(dt=1.0, min_duration=10.0)
    for t in range(1, 41):
        tr.update("a", t, 5 <= t <= 14, "J1")
        tr.update("b", t, 20 <= t <= 28, "J2")
    ev = tr.finalize(40)
    assert [(e.key, e.junction, e.start, e.end, e.duration) for e in ev] == [("a", "J1", 4.0, 14.0, 10.0)]


def test_open_event_closed_at_end_and_clipped():
    tr = RunTracker(dt=1.0, min_duration=10.0)
    for t in range(1, 101):
        tr.update("a", t, t >= 81)
    ev = tr.finalize(100)
    assert ev[0].start == 80 and ev[0].end == 100
    clipped = clip_events(ev, 0, 90)
    assert clipped[0].duration == 10
    assert clip_events(ev, 0, 80) == []


def test_two_separate_events_on_same_lane():
    tr = RunTracker(dt=2.0, min_duration=10.0)
    flags = [1] * 6 + [0] * 2 + [1] * 5  # 12 s, gap, 10 s
    for k, f in enumerate(flags, 1):
        tr.update("a", 2.0 * k, bool(f))
    ev = tr.finalize(2.0 * len(flags))
    assert [e.duration for e in ev] == [12.0, 10.0]
