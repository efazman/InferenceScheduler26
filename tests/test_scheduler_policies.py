"""Policy ordering, tie-breaking and edge cases (pure functions, no engine)."""

import random

import pytest

from scheduler.models import Request
from scheduler.policies import AdaptivePolicy, FIFOPolicy, SEJFPolicy, make_policy


def q(rid, arrival, predicted, enqueue=None):
    return Request(rid, f"prompt {rid}", arrival, enqueue_time=arrival if enqueue is None else enqueue,
                   predicted_tokens=predicted)


ALL = [FIFOPolicy(), SEJFPolicy(), AdaptivePolicy(10_000)]


@pytest.mark.parametrize("policy", ALL, ids=lambda p: p.name)
def test_empty_queue_returns_none(policy):
    assert policy.choose_next([], now=0) is None
    assert policy.order([], now=0) == []


@pytest.mark.parametrize("policy", ALL, ids=lambda p: p.name)
def test_single_request_is_chosen(policy):
    r = q("r1", 0, 500)
    assert policy.choose_next([r], now=99_999) is r


def test_fifo_earliest_enqueued_first():
    queue = [q("r3", 30, 10), q("r1", 10, 900), q("r2", 20, 50)]
    assert [r.request_id for r in FIFOPolicy().order(queue, 100)] == ["r1", "r2", "r3"]


def test_fifo_tie_breaks_on_request_id():
    queue = [q("r2", 10, 1), q("r1", 10, 1)]
    assert FIFOPolicy().choose_next(queue, 50).request_id == "r1"


def test_sejf_chooses_shortest_prediction():
    queue = [q("long", 0, 900), q("short", 50, 40), q("mid", 10, 300)]
    assert [r.request_id for r in SEJFPolicy().order(queue, 100)] == ["short", "mid", "long"]


def test_sejf_equal_cost_falls_back_to_enqueue_then_id():
    queue = [q("r9", 30, 100), q("r5", 10, 100), q("r1", 10, 100)]
    assert [r.request_id for r in SEJFPolicy().order(queue, 100)] == ["r1", "r5", "r9"]


def test_adaptive_matches_sejf_when_nothing_is_overdue():
    rng = random.Random(0)
    for _ in range(50):
        queue = [q(f"r{i:02d}", rng.uniform(0, 5000), rng.choice([40, 40, 300, 900])) for i in range(12)]
        now = 6000  # every wait < 10 s
        assert AdaptivePolicy(10_000).order(queue, now) == SEJFPolicy().order(queue, now)


def test_adaptive_promotes_overdue_request():
    long_ = q("long", arrival=0, predicted=900)
    short = q("short", arrival=11_000, predicted=40)
    p = AdaptivePolicy(max_wait_ms=10_000)
    assert p.choose_next([long_, short], now=9_999).request_id == "short"  # not overdue yet
    assert p.choose_next([long_, short], now=10_000).request_id == "long"  # exactly at threshold
    assert p.reason(long_, 10_000) == "overdue" and p.reason(short, 10_000) == "shortest_estimate"


def test_adaptive_serves_overdue_oldest_first_then_sejf():
    queue = [q("od-new", 4_000, 10), q("od-old", 1_000, 999), q("fresh-long", 19_000, 800), q("fresh-short", 19_500, 5)]
    order = [r.request_id for r in AdaptivePolicy(10_000).order(queue, now=20_000)]
    assert order == ["od-old", "od-new", "fresh-short", "fresh-long"]


def test_policies_are_deterministic_under_shuffled_input():
    base = [q(f"r{i:02d}", i * 10 % 70, [5, 50, 50, 500][i % 4]) for i in range(20)]
    for p in ALL:
        expected = [r.request_id for r in p.order(base, 30_000)]
        for seed in range(5):
            shuffled = base[:]
            random.Random(seed).shuffle(shuffled)
            assert [r.request_id for r in p.order(shuffled, 30_000)] == expected


def test_make_policy_and_validation():
    assert make_policy("FIFO").name == "fifo" and make_policy("adaptive", 5000).max_wait_ms == 5000
    with pytest.raises(ValueError):
        make_policy("lottery")
    with pytest.raises(ValueError):
        AdaptivePolicy(0)
