"""Tests for the roadmap phase map (tracera/phase_map.py)."""

from tracera.phase_map import (
    PHASES,
    STATUS_EXCLUDED,
    STATUS_IMPLEMENTED,
    STATUS_ROADMAP,
    counts,
    get_phase,
    implemented,
    roadmap,
)


def test_all_phases_1_to_72_present():
    numbers = [p.number for p in PHASES]
    assert numbers == list(range(1, 73))


def test_implemented_phases_are_1_40_and_42_59():
    impl = {p.number for p in implemented()}
    assert impl == {*range(1, 41), *range(42, 60)}


def test_excluded_and_roadmap_statuses():
    excl = {p.number for p in PHASES if p.status == STATUS_EXCLUDED}
    assert excl == {41, *range(60, 67)}
    rd = {p.number for p in PHASES if p.status == STATUS_ROADMAP}
    assert rd == set(range(67, 73))


def test_counts():
    c = counts()
    assert c[STATUS_IMPLEMENTED] == 58
    assert c[STATUS_EXCLUDED] == 8
    assert c[STATUS_ROADMAP] == 6


def test_get_phase_roundtrip():
    p = get_phase(36)
    assert p is not None and p.number == 36 and p.status == STATUS_IMPLEMENTED
    assert get_phase(41).status == STATUS_EXCLUDED
    assert get_phase(70).status == STATUS_ROADMAP
    assert get_phase(999) is None


def test_roadmap_has_static_analysis_phases():
    titles = {p.number: p.title for p in roadmap()}
    assert "Real call-graph resolution engine" in titles[67]
    assert "Language #2" in titles[72]
