"""Tests for the Phase 60 observability telemetry tracker."""

from tracera.observability import get_telemetry, reset_telemetry
from tracera.observability.telemetry import Telemetry


def test_telemetry_records_llm_calls():
    reset_telemetry()
    t = get_telemetry()
    t.record_llm(provider="groq", model="x", prompt_tokens=500, completion_tokens=250, latency_ms=100.0)
    t.record_llm(provider="groq", model="x", prompt_tokens=100, completion_tokens=50, latency_ms=50.0, error=True)

    snap = t.snapshot()
    llm = snap["llm"]
    assert llm["calls"] == 2
    assert llm["errors"] == 1
    assert llm["prompt_tokens"] == 600
    assert llm["completion_tokens"] == 300
    assert llm["total_tokens"] == 900


def test_telemetry_records_tools_and_categories():
    reset_telemetry()
    t = get_telemetry()
    t.record_tool(name="read_file", duration_ms=5.0, success=True)
    t.record_tool(name="run_command", duration_ms=20.0, success=False)
    t.record_tool(name="run_command", duration_ms=10.0, success=True)

    snap = t.snapshot()
    tools = snap["tools"]
    assert tools["calls"] == 3
    assert tools["failures"] == 1
    assert tools["per_tool"]["run_command"] == 2
    assert tools["by_category"]["read"] == 1
    assert tools["by_category"]["execute"] == 2


def test_telemetry_classifies_retrieval():
    t = get_telemetry()
    assert Telemetry.classify_tool("search_code") == "retrieval"
    assert Telemetry.classify_tool("find_symbol") == "retrieval"
    assert Telemetry.classify_tool("read_file") == "read"
    assert Telemetry.classify_tool("edit_file") == "write"
    assert Telemetry.classify_tool("unknown_tool") == "other"


def test_telemetry_retrieval_and_cost():
    reset_telemetry()
    t = get_telemetry()
    t.record_retrieval(kind="search_code")
    t.record_retrieval(kind="search_code")
    t.record_retrieval(kind="find_symbol")
    t.record_iteration()
    t.record_iteration()

    snap = t.snapshot()
    assert snap["retrieval"]["calls"] == 3
    assert snap["retrieval"]["by_kind"]["search_code"] == 2
    assert snap["agent"]["iterations"] == 2

    # 1M prompt tokens → $0.30, 1M completion → $1.20 = $1.50
    t.record_llm(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    snap = t.snapshot()
    assert abs(snap["cost"]["estimate_usd"] - 1.50) < 1e-6


def test_telemetry_reset():
    reset_telemetry()
    t = get_telemetry()
    t.record_iteration()
    assert t.snapshot()["agent"]["iterations"] == 1
    t.reset()
    assert t.snapshot()["agent"]["iterations"] == 0