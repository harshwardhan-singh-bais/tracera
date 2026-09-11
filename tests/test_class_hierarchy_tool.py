"""Unit tests for the get_class_hierarchy tool (backing /classhier and MCP)."""

from __future__ import annotations

import asyncio

from tracera.graph.symbol_graph import RelationType, SymbolGraph
from tracera.indexer.schema import LineRange, Symbol, SymbolType
from tracera.tools.ast_tools import GetClassHierarchyTool


class _FakeGraphRetriever:
    def __init__(self, graph: SymbolGraph) -> None:
        self.graph = graph


def _sym(name: str, parent: str | None = None, stype: SymbolType = SymbolType.CLASS) -> Symbol:
    return Symbol(
        name=name,
        type=stype,
        range=LineRange(start_line=1, end_line=2),
        content="...",
        parent_symbol=parent,
    )


def _make_graph() -> SymbolGraph:
    g = SymbolGraph()
    base = g.add_symbol("base.py", _sym("Animal"))
    child = g.add_symbol("zoo.py", _sym("Dog", parent="Animal"))
    g.add_relation(child, base, RelationType.INHERITS)
    g.add_symbol("zoo.py", _sym("bark", parent="Dog", stype=SymbolType.METHOD))
    return g


def _tool(graph: SymbolGraph) -> GetClassHierarchyTool:
    return GetClassHierarchyTool((None,) * 9 + (_FakeGraphRetriever(graph),))


def test_base_classes_and_subclasses_follow_edge_direction():
    res = asyncio.run(_tool(_make_graph()).execute(class_name="Dog"))
    assert res.success
    assert "Animal" in res.output  # base class
    assert "No subclasses" in res.output
    assert "bark" in res.output  # contained member

    res2 = asyncio.run(_tool(_make_graph()).execute(class_name="Animal"))
    assert res2.success
    assert "Dog" in res2.output  # subclass
    assert "No base classes" in res2.output


def test_confidence_and_freshness_metadata_present():
    res = asyncio.run(_tool(_make_graph()).execute(class_name="Dog"))
    meta = res.metadata.get("_meta", {})
    assert meta.get("confidence", 0) >= 0.5
    assert "freshness" in meta


def test_missing_class_is_honest():
    res = asyncio.run(_tool(_make_graph()).execute(class_name="DoesNotExist"))
    assert res.success
    assert "not found in the indexed/analyzable corpus" in res.output


def test_non_class_symbol_is_rejected():
    g = _make_graph()
    g.add_symbol("f.py", _sym("helper", stype=SymbolType.FUNCTION))
    res = asyncio.run(_tool(g).execute(class_name="helper"))
    assert res.success
    assert "not a class" in res.output


def test_no_graph_degrades_gracefully():
    res = asyncio.run(GetClassHierarchyTool(None).execute(class_name="Dog"))
    assert res.success
    assert "Symbol graph not available" in res.output
