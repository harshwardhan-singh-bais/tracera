"""Tests for the enhanced memory system (session, taxonomy, extractor, triples, recall)."""

import pytest
import time
import tempfile
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════


def test_session_lifecycle(tmp_path):
    """Create, populate, and close a session."""
    from tracera.memory.session import SessionManager, SessionTurn

    mgr = SessionManager(tmp_path / "mem")
    session = mgr.new_session(task="Fix login bug", entity_id="user1")
    assert session.task == "Fix login bug"
    assert session.entity_id == "user1"
    assert mgr.active_session is session

    mgr.record_user_message("Fix the login bug")
    mgr.record_tool_call("read_file", {"path": "auth.py"}, "def login(): ...", True, file_path="auth.py")
    mgr.record_agent_response("I found the bug in auth.py")

    assert session.turn_count == 3
    assert "auth.py" in session.files_touched
    assert session.tools_used["read_file"] == 1

    mgr.close_session(outcome="success", summary="Fixed null check in login()")
    assert session.is_closed
    assert session.outcome == "success"
    assert session.duration_seconds is not None


def test_session_persistence(tmp_path):
    """Sessions persist to disk and reload."""
    from tracera.memory.session import SessionManager

    mgr1 = SessionManager(tmp_path / "mem")
    s = mgr1.new_session(task="Test task")
    mgr1.record_user_message("hello")
    mgr1.close_session(outcome="success")

    mgr2 = SessionManager(tmp_path / "mem")
    assert len(mgr2.sessions) == 1
    assert mgr2.sessions[0].task == "Test task"


def test_session_find_similar(tmp_path):
    """Find sessions by task similarity."""
    from tracera.memory.session import SessionManager

    mgr = SessionManager(tmp_path / "mem")
    s1 = mgr.new_session(task="Fix authentication bug")
    mgr.close_session(outcome="success")
    s2 = mgr.new_session(task="Add user preferences")
    mgr.close_session(outcome="success")
    s3 = mgr.new_session(task="Fix login authentication")
    mgr.close_session(outcome="success")

    similar = mgr.find_similar_sessions("authentication issue", k=3)
    assert len(similar) >= 1
    # s1 and s3 should be more similar than s2
    task_words = [s.task for s in similar]
    assert any("authentication" in t.lower() for t in task_words)


def test_session_conversation_text(tmp_path):
    """Session renders conversation text for extraction."""
    from tracera.memory.session import SessionManager

    mgr = SessionManager(tmp_path / "mem")
    s = mgr.new_session(task="test")
    mgr.record_user_message("What is auth?")
    mgr.record_agent_response("Auth uses JWT")
    text = s.conversation_text()
    assert "User: What is auth?" in text
    assert "Assistant: Auth uses JWT" in text


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════


def test_memory_types_create_and_serialize():
    """All memory types can be created and serialized."""
    from tracera.memory.taxonomy import (
        create_fact, create_rule, create_relationship,
        create_skill, create_preference, create_event,
        MemoryType,
    )

    fact = create_fact("Project uses pytest", symbol="pytest", file_path="pyproject.toml")
    assert fact.memory_type == MemoryType.FACT
    assert fact.symbol == "pytest"
    d = fact.to_dict()
    assert d["memory_type"] == "fact"

    rule = create_rule("Always run tests after edits", priority=1)
    assert rule.memory_type == MemoryType.RULE
    assert rule.priority == 1

    rel = create_relationship("AuthMiddleware", "calls", "UserService")
    assert rel.subject == "AuthMiddleware"
    assert rel.predicate == "calls"
    assert rel.object == "UserService"

    skill = create_skill("Can parse pytest output", proficiency=0.8)
    assert skill.proficiency == 0.8

    pref = create_preference("User prefers TypeScript", strength=0.9)
    assert pref.strength == 0.9

    event = create_event("Fixed null check", event_type="fix", related_files=["auth.py"])
    assert event.event_type == "fix"
    assert "auth.py" in event.related_files


def test_memory_observe_increases_frequency():
    """Observing a memory increases its frequency and confidence."""
    from tracera.memory.taxonomy import create_fact

    fact = create_fact("test fact", confidence=0.7)
    assert fact.frequency == 1
    fact.observe()
    assert fact.frequency == 2
    assert fact.confidence > 0.7


def test_memory_serialization_roundtrip():
    """Memory entries survive serialization roundtrip."""
    from tracera.memory.taxonomy import create_fact, MemoryType, StructuredMemory

    fact = create_fact("Roundtrip test", confidence=0.9)
    d = fact.to_dict()
    restored = StructuredMemory.from_dict(d)
    assert restored.content == "Roundtrip test"
    assert restored.memory_type == MemoryType.FACT
    assert restored.confidence == 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════


def test_rule_based_extraction():
    """Rule-based extractor catches common patterns."""
    from tracera.memory.extractor import ConversationExtractor
    from tracera.memory.taxonomy import MemoryType

    extractor = ConversationExtractor(provider=None)  # no LLM → rule-based

    conversation = (
        "I prefer using pytest for testing\n"
        "Always run linter before committing\n"
        "The project uses FastAPI for the web server\n"
    )

    import asyncio
    memories = asyncio.run(extractor.extract(conversation, session_id="test"))
    assert len(memories) >= 2
    types = {m.memory_type for m in memories}
    # Should detect at least a preference and a rule/fact
    assert MemoryType.PREFERENCE in types or MemoryType.RULE in types or MemoryType.FACT in types


def test_empty_conversation_returns_nothing():
    """Empty conversation produces no memories."""
    from tracera.memory.extractor import ConversationExtractor

    extractor = ConversationExtractor(provider=None)
    import asyncio
    memories = asyncio.run(extractor.extract(""))
    assert memories == []


# ═══════════════════════════════════════════════════════════════════════════════
# TRIPLE STORE
# ═══════════════════════════════════════════════════════════════════════════════


def test_triple_store_add_and_query():
    """Add triples and query by subject/object."""
    from tracera.memory.triples import TripleStore, Triple

    store = TripleStore()
    store.add_triple(Triple("AuthMiddleware", "calls", "UserService"))
    store.add_triple(Triple("AuthMiddleware", "imports", "jwt"))
    store.add_triple(Triple("UserService", "queries", "UserRepository"))

    # Query by subject
    objs = store.get_objects("AuthMiddleware")
    assert len(objs) == 2
    obj_names = {t.object for t in objs}
    assert "UserService" in obj_names

    # Query by object
    subs = store.get_subjects("UserService")
    assert len(subs) == 1
    assert subs[0].subject == "AuthMiddleware"


def test_triple_store_dedup():
    """Duplicate triples increment frequency instead of creating new ones."""
    from tracera.memory.triples import TripleStore, Triple

    store = TripleStore()
    store.add_triple(Triple("A", "calls", "B"))
    store.add_triple(Triple("A", "calls", "B"))
    assert store.triple_count == 1
    triple = store.get_objects("A")[0]
    assert triple.frequency == 2


def test_triple_store_neighbors():
    """Graph traversal finds multi-hop neighbors."""
    from tracera.memory.triples import TripleStore, Triple

    store = TripleStore()
    store.add_triple(Triple("A", "calls", "B"))
    store.add_triple(Triple("B", "calls", "C"))
    store.add_triple(Triple("C", "queries", "D"))

    neighbors = store.get_neighbors("A", max_depth=3)
    neighbor_concepts = set()
    for t in neighbors:
        neighbor_concepts.add(t.subject.lower())
        neighbor_concepts.add(t.object.lower())
    assert "b" in neighbor_concepts
    assert "c" in neighbor_concepts


def test_triple_store_persistence(tmp_path):
    """Triples persist to disk and reload."""
    from tracera.memory.triples import TripleStore, Triple

    path = tmp_path / "triples.json"
    store1 = TripleStore()
    store1.add_triple(Triple("X", "uses", "Y"))
    store1.save(path)

    store2 = TripleStore.load(path)
    assert store2.triple_count == 1
    assert store2.get_objects("X")[0].object == "Y"


def test_triple_store_to_text():
    """Triples render as readable text."""
    from tracera.memory.triples import TripleStore, Triple

    store = TripleStore()
    store.add_triple(Triple("A", "calls", "B"))
    text = store.to_text()
    assert "A" in text
    assert "calls" in text
    assert "B" in text


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT RECALL
# ═══════════════════════════════════════════════════════════════════════════════


def test_context_recall_assembles_from_multiple_sources(tmp_path):
    """Context recall gathers from memory store, sessions, triples, and legacy."""
    from tracera.memory.recall import ContextRecall, EnhancedMemoryStore
    from tracera.memory.session import SessionManager
    from tracera.memory.triples import TripleStore, Triple
    from tracera.memory.taxonomy import create_fact
    from tracera.agent.memory import AgentMemory, MemoryCategory

    # Set up each source
    enhanced = EnhancedMemoryStore(tmp_path / "mem")
    enhanced.add(create_fact("Project uses pytest for testing"))

    sessions = SessionManager(tmp_path / "mem")
    s = sessions.new_session(task="Fix test failures")
    sessions.close_session(outcome="success", summary="Fixed 3 tests")

    triples = TripleStore()
    triples.add_triple(Triple("AuthMiddleware", "calls", "UserService"))

    legacy = AgentMemory(tmp_path / "mem")
    legacy.add("JWT used for auth", MemoryCategory.PROJECT_FACT)

    recall = ContextRecall(
        memory_store=enhanced,
        session_manager=sessions,
        triple_store=triples,
        legacy_memory=legacy,
    )

    ctx = recall.recall("authentication testing")
    assert "pytest" in ctx or "JWT" in ctx or "AuthMiddleware" in ctx


def test_context_recall_empty_returns_empty():
    """Empty recall sources return empty string."""
    from tracera.memory.recall import ContextRecall

    recall = ContextRecall()
    assert recall.recall("anything") == ""


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED MEMORY STORE
# ═══════════════════════════════════════════════════════════════════════════════


def test_enhanced_memory_store_add_and_recall(tmp_path):
    """Add memories and recall by query."""
    from tracera.memory.recall import EnhancedMemoryStore
    from tracera.memory.taxonomy import create_fact, create_rule

    store = EnhancedMemoryStore(tmp_path / "mem")
    store.add(create_fact("The project uses FastAPI for web routing"))
    store.add(create_fact("pytest is the test framework"))
    store.add(create_rule("Always run tests before committing"))

    results = store.recall("testing framework", k=5)
    assert len(results) >= 1
    # pytest fact should be found
    contents = " ".join(m.content for m in results)
    assert "pytest" in contents


def test_enhanced_memory_dedup(tmp_path):
    """Similar memories are merged, not duplicated."""
    from tracera.memory.recall import EnhancedMemoryStore
    from tracera.memory.taxonomy import create_fact

    store = EnhancedMemoryStore(tmp_path / "mem")
    m1 = store.add(create_fact("Project uses pytest"))
    m2 = store.add(create_fact("Project uses pytest"))
    # Should be deduped — second add updates the existing one
    assert store.count == 1
    assert m2.frequency >= 2


def test_enhanced_memory_persistence(tmp_path):
    """Enhanced memory persists to disk."""
    from tracera.memory.recall import EnhancedMemoryStore
    from tracera.memory.taxonomy import create_fact

    store1 = EnhancedMemoryStore(tmp_path / "mem")
    store1.add(create_fact("Persistent test fact"))

    store2 = EnhancedMemoryStore(tmp_path / "mem")
    assert store2.count == 1


def test_enhanced_memory_stats(tmp_path):
    """Stats report correct counts by type."""
    from tracera.memory.recall import EnhancedMemoryStore
    from tracera.memory.taxonomy import create_fact, create_rule

    store = EnhancedMemoryStore(tmp_path / "mem")
    store.add(create_fact("fact one"))
    store.add(create_fact("fact two"))
    store.add(create_rule("rule one"))

    stats = store.stats()
    assert stats["total"] == 3
    assert stats["by_type"]["fact"] == 2
    assert stats["by_type"]["rule"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


def test_remember_memory_tool(tmp_path):
    """RememberMemoryTool stores memories."""
    import asyncio
    from tracera.memory.recall import EnhancedMemoryStore
    from tracera.tools.memory_tools import RememberMemoryTool

    store = EnhancedMemoryStore(tmp_path / "mem")
    tool = RememberMemoryTool(store)

    result = asyncio.run(tool.execute(content="Test fact", memory_type="fact"))
    assert result.success
    assert store.count == 1


def test_recall_memory_tool(tmp_path):
    """RecallMemoryTool retrieves memories."""
    import asyncio
    from tracera.memory.recall import ContextRecall, EnhancedMemoryStore
    from tracera.memory.taxonomy import create_fact
    from tracera.tools.memory_tools import RecallMemoryTool

    store = EnhancedMemoryStore(tmp_path / "mem")
    store.add(create_fact("Auth uses JWT tokens for authentication"))

    recall = ContextRecall(memory_store=store)
    tool = RecallMemoryTool(recall)

    result = asyncio.run(tool.execute(query="authentication"))
    assert result.success
    assert "JWT" in result.output


def test_forget_memory_tool(tmp_path):
    """ForgetMemoryTool deletes memories."""
    import asyncio
    from tracera.memory.recall import EnhancedMemoryStore
    from tracera.memory.taxonomy import create_fact
    from tracera.tools.memory_tools import ForgetMemoryTool

    store = EnhancedMemoryStore(tmp_path / "mem")
    mem = create_fact("Delete me")
    store.add(mem)
    assert store.count == 1

    tool = ForgetMemoryTool(store)
    result = asyncio.run(tool.execute(memory_id=mem.id))
    assert result.success
    assert store.count == 0


def test_list_sessions_tool(tmp_path):
    """ListSessionsTool shows past sessions."""
    import asyncio
    from tracera.memory.session import SessionManager
    from tracera.tools.memory_tools import ListSessionsTool

    mgr = SessionManager(tmp_path / "mem")
    s = mgr.new_session(task="Fix bug")
    mgr.record_tool_call("run_command", {"command": "pytest"}, "5 passed", True)
    mgr.close_session(outcome="success", summary="Fixed the bug")

    tool = ListSessionsTool(mgr)
    result = asyncio.run(tool.execute(k=5))
    assert result.success
    assert "Fix bug" in result.output
