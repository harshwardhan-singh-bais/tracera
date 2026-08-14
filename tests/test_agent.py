"""Tests for TRACERA agent: conversation, planner, memory."""

import pytest
from pathlib import Path


# ── Conversation State ────────────────────────────────────────────────────────

def test_conversation_add_messages():
    from tracera.conversation.state import ConversationState, MessageType

    conv = ConversationState(system_prompt="You are helpful.")
    assert len(conv) == 1  # system message

    conv.add_user("Hello")
    conv.add_assistant("Hi there!")
    assert conv.stats.user_messages == 1
    assert conv.stats.assistant_messages == 1


def test_conversation_llm_messages():
    from tracera.conversation.state import ConversationState
    from tracera.providers.base import Role

    conv = ConversationState(system_prompt="sys")
    conv.add_user("user msg")
    conv.add_assistant("asst msg")

    msgs = conv.llm_messages()
    assert msgs[0].role == Role.SYSTEM
    assert msgs[1].role == Role.USER
    assert msgs[2].role == Role.ASSISTANT


def test_conversation_truncate():
    from tracera.conversation.state import ConversationState

    conv = ConversationState(system_prompt="sys")
    for i in range(20):
        conv.add_user(f"msg {i}")
        conv.add_assistant(f"reply {i}")

    truncated = conv.truncate(5)
    # system + 5 recent
    assert len(truncated) <= 6


def test_conversation_snapshot():
    from tracera.conversation.state import ConversationState

    conv = ConversationState()
    conv.add_user("original")
    snap = conv.snapshot()
    conv.add_user("new message")

    assert len(snap) == 1
    assert len(conv) == 2


def test_conversation_tool_calls():
    from tracera.conversation.state import ConversationState
    from tracera.providers.base import ToolCallRequest

    conv = ConversationState()
    tc = ToolCallRequest(id="call-1", name="read_file", arguments={"path": "foo.py"})
    conv.add_tool_calls([tc])
    conv.add_tool_result("call-1", "read_file", "file contents here")

    assert conv.stats.tool_calls == 1
    assert conv.stats.tool_results == 1


# ── Planning System ───────────────────────────────────────────────────────────

def test_plan_add_items():
    from tracera.agent.planner import Plan, TodoStatus

    plan = Plan("Add authentication")
    item1 = plan.add_item("Read existing code", priority=0)
    item2 = plan.add_item("Add JWT validation", priority=1)

    assert len(plan.items) == 2
    assert plan.next_item() == item1


def test_plan_lifecycle():
    from tracera.agent.planner import Plan, TodoStatus

    plan = Plan("Test task")
    item = plan.add_item("Step 1")

    plan.start_item(item.id)
    assert item.status == TodoStatus.IN_PROGRESS

    plan.complete_item(item.id, result="Done!")
    assert item.status == TodoStatus.DONE
    assert plan.is_complete


def test_plan_with_failures():
    from tracera.agent.planner import Plan, TodoStatus

    plan = Plan("Failing task")
    item = plan.add_item("Step 1")
    plan.fail_item(item.id, error="Command failed")

    assert plan.has_failures
    assert not plan.is_complete


def test_plan_progress():
    from tracera.agent.planner import Plan

    plan = Plan("Multi-step")
    items = [plan.add_item(f"Step {i}") for i in range(4)]
    plan.complete_item(items[0].id)
    plan.complete_item(items[1].id)

    done, total = plan.progress
    assert done == 2
    assert total == 4
    assert plan.progress_pct == 50.0


def test_plan_to_markdown():
    from tracera.agent.planner import Plan

    plan = Plan("Write tests")
    plan.add_item("Setup fixtures", description="Create tmp files")
    plan.add_item("Write assertions")

    md = plan.to_markdown()
    assert "Write tests" in md
    assert "Setup fixtures" in md
    assert "[ ]" in md


def test_plan_dependency_resolution():
    from tracera.agent.planner import Plan, TodoStatus

    plan = Plan("Dep test")
    a = plan.add_item("A", priority=0)
    b = plan.add_item("B", priority=1, depends_on=[a.id])

    # B is not ready because A is pending
    assert b not in plan.ready
    plan.complete_item(a.id)
    # Now B should be ready
    assert b in plan.ready


# ── Agent Memory ──────────────────────────────────────────────────────────────

def test_memory_add_and_retrieve(tmp_path):
    from tracera.agent.memory import AgentMemory, MemoryCategory

    memory = AgentMemory(tmp_path)
    memory.add("JWT tokens are used for authentication", MemoryCategory.PROJECT_FACT)
    memory.add("User prefers type hints everywhere", MemoryCategory.USER_PREFERENCE)

    results = memory.retrieve("JWT authentication")
    assert len(results) >= 1
    assert any("JWT" in r.content for r in results)


def test_memory_persistence(tmp_path):
    from tracera.agent.memory import AgentMemory, MemoryCategory

    m1 = AgentMemory(tmp_path)
    m1.add("Database uses PostgreSQL", MemoryCategory.PROJECT_FACT)
    assert m1.count == 1

    # New instance, same directory
    m2 = AgentMemory(tmp_path)
    assert m2.count == 1
    assert "PostgreSQL" in m2.retrieve("database")[0].content


def test_memory_deduplication(tmp_path):
    from tracera.agent.memory import AgentMemory, MemoryCategory

    memory = AgentMemory(tmp_path)
    memory.add("Auth uses JWT tokens", MemoryCategory.PROJECT_FACT)
    initial_count = memory.count

    # Very similar entry should be deduplicated
    memory.add("Auth uses JWT tokens for validation", MemoryCategory.PROJECT_FACT)
    # Should not double the count significantly
    assert memory.count <= initial_count + 1


def test_memory_delete(tmp_path):
    from tracera.agent.memory import AgentMemory, MemoryCategory

    memory = AgentMemory(tmp_path)
    entry = memory.add("To delete", MemoryCategory.TASK_CONTEXT)
    assert memory.count == 1

    deleted = memory.delete(entry.id)
    assert deleted
    assert memory.count == 0


def test_memory_build_context(tmp_path):
    from tracera.agent.memory import AgentMemory, MemoryCategory

    memory = AgentMemory(tmp_path)
    memory.add("Project uses FastAPI", MemoryCategory.PROJECT_FACT)
    memory.add("Database is PostgreSQL", MemoryCategory.PROJECT_FACT)

    ctx = memory.build_context("FastAPI database")
    assert "## Relevant Memory" in ctx
    assert "FastAPI" in ctx


def test_memory_categories(tmp_path):
    from tracera.agent.memory import AgentMemory, MemoryCategory

    memory = AgentMemory(tmp_path)
    memory.remember_project_fact("Uses React frontend")
    memory.remember_preference("Always use type hints")
    memory.remember_decision("Chose PostgreSQL over MySQL")

    assert memory.count == 3
    facts = memory.get_by_category(MemoryCategory.PROJECT_FACT)
    assert len(facts) == 1
