"""
TRACERA Enhanced Memory System.

Inspired by agent-native memory concepts (session scoping, structured
extraction, semantic triples, context recall) — all original code adapted
to TRACERA's architecture.

Components:
  - SessionManager: groups coding sessions with metadata
  - MemoryTaxonomy: rich memory types (facts, rules, relationships, skills)
  - ConversationExtractor: LLM-based auto-extraction from conversations
  - TripleStore: semantic triple knowledge graph
  - ContextRecall: retrieval-augmented memory injection before LLM calls
"""

from tracera.memory.session import SessionManager, Session
from tracera.memory.taxonomy import (
    MemoryType,
    StructuredMemory,
    MemoryFact,
    MemoryRule,
    MemoryRelationship,
    MemorySkill,
    MemoryPreference,
    MemoryEvent,
)
from tracera.memory.extractor import ConversationExtractor
from tracera.memory.triples import TripleStore, Triple
from tracera.memory.recall import ContextRecall

__all__ = [
    "SessionManager",
    "Session",
    "MemoryType",
    "StructuredMemory",
    "MemoryFact",
    "MemoryRule",
    "MemoryRelationship",
    "MemorySkill",
    "MemoryPreference",
    "MemoryEvent",
    "ConversationExtractor",
    "TripleStore",
    "Triple",
    "ContextRecall",
]
