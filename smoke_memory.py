"""Scratch smoke test for the memory layer store — deleted after use."""
import math
import pathlib
import sys
import tempfile

sys.path.insert(0, ".")

from tracera.memory.layer import MemoryStore


def emb(t: str) -> list[float]:
    v = [0.0] * 32
    for ch in t.lower():
        v[ord(ch) % 32] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def main() -> None:
    db = pathlib.Path(tempfile.mkdtemp()) / "mem.db"
    s = MemoryStore(db)
    inserted, rec = s.upsert_memory(
        entity_id="user_a", process_id="agent", kind="preference",
        subject="user", predicate="favorite_color", object="blue",
        text="User's favorite color is blue.", embedding=emb("favorite color blue"),
        job_id=1,
    )
    print("first insert:", inserted, rec.mention_count)
    assert inserted and rec.mention_count == 1

    inserted, rec = s.upsert_memory(
        entity_id="user_a", process_id="agent", kind="preference",
        subject="user", predicate="favorite_color", object="blue",
        text="The color the user likes most is blue.",
        embedding=emb("color prefers blue"), job_id=2,
    )
    print("restated different words:", inserted, rec.mention_count)
    assert not inserted and rec.mention_count == 2

    inserted, rec = s.upsert_memory(
        entity_id="user_a", process_id="agent", kind="preference",
        subject="user", predicate="favorite_color", object="blue",
        text="The color the user likes most is blue.",
        embedding=emb("color prefers blue"), job_id=2,
    )
    print("retry same job (idempotent):", inserted, rec.mention_count)
    assert not inserted and rec.mention_count == 2

    s.upsert_memory(
        entity_id="user_b", process_id="agent", kind="fact",
        subject="user", predicate="role", object="admin",
        text="User is admin.", embedding=emb("admin role"), job_id=3,
    )
    assert s.count_memories("user_a") == 1
    assert s.count_memories("user_b") == 1
    assert s.count_memories() == 2

    hits = s.recall("user_a", emb("what color does the user like"), k=5)
    print("recall hits for A:", [(r.kind, r.predicate, r.mention_count) for r, _ in hits])
    assert hits and hits[0][0].predicate == "favorite_color"
    assert all(r.entity_id == "user_a" for r, _ in hits)

    print("stats:", s.stats())
    print("STORESMOKE OK")


if __name__ == "__main__":
    main()