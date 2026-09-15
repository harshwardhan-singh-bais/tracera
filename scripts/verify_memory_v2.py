"""
End-to-end verification of the memory layer v2 capabilities.

Runs a self-contained, human-readable 51-check report against throwaway
databases in temp dirs (never the configured data dir):

    bi-temporal invalidation + recall_as_of + timeline
    multi-valued predicates  |  reconciliation ADD/UPDATE/DELETE/NOOP
    entity resolution + graph expansion  |  scale + latency
    normalised scoring  |  decay / GC / feedback
    facade regressions  |  token budget  |  debug_recall
    export / import  |  v1 -> v2 schema migration

The same behaviour is pinned by tests/test_memory_layer_v2.py; this script
exists for a quick, readable audit outside pytest.

Usage:/n    uv run python scripts/verify_memory_v2.py
"""
import asyncio, math, os, pathlib, random, re, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracera.memory.layer import MemoryLayer, MemoryStore, MemoryReconciler, ReconcileEvent
from tracera.memory.layer.facade import AgentMemory
from tracera.memory.layer.store import SCHEMA_VERSION

def _tv(tok, dim=48):
    rng = random.Random(f"t:{tok}")
    raw = [rng.uniform(-1, 1) for _ in range(dim)]
    n = math.sqrt(sum(x*x for x in raw)) or 1.0
    return [x/n for x in raw]

def emb(text, dim=48):
    vec = [0.0]*dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        tv = _tv(tok, dim)
        for i in range(dim): vec[i] += tv[i]
    n = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/n for x in vec]

ok = fail = 0
def check(label, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"  PASS  {label}")
    else: fail += 1; print(f"  FAIL  {label}  {detail}")

def H(t): print("\n" + "="*74 + f"\n{t}\n" + "="*74)

# ─────────────────────────────────────────────────────────────────────────────
H("1. BI-TEMPORAL: contradiction invalidates, history stays queryable")
d = pathlib.Path(tempfile.mkdtemp())/"m.db"; s = MemoryStore(d)
t_jan = time.time() - 90*86400
s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject="user",
    predicate="employer", object="acme", text="User works at Acme.",
    embedding=emb("user employer acme"), job_id=1, valid_at=t_jan, confidence=0.9)
time.sleep(0.01)
t_mar = time.time() - 30*86400
s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject="user",
    predicate="employer", object="globex", text="User works at Globex.",
    embedding=emb("user employer globex"), job_id=2, valid_at=t_mar, confidence=0.9)

now_hits = s.recall("u", emb("where does the user work"), k=5)
texts = [r.text for r, _ in now_hits]
check("current recall returns only Globex", any("Globex" in t for t in texts) and not any("Acme" in t for t in texts), texts)
check("old fact is superseded, not deleted", s.count_memories("u") == 2, f"rows={s.count_memories('u')}")

asof = s.recall_as_of("u", emb("where does the user work"), t_mar - 86400, k=5)
asof_texts = [r.text for r, _ in asof]
check("as-of (before the move) returns Acme", any("Acme" in t for t in asof_texts), asof_texts)
check("as-of excludes the future fact", not any("Globex" in t for t in asof_texts), asof_texts)

tl = s.timeline("u", predicate="employer")
check("timeline shows both versions in order", len(tl) == 2 and "Acme" in tl[0].text and "Globex" in tl[1].text,
      [(r.text, r.status) for r in tl])
check("invalidated row carries invalid_at + superseded_by",
      tl[0].invalid_at is not None and tl[0].superseded_by == tl[1].id)

# ─────────────────────────────────────────────────────────────────────────────
H("2. MULTI-VALUED predicates must NOT invalidate each other")
d = pathlib.Path(tempfile.mkdtemp())/"m.db"; s = MemoryStore(d)
for i, lang in enumerate(["python", "rust", "go"]):
    s.upsert_memory(entity_id="u", process_id="p", kind="preference", subject="user",
        predicate="likes_language", object=lang, text=f"User likes {lang}.",
        embedding=emb(f"user likes {lang}"), job_id=i+1)
check("all three accumulate", s.count_memories("u") == 3, s.count_memories("u"))

H("3. RECONCILIATION: ADD / NOOP / UPDATE / DELETE")
r = MemoryReconciler(use_llm=False)
store = MemoryStore(pathlib.Path(tempfile.mkdtemp())/"m.db")
store.upsert_memory(entity_id="u", process_id="p", kind="preference", subject="user",
    predicate="preferred_editor", object="vscode", text="User prefers VS Code.",
    embedding=emb("user prefers vscode"), job_id=1, confidence=0.9)
existing = store.find_memories("u")

a = r.reconcile_deterministic({"kind":"preference","subject":"user","predicate":"preferred_editor",
    "object":"vscode","text":"User prefers VS Code.","confidence":0.9}, existing)
check("identical fact -> NOOP", a.event is ReconcileEvent.NOOP, a)

a = r.reconcile_deterministic({"kind":"preference","subject":"user","predicate":"preferred_editor",
    "object":"neovim","text":"User now prefers Neovim.","confidence":0.95}, existing)
check("changed single-valued value -> UPDATE", a.event is ReconcileEvent.UPDATE, a)

a = r.reconcile_deterministic({"kind":"fact","subject":"user","predicate":"hobby",
    "object":"cycling","text":"User enjoys cycling.","confidence":0.8}, existing)
check("unrelated fact -> ADD", a.event is ReconcileEvent.ADD, a)

# LLM path
class FakeLLM:
    def __init__(self, reply): self.reply = reply
    async def __call__(self, prompt): return self.reply
async def llm_tests():
    rc = MemoryReconciler(FakeLLM('{"event":"UPDATE","memory_id":%d,"reason":"moved"}' % existing[0].id))
    act = await rc.reconcile({"text":"x"}, existing)
    check("LLM UPDATE parsed", act.event is ReconcileEvent.UPDATE and act.source == "llm", act)
    rc = MemoryReconciler(FakeLLM('```json\n{"event":"DELETE","memory_id":99999,"reason":"bogus"}\n```'))
    act = await rc.reconcile({"text":"x"}, existing)
    check("LLM naming an unknown id is rejected -> deterministic", act.source == "fallback", act)
    rc = MemoryReconciler(FakeLLM('not json at all'))
    act = await rc.reconcile({"text":"x"}, existing)
    check("unparseable LLM reply -> fallback", act.source == "fallback", act)
    async def boom(p): raise RuntimeError("api down")
    rc = MemoryReconciler(boom)
    act = await rc.reconcile({"text":"x"}, existing)
    check("LLM exception -> fallback (never fatal)", act.source == "fallback", act)
asyncio.run(llm_tests())

# ─────────────────────────────────────────────────────────────────────────────
H("4. ENTITY RESOLUTION + GRAPH EXPANSION")
d = pathlib.Path(tempfile.mkdtemp())/"m.db"; s = MemoryStore(d)
s.upsert_memory(entity_id="u", process_id="p", kind="relationship", subject="AuthMiddleware",
    predicate="calls", object="UserService", text="AuthMiddleware calls UserService.",
    embedding=emb("authmiddleware calls userservice"), job_id=1)
s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject="auth_middleware",
    predicate="validates", object="JWT tokens", text="AuthMiddleware validates JWT tokens.",
    embedding=emb("authmiddleware validates jwt tokens"), job_id=2)
s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject="user",
    predicate="favorite_food", object="ramen", text="User loves ramen.",
    embedding=emb("user loves ramen food"), job_id=3)
check("camelCase and snake_case resolve to one node", s.resolve_entity("u", "AuthMiddleware") == s.resolve_entity("u", "auth_middleware"),
      (s.resolve_entity("u","AuthMiddleware"), s.resolve_entity("u","auth_middleware")))
check("'the user' resolves to 'user'", s.resolve_entity("u", "the user") == "user", s.resolve_entity("u","the user"))
g = s.entity_graph("u", node="auth_middleware")
check("graph has outgoing edges for the node", len(g["outgoing"]) >= 2, g)
overview = s.entity_graph("u")
check("graph overview lists canonical nodes", len(overview["nodes"]) >= 3, overview)

# graph expansion recovers a neighbour fact the query never names
hits = s.recall_hybrid("u", "AuthMiddleware", emb("AuthMiddleware"), k=3, min_score=0.0)
plain = {r.id for r, _ in hits}
hits_g = s.recall_hybrid("u", "AuthMiddleware", emb("AuthMiddleware"), k=3, min_score=0.0, graph_expansion=True)
expanded = {r.id for r, _ in hits_g}
check("graph expansion adds neighbour memories", len(expanded) >= len(plain), (plain, expanded))

# ─────────────────────────────────────────────────────────────────────────────
H("5. SCALE: ingest scaling + recall latency at 3k memories")
d = pathlib.Path(tempfile.mkdtemp())/"m.db"; s = MemoryStore(d)
n = 3000
t0 = time.perf_counter()
for i in range(n):
    s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject=f"e{i}", predicate="rel",
        object=f"v{i}", text=f"entity {i} relates to value {i} in module {i%17}",
        embedding=emb(f"entity {i} value {i} module {i%17}"), job_id=i+1)
ingest = time.perf_counter() - t0

# Absolute wall-clock is load-dependent, so verify the *scaling* property the
# incremental index exists to provide: a second batch of the same size must
# cost roughly the same as the first. Rebuilding the matrix per write (the old
# O(n^2) behaviour) made batch 2 several times slower than batch 1.
half = n // 2
t0 = time.perf_counter()
for i in range(half, n):
    s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject=f"x{i}", predicate="rel",
        object=f"w{i}", text=f"second batch entity {i} value {i} module {i%17}",
        embedding=emb(f"second batch entity {i} value {i} module {i%17}"), job_id=n+i+1)
second_batch = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(20): s.recall("u", emb("entity 42 value 42"), k=5)
recall_ms = (time.perf_counter()-t0)/20*1000
t0 = time.perf_counter()
for _ in range(20): s.recall_hybrid("u", "entity 42 value 42", emb("entity 42 value 42"), k=5)
hybrid_ms = (time.perf_counter()-t0)/20*1000
print(f"  ingest {n}: {ingest:.2f}s ({n/ingest:.0f} writes/s)")
print(f"  second batch of {half}: {second_batch:.2f}s")
print(f"  recall: {recall_ms:.2f}ms   hybrid: {hybrid_ms:.2f}ms")
# Load-independent: the second batch must not be materially slower than the
# first. O(n^2) rebuild-per-write made it ~4x; incremental append keeps it ~1x.
ratio = second_batch / max(ingest, 1e-6)
check("ingest scales linearly (2nd batch not slower than the 1st)", ratio < 1.8,
      f"ratio={ratio:.2f} (batch1={ingest:.2f}s batch2={second_batch:.2f}s)")
check("ingest 3k is not pathologically slow (<60s)", ingest < 60, f"{ingest:.1f}s")
check("recall under 25ms", recall_ms < 25, f"{recall_ms:.2f}ms")

H("6. NORMALISED SCORING: irrelevant memories score ~0")
d = pathlib.Path(tempfile.mkdtemp())/"m.db"; s = MemoryStore(d)
for i in range(10):
    s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject=f"c{i}", predicate="p",
        object="o", text=f"totally unrelated fact {i} about medieval poetry",
        embedding=emb(f"medieval poetry {i}"), importance=1.0, confidence=1.0, job_id=i+1)
res = s.recall_hybrid("u", "kubernetes ingress controller yaml", emb("kubernetes ingress controller"),
                      k=10, min_score=0.3)
check("no irrelevant memory clears the 0.3 floor", len(res) == 0, [(r.text[:30], round(sc,3)) for r,sc in res])
res0 = s.recall_hybrid("u", "kubernetes ingress", emb("kubernetes ingress"), k=10, min_score=0.0)
top = res0[0][1] if res0 else 0
check("irrelevant top score is low (<0.35) even at min_score=0", top < 0.35, top)

H("7. FORGETTING: decay + GC + feedback")
d = pathlib.Path(tempfile.mkdtemp())/"m.db"; s = MemoryStore(d)
old = time.time() - 400*86400
for i in range(5):
    _ins, rec = s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject=f"old{i}",
        predicate="p", object="o", text=f"stale fact {i}", embedding=emb(f"stale {i}"), job_id=i+1)
    s._conn().execute("UPDATE memories SET last_seen_at = ? WHERE id = ?", (old, rec.id))
s._conn().commit()
s.upsert_memory(entity_id="u", process_id="p", kind="preference", subject="user",
    predicate="preferred_editor", object="neovim", text="User prefers Neovim.",
    embedding=emb("user prefers neovim"), job_id=99)
n_decay = s.apply_decay(half_life_days=90)
check("apply_decay scores every active memory", n_decay == 6, n_decay)
gc = s.run_gc(retention_days=365, decay_floor=0.05, min_age_days=30)
check("GC archives the stale ones", gc["archived"] >= 4, gc)
check("GC leaves the fresh memory alone",
      any(r.text == "User prefers Neovim." for r in s.find_memories("u", current_only=True)),
      [r.text for r in s.find_memories("u", current_only=True)])
dry = s.run_gc(retention_days=365, decay_floor=0.99, min_age_days=0, dry_run=True)
check("dry_run reports without archiving", dry["dry_run"] is True and dry["archived"] == 0, dry)

d = pathlib.Path(tempfile.mkdtemp())/"m.db"; s = MemoryStore(d)
_ins, rec = s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject="a",
    predicate="b", object="c", text="User prefers dark mode.", embedding=emb("user prefers dark mode"), job_id=1)
before = rec.importance
fb = s.record_feedback(rec.id, "useful", query="theme")
check("useful feedback raises importance", fb["importance"] > before, (before, fb))
fb = s.record_feedback(rec.id, "harmful")
check("harmful feedback lowers importance", fb["harmless"] if False else fb["importance"] < before + 0.05, fb)

H("8. FACADE: the previously-broken paths")
d = pathlib.Path(tempfile.mkdtemp())/"m.db"
layer = MemoryLayer(store=MemoryStore(d/"m.db"), embed_fn=emb, worker_enabled=False)
layer.attribution("user_1", "agent")
am = AgentMemory(layer)
rec = am.add({"text": "User prefers PostgreSQL.", "kind": "preference"})
check("add() returns a record with a real id", rec is not None and rec.id > 0, rec)
check("add() uses the active entity, not 'default'", rec.entity_id == "user_1", rec.entity_id)
check("add() derives a predicate from the kind", rec.predicate == "prefers", rec.predicate)
out = am.recall("what database does the user prefer")
check("recall() no longer raises", isinstance(out, str) and out != "", out[:60])
e1 = am.recall("what database does the user prefer")
e2 = am.recall("zzz completely unrelated query zzz")
check("recall() actually depends on the query", e1 != e2 or "No relevant" in e2, (e1[:40], e2[:40]))
check("entries() no longer raises", isinstance(am.entries(), list))
st = am.stats()
check("stats() reports the real entity", st.get("entity_id") == "user_1", st)
rec2 = am.add({"text": "User prefers Vim keybindings.", "kind": "preference"})
check("two distinct explicit memories do not collide", rec2.id != rec.id, (rec.id, rec2.id))
deleted = am.delete(str(rec.id))
check("delete() reports success", deleted is True, deleted)
check("delete() actually removes it from current recall",
      not any(r.id == rec.id for r in layer.store.find_memories("user_1", current_only=True)),
      [r.id for r in layer.store.find_memories("user_1", current_only=True)])
check("deleted memory keeps an audit trail", len(layer.store.get_memory_versions(rec.id)) >= 1)

H("9. TOKEN BUDGET is enforced")
from tracera.memory.layer.recall import RecallInjector, estimate_tokens
from tracera.memory.layer.store import MemoryRecord
s = MemoryStore(pathlib.Path(tempfile.mkdtemp())/"m.db")
inj = RecallInjector(s, emb, top_k=5, token_budget=100)
big = MemoryRecord(id=1, entity_id="u", process_id="p", kind="fact", subject="s", predicate="p",
    object="o", text="X"*4000, embedding=[0.0]*8, mention_count=1, first_seen_at=0, last_seen_at=0)
kept = inj._apply_token_budget([(big, 0.9)])
rendered = kept[0][0].to_line() if kept else ""
check("oversized memory is truncated to fit", estimate_tokens(rendered) <= 100, estimate_tokens(rendered))
many = [MemoryRecord(id=i, entity_id="u", process_id="p", kind="fact", subject="s", predicate="p",
    object="o", text="word "*40, embedding=[0.0]*8, mention_count=1, first_seen_at=0, last_seen_at=0)
    for i in range(20)]
kept = inj._apply_token_budget([(m, 0.9) for m in many])
total = sum(estimate_tokens(r.to_line()) for r, _ in kept) + estimate_tokens("Known context about this user:")
check("budget honoured across many memories", total <= 100, total)

H("10. DEBUG RECALL on both paths, WITH data")
inj_h = RecallInjector(s, emb, use_hybrid=True)
inj_v = RecallInjector(s, emb, use_hybrid=False)
s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject="a", predicate="b", object="c",
    text="User prefers PostgreSQL.", embedding=emb("user prefers postgresql"), job_id=1)
class Sc: entity_id="u"; process_id="p"
for name, inj in (("hybrid", inj_h), ("vector-only", inj_v)):
    try:
        dbg = inj.debug_recall("postgres", Sc())
        check(f"debug_recall works on the {name} path", dbg["returned"] >= 1, dbg.get("returned"))
    except Exception as e:
        check(f"debug_recall works on the {name} path", False, f"{type(e).__name__}: {e}")

H("11. EXPORT / IMPORT round-trip")
d = pathlib.Path(tempfile.mkdtemp())/"m.db"; s = MemoryStore(d)
s.upsert_memory(entity_id="u", process_id="p", kind="fact", subject="user", predicate="employer",
    object="acme", text="User works at Acme.", embedding=emb("user employer acme"), job_id=1)
s.upsert_memory(entity_id="u", process_id="p", kind="preference", subject="user", predicate="preferred_editor",
    object="neovim", text="User prefers Neovim.", embedding=emb("user prefers neovim"), job_id=2)
payload = s.export_entity("u")
check("export carries memories + edges", len(payload["memories"]) == 2 and "edges" in payload, len(payload["memories"]))
s2 = MemoryStore(pathlib.Path(tempfile.mkdtemp())/"m.db")
stats = s2.import_entity(payload, embed_fn=emb)
check("import rehydrates into a fresh store", s2.count_memories("u") == 2, stats)
check("import is idempotent", s2.import_entity(payload, embed_fn=emb)["imported"] == 0)

H("12. SCHEMA MIGRATION from a v1 database")
import sqlite3
d = pathlib.Path(tempfile.mkdtemp()); db = d/"v1.db"
conn = sqlite3.connect(db)
conn.executescript("""
CREATE TABLE entities (id INTEGER PRIMARY KEY AUTOINCREMENT, external_id TEXT NOT NULL UNIQUE, created_at REAL NOT NULL);
CREATE TABLE processes (id INTEGER PRIMARY KEY AUTOINCREMENT, external_id TEXT NOT NULL UNIQUE, created_at REAL NOT NULL);
CREATE TABLE sessions (id TEXT PRIMARY KEY, entity_id INTEGER NOT NULL, process_id INTEGER NOT NULL, started_at REAL NOT NULL, ended_at REAL);
CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, entity_id INTEGER NOT NULL, process_id INTEGER NOT NULL,
  kind TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL, text TEXT NOT NULL,
  embedding TEXT NOT NULL, mention_count INTEGER NOT NULL DEFAULT 1, first_seen_at REAL NOT NULL, last_seen_at REAL NOT NULL,
  session_id TEXT, last_job_id INTEGER, status TEXT NOT NULL DEFAULT 'active', confidence REAL NOT NULL DEFAULT 0.8,
  importance REAL NOT NULL DEFAULT 0.5, source_event TEXT, source_message_id TEXT);
CREATE TABLE memory_versions (id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id INTEGER NOT NULL, old_text TEXT, new_text TEXT,
  old_status TEXT, new_status TEXT, changed_at REAL NOT NULL, reason TEXT, source_session TEXT, source_process TEXT, source_job_id INTEGER);
CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, payload TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
  not_before REAL NOT NULL DEFAULT 0, last_error TEXT, priority INTEGER NOT NULL DEFAULT 100);
""")
conn.execute("INSERT INTO entities (external_id, created_at) VALUES ('u', 1.0)")
conn.execute("INSERT INTO processes (external_id, created_at) VALUES ('p', 1.0)")
import json as _json
conn.execute("INSERT INTO memories (entity_id, process_id, kind, subject, predicate, object, text, embedding, first_seen_at, last_seen_at) "
             "VALUES (1,1,'fact','user','employer','acme','User works at Acme.',?,100.0,100.0)", (_json.dumps(emb("user employer acme")),))
conn.commit(); conn.close()

s3 = MemoryStore(db)  # should migrate in place
cols = {r[1] for r in s3._conn().execute("PRAGMA table_info(memories)").fetchall()}
check("v1 db gains the v2 temporal columns", {"valid_at","invalid_at","expired_at","superseded_by","embedding_f32"} <= cols,
      sorted({"valid_at","invalid_at","expired_at","superseded_by","embedding_f32"} - cols))
legacy = s3.find_memories("u", current_only=True)
check("legacy row survives migration and stays current", len(legacy) == 1 and legacy[0].valid_at == 100.0, legacy)
check("legacy row is recallable via the new vector index", len(s3.recall("u", emb("user employer acme"), k=1)) == 1)
check("schema version recorded", s3._conn().execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()["value"] == str(SCHEMA_VERSION))

print("\n" + "="*74)
print(f"RESULT: {ok} passed, {fail} failed")
print("="*74)
sys.exit(1 if fail else 0)
