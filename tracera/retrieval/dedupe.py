"""
Per-file deduplication for retrieval results.

Retrieval ranks *chunks*; the consumer wants *files*. Three fragments of one file
occupy three of the k slots the caller paid for, repeat context the model already
has, and push other candidate files out of the window.

A per-query failure analysis on this repository found hybrid's top-5 was
regularly only two or three distinct files, so both the recall and the context
bytes were worse than the score suggested: the same file was being counted more
than once while a correct file sat just outside the window.

Deduplicating is not just tidiness, and the effect is not the one first assumed.
Measured on this repository's 120-query set at ``k=10``, a cap of one chunk per
file cut mean context bytes by 31% for hybrid, 43% for BM25 and 35% for dense —
*at unchanged* ``k``. A file's later chunks are its larger bodies; only the
top-ranked one survives, and the window refills from files that would otherwise
have been cut off.

The bytes are not free. Recall@10 fell over the same run (hybrid 0.983 → 0.950,
dense 0.921 → 0.854). :func:`dedupe_by_file` always keeps a path's first
occurrence and preserves rank order, so a matched **file** can only move up;
every item this can lose is a *symbol*, whose only matching chunk was a duplicate
of an already-kept file.

Cap 2 measured as the free point and is the default the agent tool uses. The
figures below are from the re-run **after** the stale-vector eviction; the
earlier numbers were taken on a store where 63.9% of rows were leftover
``docs-site/.next`` build output, whose near-duplicates filled the window and
made the byte saving look half its real size:

| strategy | cap | R@5 | R@10 | mean ctx bytes |
|---|---|---|---|---|
| hybrid | 0 | .954 | .983 | 8756 |
| hybrid | 2 | .950 | .983 | **6827** (−22%) |
| dense | 0 | .900 | .921 | 8956 |
| dense | 2 | .904 | .908 | **8108** (−9%) |
| bm25 | 0 | .500 | .500 | 10250 |
| bm25 | 2 | .500 | .500 | **7596** (−26%) |

So cap 2 costs hybrid half a query of recall@5 and buys a fifth of the context.
Cap 1 is the aggressive setting.

``precision@k`` is **not comparable across caps**. Ground truth here is two items
per query, so a cap of one mechanically bounds how many relevant chunks can sit
in the top ``k``: hybrid's baseline precision@5 of 0.63 falls to 0.37 at cap 2
purely because the window holds *distinct* files instead of duplicates of an
already-matched one. Compare recall, MRR and nDCG across arms — not precision.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def dedupe_by_file[T](
    items: Sequence[T],
    *,
    max_per_file: int | None,
    file_path_of: Callable[[T], str | None],
) -> list[T]:
    """
    Keep at most ``max_per_file`` items per file, preserving rank order.

    ``max_per_file`` of ``None`` or ``<= 0`` disables the cap and returns the
    input unchanged, which is what every strategy ran with before this existed.

    ``file_path_of`` is passed in rather than sniffed off the item. The two
    callers use different shapes — ``RetrievalHit`` objects and plain dicts — and
    a helper that guessed would quietly match nothing on the wrong one, leaving
    the caller convinced dedup had been applied when it had not.

    Items with no resolvable path are always kept: there is no file to group them
    by, and silently collapsing them into a single bucket would cap them all
    together.
    """
    if not max_per_file or max_per_file <= 0:
        return list(items)

    seen: dict[str, int] = {}
    kept: list[T] = []
    for item in items:
        path = file_path_of(item)
        if not path:
            kept.append(item)
            continue
        count = seen.get(path, 0)
        if count >= max_per_file:
            continue
        seen[path] = count + 1
        kept.append(item)
    return kept


def overfetch_k(k: int, *, filtering: bool) -> int:
    """
    How many candidates to request so that ``k`` survive post-filtering.

    Asking for exactly ``k`` and then dropping duplicates or non-matching files
    returns *fewer* than ``k`` results — the opposite of the intent. The window
    should hold ``k`` distinct files, so the request is widened whenever any
    post-filter is active.
    """
    return max(k * 4, k + 20) if filtering else k
