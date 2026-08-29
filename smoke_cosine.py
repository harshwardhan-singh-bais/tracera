"""Scratch: cosine values for fake embedding — deleted after use."""
import math
import sys

sys.path.insert(0, ".")


def emb(t: str) -> list[float]:
    v = [0.0] * 32
    for ch in t.lower():
        if ch.isalnum():
            v[ord(ch) % 32] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


pairs = [
    ("how do I fix a type error?", "chess hobby"),
    ("what is my favorite color?", "favorite color blue"),
    ("what is my favorite color?", "favorite color green"),
    ("favorite color", "favorite color green"),
    ("what is seed 7?", "seed memory number 7"),
    ("what is seed 7?", "seed memory number 5"),
]
for a, b in pairs:
    print(f"{cos(emb(a), emb(b)):.3f}  {a!r} / {b!r}")