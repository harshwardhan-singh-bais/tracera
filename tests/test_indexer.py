"""
Tests for the Tracera Indexer Pipeline (Phases 11-15).
"""

from pathlib import Path

from tracera.indexer.chunker import SymbolAwareChunker
from tracera.indexer.extractor import SymbolExtractor
from tracera.indexer.parser import LanguageParser
from tracera.indexer.scanner import RepositoryScanner
from tracera.indexer.schema import SymbolType


def test_scanner(tmp_path: Path):
    """Test repository scanning and exclusion logic."""
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "data.bin").write_bytes(b"hello\x00world")

    # Create gitignore
    (tmp_path / ".gitignore").write_text("ignore_me.py\n")
    (tmp_path / "ignore_me.py").write_text("print('ignored')")

    # Nested directory
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "utils.py").write_text("def x(): pass")

    scanner = RepositoryScanner(workspace_root=tmp_path)
    files = list(scanner.scan())

    paths = {f.path for f in files}
    assert "main.py" in paths
    assert "nested/utils.py" in paths
    assert "data.bin" not in paths
    assert "ignore_me.py" not in paths


def test_scanner_honours_nested_gitignore(tmp_path: Path):
    """
    A nested .gitignore must apply to its own subtree.

    This is the TRACERA bug: docs-site/.gitignore contains `/.next/`, and that
    pattern is anchored, so the root .gitignore excludes nothing. Reading only
    the root file left every Next.js build artefact in the index.
    """
    site = tmp_path / "docs-site"
    site.mkdir()
    (site / ".gitignore").write_text("/.next/\n/out/\n.source\n")
    (site / "app").mkdir()
    (site / "app" / "page.tsx").write_text("export default function P() {}")
    for build_dir in (".next", "out", ".source"):
        d = site / build_dir / "dev" / "chunks"
        d.mkdir(parents=True)
        (d / "bundle.js").write_text("var bundled = 1;")

    (tmp_path / "main.py").write_text("print('hi')")

    paths = {f.path for f in RepositoryScanner(workspace_root=tmp_path).scan()}

    assert "docs-site/app/page.tsx" in paths, "real source must be indexed"
    assert "main.py" in paths
    assert not [p for p in paths if ".next" in p], f"build output leaked: {paths}"
    assert not [p for p in paths if p.startswith("docs-site/out/")]
    assert not [p for p in paths if ".source" in p]


def test_nested_gitignore_does_not_leak_to_siblings(tmp_path: Path):
    """A nested .gitignore is scoped to its directory, not the whole repo."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / ".gitignore").write_text("secret.py\n")
    (a / "secret.py").write_text("x = 1")
    (b / "secret.py").write_text("x = 2")

    paths = {f.path for f in RepositoryScanner(workspace_root=tmp_path).scan()}

    assert "a/secret.py" not in paths
    assert "b/secret.py" in paths, "sibling directory must not inherit a's rules"


def test_deeper_gitignore_can_unignore(tmp_path: Path):
    """
    Git lets a deeper .gitignore re-include what an ancestor excluded.

    `PathSpec.match_file` returns False for both "negated" and "no match", so
    the layering has to distinguish them via each pattern's `include` flag.
    """
    (tmp_path / ".gitignore").write_text("*.log\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".gitignore").write_text("!keep.log\n")
    (tmp_path / "a.log").write_text("a")
    (sub / "keep.log").write_text("keep")
    (sub / "b.log").write_text("b")

    paths = {f.path for f in RepositoryScanner(workspace_root=tmp_path).scan()}

    assert "a.log" not in paths
    assert "sub/b.log" not in paths
    assert "sub/keep.log" in paths, "deeper negation must win"


def test_build_output_excluded_by_default(tmp_path: Path):
    """Common build dirs are skipped even with no .gitignore at all."""
    for d in ("dist", "build", "out", "coverage", ".turbo", "target"):
        p = tmp_path / d
        p.mkdir()
        (p / "generated.js").write_text("var g = 1;")
    (tmp_path / "real.py").write_text("x = 1")

    paths = {f.path for f in RepositoryScanner(workspace_root=tmp_path).scan()}

    assert paths == {"real.py"}, f"unexpected: {paths}"


def test_scanner_ignores_gitignore_files_themselves(tmp_path: Path):
    """.gitignore is config, not source — it should not end up in the index."""
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "main.py").write_text("x = 1")

    paths = {f.path for f in RepositoryScanner(workspace_root=tmp_path).scan()}

    assert ".gitignore" not in paths
    assert "main.py" in paths


def test_scanner_skips_agent_tooling_directories(tmp_path: Path):
    """
    ``.workbuddy-ai/`` holds agent notes, not source — and ``.md`` is indexed.

    Because markdown is an indexed language, those notes were retrieved for
    questions about the code and outranked the implementation. ``.tracera/`` is
    included here as a regression guard for the same class of directory.
    """
    for d in (".workbuddy-ai", ".tracera"):
        memory = tmp_path / d / "memory"
        memory.mkdir(parents=True)
        (memory / "NOTES.md").write_text("# agent notes\n")
    (tmp_path / "real.py").write_text("x = 1")

    paths = {f.path for f in RepositoryScanner(workspace_root=tmp_path).scan()}

    assert paths == {"real.py"}, f"agent tooling leaked into the index: {paths}"


def test_typescript_query_is_valid():
    """
    Regression: the TypeScript query used to name `identifier` for class and
    interface names, but that grammar types them as `type_identifier`. One
    impossible pattern invalidates the whole Query, so every .ts/.tsx file
    extracted zero symbols — with no error anyone would notice.
    """
    parser = LanguageParser()
    extractor = SymbolExtractor(parser)

    code = b"""
interface Props { a: string }
export class Widget { render() { return 1; } }
export function helper(): void {}
export const Header = () => null;
"""
    symbols = extractor.extract_symbols(code, "typescript")
    names = {s.name for s in symbols}

    assert "Widget" in names, f"class not extracted: {names}"
    assert "Props" in names, f"interface not extracted: {names}"
    assert "helper" in names
    assert "Header" in names, "arrow-function component should count as a function"


def test_tsx_uses_the_jsx_grammar():
    """
    Regression: .tsx was parsed with the plain TypeScript grammar, which does
    not understand JSX, leaving component files full of parse errors.
    """
    parser = LanguageParser()
    extractor = SymbolExtractor(parser)

    code = b"""
export default function Page() {
  return <div className="x"><span>hi</span></div>;
}
"""
    tree = parser.parse(code, "tsx")
    assert tree is not None
    assert not tree.root_node.has_error, "TSX must parse cleanly"

    names = {s.name for s in extractor.extract_symbols(code, "tsx")}
    assert "Page" in names, f"component not extracted: {names}"


def test_scanner_maps_tsx_to_its_own_language(tmp_path: Path):
    (tmp_path / "page.tsx").write_text("export default function P() { return null; }")
    (tmp_path / "lib.ts").write_text("export const x = 1;")

    langs = {f.path: f.language for f in RepositoryScanner(workspace_root=tmp_path).scan()}

    assert langs["page.tsx"] == "tsx"
    assert langs["lib.ts"] == "typescript"


def test_parser_and_extractor():
    """Test tree-sitter parsing and symbol extraction."""
    code = b"""
import os

class MyClass:
    def method_a(self):
        pass

def my_func():
    return True
"""
    parser = LanguageParser()
    extractor = SymbolExtractor(parser)

    symbols = extractor.extract_symbols(code, "python")

    # Verify symbols
    assert len(symbols) >= 3

    names = {s.name: s for s in symbols}

    assert "MyClass" in names
    assert names["MyClass"].type == SymbolType.CLASS

    assert "method_a" in names
    assert names["method_a"].type == SymbolType.METHOD
    assert names["method_a"].parent_symbol == "MyClass"

    assert "my_func" in names
    assert names["my_func"].type == SymbolType.FUNCTION


def test_chunker():
    """Test that the chunker splits files properly based on symbols."""
    content = """
import os

class MyClass:
    def method_a(self):
        pass

def my_func():
    return True
"""
    parser = LanguageParser()
    extractor = SymbolExtractor(parser)
    symbols = extractor.extract_symbols(content.encode("utf-8"), "python")

    chunker = SymbolAwareChunker()
    chunks = chunker.chunk_file("test.py", "python", content, symbols)

    assert len(chunks) >= 3

    # At least one chunk for MyClass and one for my_func
    primary_symbols = {c.primary_symbol for c in chunks if c.primary_symbol}
    assert "MyClass" in primary_symbols
    assert "my_func" in primary_symbols


# ── Language filter aliases ──────────────────────────────────────────────────


def test_typescript_lang_filter_covers_tsx():
    """
    ``--lang typescript`` must match ``.tsx``.

    TSX is indexed under its own language key (it is a separate tree-sitter
    grammar), so a plain equality filter silently returned zero frontend
    results — the whole point of the alias map.
    """
    from tracera.indexer.parser import expand_language_filter

    assert expand_language_filter("typescript") == ["typescript", "tsx"]
    assert expand_language_filter("ts") == ["typescript", "tsx"]
    assert expand_language_filter("TypeScript") == ["typescript", "tsx"]


def test_lang_filter_is_case_and_whitespace_tolerant():
    from tracera.indexer.parser import expand_language_filter

    assert expand_language_filter("  PYTHON  ") == ["python"]
    assert expand_language_filter("tsx") == ["tsx"], "an exact key is never widened"
    assert expand_language_filter(None) is None, "no flag means no filter"
    assert expand_language_filter("") is None


def test_in_clause_builds_equality_for_one_and_in_for_many():
    from tracera.retrieval.vector_store import _in_clause

    assert _in_clause("language", ["python"]) == "language = 'python'"
    assert _in_clause("language", ["typescript", "tsx"]) == "language IN ('typescript', 'tsx')"


def test_in_clause_escapes_quotes():
    """Filter values come from a CLI flag, so they must not break the literal."""
    from tracera.retrieval.vector_store import _in_clause

    clause = _in_clause("language", ["py'thon"])
    assert clause == r"language = 'py\'thon'"


def test_vector_store_passes_a_language_list_through(tmp_path, monkeypatch):
    """
    The widened filter must reach LanceDB as an ``IN`` clause, not ``=``.
    """
    import tracera.retrieval.vector_store as vs

    captured: list[str] = []

    class FakeQuery:
        def where(self, clause, prefilter=False):
            captured.append(clause)
            return self

        def limit(self, k):
            return self

        def to_list(self):
            return []

    class FakeTable:
        def search(self, _vec):
            return FakeQuery()

    monkeypatch.setattr(vs.VectorStore, "_get_or_create_table", lambda self: FakeTable())
    store = vs.VectorStore(tmp_path / "lance")
    store.search([0.1, 0.2], k=5, language=["typescript", "tsx"], symbol_type="class")

    assert any("language IN ('typescript', 'tsx')" in c for c in captured)
    assert any("symbol_type = 'class'" in c for c in captured)
