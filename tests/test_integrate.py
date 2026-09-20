"""
Tests for the agent-harness integration layer.

Covers :mod:`tracera.mcp.agent_brief`, :mod:`tracera.mcp.harness` and
:mod:`tracera.mcp.integrate`.

Two properties matter more than any individual assertion here, because getting
them wrong silently damages a user's repo:

1. **Idempotency** — a second ``apply`` must write nothing.
2. **Non-destruction** — user prose outside the managed block must survive
   byte-for-byte, and a ``--dry-run`` must not touch the filesystem at all.

Everything runs against a ``tmp_path`` repo, so no test writes into the
developer's real home directory or the real ``.tracera`` data dir.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tracera.mcp import agent_brief, integrate
from tracera.mcp.harness import HARNESSES, catalog, resolve_keys, targets_for
from tracera.mcp.hosts import HOSTS

# ── agent_brief ──────────────────────────────────────────────────────────────


def test_brief_body_names_real_tools() -> None:
    """Every tool the brief advertises must exist in the live registry."""
    from tracera.mcp.server import ALL_MCP_TOOLS

    body = agent_brief.brief_body(mcp=True)
    for tool in ALL_MCP_TOOLS:
        assert f"`{tool}`" in body, f"{tool} is missing from the brief"


def test_brief_tool_count_matches_registry() -> None:
    from tracera.mcp.server import ALL_MCP_TOOLS

    assert agent_brief.describe()["tool_count"] == len(ALL_MCP_TOOLS)


def test_cli_brief_does_not_name_mcp_tools() -> None:
    """The CLI variant must not tell an MCP-less harness to call MCP tools."""
    body = agent_brief.brief_body(mcp=False)
    assert "tracera search" in body
    assert "get_blast_radius" not in body
    assert "check_edit_safe" not in body


def test_splice_into_missing_file() -> None:
    text, status = agent_brief.splice(None, "BODY", "markdown")
    assert status == "installed"
    assert agent_brief.has_managed_block(text, "markdown")


def test_splice_is_idempotent() -> None:
    body = agent_brief.brief_body(mcp=True)
    first, _ = agent_brief.splice(None, body, "markdown")
    second, status = agent_brief.splice(first, body, "markdown")
    assert status == "unchanged"
    assert first == second


def test_splice_preserves_surrounding_prose() -> None:
    body = agent_brief.brief_body(mcp=True)
    block, _ = agent_brief.splice(None, body, "markdown")
    user_text = f"# House rules\n\nBe terse.\n\n{block}\n## Later\n\nkeep me\n"

    updated, status = agent_brief.splice(user_text, body + "\n\nEXTRA", "markdown")

    assert status == "updated"
    assert "Be terse." in updated
    assert "keep me" in updated
    assert "EXTRA" in updated


def test_splice_refuses_malformed_block() -> None:
    """A begin marker with no end must not be silently rewritten."""
    orphan = agent_brief._HTML_BEGIN + "\norphan content\n"
    with pytest.raises(agent_brief.MalformedBlockError):
        agent_brief.splice(orphan, "BODY", "markdown")


def test_mdc_gets_frontmatter_once() -> None:
    body = agent_brief.brief_body(mcp=True)
    text, _ = agent_brief.splice(None, body, "mdc")
    assert text.startswith("---\n")
    assert "alwaysApply: true" in text

    # An existing frontmatter block must not be duplicated.
    with_fm = "---\nx: 1\n---\n\nstuff\n"
    merged, _ = agent_brief.splice(with_fm, body, "mdc")
    assert merged.count("---\n") == 2


def test_plaintext_uses_hash_markers() -> None:
    text, _ = agent_brief.splice(None, "BODY", "plaintext")
    assert agent_brief._HASH_BEGIN in text
    assert agent_brief._HTML_BEGIN not in text


def test_unknown_format_rejected() -> None:
    with pytest.raises(ValueError, match="unknown instruction format"):
        agent_brief.render("BODY", "nonsense")


def test_strip_block_round_trips() -> None:
    body = agent_brief.brief_body(mcp=True)
    text, _ = agent_brief.splice(None, body, "markdown")
    assert agent_brief.strip_block(text, "markdown").strip() == ""


# ── harness catalog ──────────────────────────────────────────────────────────


def test_every_mcp_host_reference_exists() -> None:
    """A typo in a harness's mcp_host would only surface at apply time."""
    for harness in HARNESSES.values():
        if harness.mcp_host is not None:
            assert harness.mcp_host in HOSTS, (
                f"{harness.key} points at unknown host {harness.mcp_host!r}"
            )


def test_every_harness_format_is_renderable() -> None:
    for harness in HARNESSES.values():
        for target in harness.instructions:
            assert target.fmt in agent_brief.FORMATS, (
                f"{harness.key}/{target.key} uses unsupported format {target.fmt!r}"
            )


def test_harness_keys_match_their_dict_keys() -> None:
    for key, harness in HARNESSES.items():
        assert key == harness.key


def test_targets_resolve_under_repo_root(tmp_path: Path) -> None:
    """Project targets must stay inside the repo; user targets must not."""
    for harness in HARNESSES.values():
        for target in harness.instructions:
            path = target.path(tmp_path)
            if target.scope == "project":
                assert tmp_path in path.parents, f"{harness.key} escapes the repo"
            else:
                assert tmp_path not in path.parents, f"{harness.key} is not user-scoped"


def test_agents_md_deduplicated(tmp_path: Path) -> None:
    """Many harnesses share AGENTS.md; it must be resolved once."""
    pairs = targets_for(None, repo_root=tmp_path)
    agents = [p for _, p in pairs if p.path(tmp_path) == tmp_path / "AGENTS.md"]
    assert len(agents) == 1


def test_explicit_keys_override_detection(tmp_path: Path) -> None:
    """Naming a harness integrates it even when nothing suggests it is installed."""
    pairs = targets_for(["aider"], repo_root=tmp_path)
    assert pairs, "explicit key produced no targets"
    assert all(h.key == "aider" for h, _ in pairs)


def test_resolve_keys_splits_known_and_unknown() -> None:
    known, unknown = resolve_keys(["cursor", "not-a-real-tool", "aider"])
    assert known == ["cursor", "aider"]
    assert unknown == ["not-a-real-tool"]


def test_catalog_is_json_serialisable() -> None:
    json.dumps(catalog())


# ── integrate.apply ──────────────────────────────────────────────────────────


def test_apply_dry_run_writes_nothing(tmp_path: Path) -> None:
    report = integrate.apply(["claude-code", "aider"], repo_root=tmp_path, mcp=False, dry_run=True)
    assert report.dry_run
    assert report.actions
    assert all(not a.changed for a in report.actions)
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / "CLAUDE.md").exists()


def test_apply_is_idempotent(tmp_path: Path) -> None:
    keys = ["claude-code", "cursor", "aider"]
    first = integrate.apply(keys, repo_root=tmp_path, mcp=False)
    assert first.changed

    second = integrate.apply(keys, repo_root=tmp_path, mcp=False)
    assert not second.changed, "second apply rewrote files"
    assert all(a.status == "unchanged" for a in second.actions)


def test_apply_preserves_existing_instruction_file(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# House rules\n\nBe terse.\n", encoding="utf-8")

    integrate.apply(["claude-code"], repo_root=tmp_path, mcp=False)

    text = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Be terse." in text
    assert agent_brief.has_managed_block(text, "markdown")


def test_apply_writes_bak_sidecar(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# House rules\n", encoding="utf-8")
    integrate.apply(["claude-code"], repo_root=tmp_path, mcp=False)
    assert (tmp_path / "CLAUDE.md.bak").exists()


def test_apply_scope_project_excludes_user_targets(tmp_path: Path) -> None:
    integrate.apply(["claude-code"], repo_root=tmp_path, scope="project", mcp=False)
    targets = {a.target for a in integrate.apply(
        ["claude-code"], repo_root=tmp_path, scope="project", mcp=False
    ).actions}
    assert not any(t.startswith("~") for t in targets)


def test_apply_project_mcp_hosts_stay_in_repo(tmp_path: Path) -> None:
    """claude-code/cursor/vscode use project-scoped config files."""
    report = integrate.apply(
        ["claude-code", "cursor", "vscode-copilot"],
        repo_root=tmp_path,
        instructions=False,
    )
    assert report.actions
    for action in report.actions:
        assert action.path is not None
        assert tmp_path in action.path.parents

    assert (tmp_path / ".mcp.json").exists()
    assert (tmp_path / ".cursor" / "mcp.json").exists()
    assert (tmp_path / ".vscode" / "mcp.json").exists()


def test_project_mcp_config_shapes(tmp_path: Path) -> None:
    integrate.apply(
        ["claude-code", "cursor", "vscode-copilot"], repo_root=tmp_path, instructions=False
    )

    claude = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert "tracera" in claude["mcpServers"]

    cursor = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    assert "tracera" in cursor["mcpServers"]

    # VS Code uses "servers" and an explicit transport type.
    vscode = json.loads((tmp_path / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
    assert "tracera" in vscode["servers"]
    assert vscode["servers"]["tracera"]["type"] == "stdio"


def test_apply_never_aborts_on_one_bad_harness(tmp_path: Path) -> None:
    """One unwritable target must not stop the rest of the run."""
    (tmp_path / "CLAUDE.md").write_text("<!-- tracera:begin (managed by `tracera integrate`) -->\n", encoding="utf-8")

    report = integrate.apply(["claude-code", "aider"], repo_root=tmp_path, mcp=False)

    claude = [a for a in report.actions if a.harness == "claude-code"]
    assert claude and claude[0].status == "skipped"
    # Aider still got its brief.
    assert any(a.harness == "aider" and a.changed for a in report.actions)


def test_report_serialises_to_json(tmp_path: Path) -> None:
    report = integrate.apply(["claude-code"], repo_root=tmp_path, mcp=False)
    payload = json.loads(integrate.to_json(report))
    assert payload["workspace"] == str(tmp_path)
    assert payload["actions"]


# ── integrate.doctor ─────────────────────────────────────────────────────────


def test_doctor_is_read_only(tmp_path: Path) -> None:
    rows = integrate.doctor(tmp_path, ["claude-code"])
    assert rows
    assert not (tmp_path / "CLAUDE.md").exists()
    assert not (tmp_path / "AGENTS.md").exists()


def test_doctor_reports_states(tmp_path: Path) -> None:
    before = integrate.doctor(tmp_path, ["claude-code"])[0]
    states = {b["target"]: b["state"] for b in before["briefs"]}
    assert states["CLAUDE.md"] == "missing"

    integrate.apply(["claude-code"], repo_root=tmp_path, mcp=False)

    after = integrate.doctor(tmp_path, ["claude-code"])[0]
    states = {b["target"]: b["state"] for b in after["briefs"]}
    assert states["CLAUDE.md"] == "current"


def test_doctor_flags_foreign_content_without_block(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# Mine\n\nNo block here.\n", encoding="utf-8")
    row = integrate.doctor(tmp_path, ["claude-code"])[0]
    states = {b["target"]: b["state"] for b in row["briefs"]}
    assert states["CLAUDE.md"] == "no-block"


def test_doctor_mcp_state_matches_install(tmp_path: Path) -> None:
    row = integrate.doctor(tmp_path, ["claude-code"])[0]
    assert row["mcp"] == "missing"

    integrate.apply(["claude-code"], repo_root=tmp_path, instructions=False)

    row = integrate.doctor(tmp_path, ["claude-code"])[0]
    assert row["mcp"] == "installed"


def test_doctor_json_serialisable(tmp_path: Path) -> None:
    json.dumps(integrate.doctor(tmp_path))


# ── JSONC tolerance (Zed ships a commented settings.json) ────────────────────


def test_strip_jsonc_preserves_comment_like_text_in_strings() -> None:
    """A `//` inside a value must not be treated as a comment."""
    from tracera.mcp.install import _strip_jsonc

    out = _strip_jsonc('{"url": "https://example.com//path", "n": 1}')
    assert json.loads(out)["url"] == "https://example.com//path"


def test_strip_jsonc_handles_line_and_block_comments() -> None:
    from tracera.mcp.install import _strip_jsonc

    src = '{\n  // line\n  "a": 1, /* block */\n  "b": 2\n}'
    assert json.loads(_strip_jsonc(src)) == {"a": 1, "b": 2}


def test_strip_jsonc_removes_trailing_commas() -> None:
    from tracera.mcp.install import _strip_jsonc

    assert json.loads(_strip_jsonc('{"a": [1, 2,], "b": 3,}')) == {"a": [1, 2], "b": 3}


def test_strip_jsonc_leaves_escaped_quote_intact() -> None:
    from tracera.mcp.install import _strip_jsonc

    assert json.loads(_strip_jsonc(r'{"q": "a \" // b"}'))["q"] == 'a " // b'


def _redirect_host(monkeypatch, key: str, path: Path) -> None:
    """Point a host's config at ``path``. Host is frozen, so swap the entry."""
    import dataclasses

    from tracera.mcp.hosts import HOSTS

    monkeypatch.setitem(HOSTS, key, dataclasses.replace(HOSTS[key], config_path=lambda: path))


def test_zed_config_with_comments_is_readable(tmp_path: Path, monkeypatch) -> None:
    """
    Zed's default settings.json carries a commented header.

    A plain json.loads fails on it, which used to make `apply zed` silently
    skip with 'cannot parse' — the MCP server never got registered.
    """
    from tracera.mcp import install as install_mod
    from tracera.mcp.hosts import HOSTS

    settings = tmp_path / "settings.json"
    settings.write_text(
        '// Zed settings\n// docs: https://zed.dev/docs/configuring-zed\n'
        '{\n  "theme": "One Dark",\n}\n',
        encoding="utf-8",
    )
    _redirect_host(monkeypatch, "zed", settings)

    status, detail = install_mod.inspect_host(HOSTS["zed"], tmp_path)
    assert status == "missing", detail

    result = install_mod.install_into_host(HOSTS["zed"], tmp_path)
    assert result.status == "installed", result.detail

    written = json.loads(settings.read_text(encoding="utf-8"))
    assert "tracera" in written["context_servers"]
    # The original comments survive in the backup.
    assert "// Zed settings" in (tmp_path / "settings.json.bak").read_text(encoding="utf-8")


def test_zed_entry_shape_uses_context_servers(tmp_path: Path, monkeypatch) -> None:
    from tracera.mcp import install as install_mod
    from tracera.mcp.hosts import HOSTS

    settings = tmp_path / "settings.json"
    settings.write_text("{}\n", encoding="utf-8")
    _redirect_host(monkeypatch, "zed", settings)

    install_mod.install_into_host(HOSTS["zed"], tmp_path)

    entry = json.loads(settings.read_text(encoding="utf-8"))["context_servers"]["tracera"]
    assert entry["source"] == "custom"
    assert isinstance(entry["command"], str)
    assert isinstance(entry["args"], list)


def _mcp_handshake(
    workspace: Path, timeout: float = 90.0, argv: list[str] | None = None
) -> dict:
    """
    Speak MCP over stdio exactly as a harness would, and return the replies.

    Subprocess-based on purpose: it exercises the real command line every
    generated config entry points at, including the UTF-8 stdio hygiene. A
    unit test against the in-process server would not catch a protocol-level
    or startup regression.

    ``argv`` overrides the command line, so a caller can spawn the *generated*
    entry rather than a hand-written one. ``workspace`` is then only descriptive.
    """
    import queue
    import subprocess
    import sys
    import threading

    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONUTF8": "1", "TRACERA_MCP_STDIO": "1"}

    if argv is None:
        argv = [
            sys.executable, "-X", "utf8", "-m", "tracera.main",
            "mcp", "serve", "--workspace", str(workspace),
        ]

    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(repo_root),
        text=True,
        bufsize=1,
    )

    inbox: queue.Queue = queue.Queue()
    errors: list[str] = []

    def _read_stdout() -> None:
        try:
            for line in proc.stdout:
                inbox.put(line)
        except (ValueError, OSError):
            pass
        inbox.put(None)

    def _read_stderr() -> None:
        try:
            for line in proc.stderr:
                errors.append(line.rstrip())
        except (ValueError, OSError):
            pass

    threading.Thread(target=_read_stdout, daemon=True).start()
    threading.Thread(target=_read_stderr, daemon=True).start()

    def send(obj: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def recv() -> dict | None:
        try:
            line = inbox.get(timeout=timeout)
        except Exception:
            return None
        if line is None:
            return None
        return json.loads(line)

    try:
        send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "tracera-test", "version": "0"},
            },
        })
        init = recv()
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = recv()
        send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "get_server_status", "arguments": {}},
        })
        call = recv()
        return {"initialize": init, "tools": tools, "call": call}
    finally:
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        _mcp_handshake.last_errors = errors  # type: ignore[attr-defined]


def test_mcp_server_speaks_the_protocol_over_stdio(tmp_path: Path) -> None:
    """
    End-to-end: a client can initialize, list tools and call one.

    This is the claim the whole integration rests on — every harness we write
    config for connects this way. If the server stops speaking MCP, or startup
    breaks, this fails even though the unit tests still pass.
    """
    replies = _mcp_handshake(tmp_path)

    assert replies["initialize"] is not None, (
        "server never replied to initialize; stderr="
        + "\n".join(getattr(_mcp_handshake, "last_errors", [])[-10:])
    )
    result = replies["initialize"].get("result", {})
    assert "protocolVersion" in result, f"bad initialize reply: {result}"
    assert result.get("serverInfo", {}).get("name")

    assert replies["tools"] is not None, "no reply to tools/list"
    tools = replies["tools"].get("result", {}).get("tools", [])
    assert len(tools) == 44, f"expected 44 tools, got {len(tools)}"
    names = [t["name"] for t in tools]
    assert "search_code" in names
    assert "recall_memory" in names
    # Every advertised tool must carry a description, or the model has no
    # basis for choosing it.
    assert all(t.get("description") for t in tools), "some tool has no description"

    assert replies["call"] is not None, "no reply to tools/call"
    call_result = replies["call"].get("result", {})
    assert "content" in call_result, f"bad tools/call reply: {call_result}"
    assert not replies["call"].get("error"), f"tool call errored: {replies['call']}"


def test_opencode_entry_shape_is_argv(tmp_path: Path, monkeypatch) -> None:
    """OpenCode wants one argv array plus an explicit local transport type."""
    from tracera.mcp import install as install_mod
    from tracera.mcp.hosts import HOSTS

    cfg = tmp_path / "opencode.json"
    cfg.write_text("{}\n", encoding="utf-8")
    _redirect_host(monkeypatch, "opencode", cfg)

    install_mod.install_into_host(HOSTS["opencode"], tmp_path)

    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcp"]["tracera"]
    assert entry["type"] == "local"
    assert entry["enabled"] is True
    assert isinstance(entry["command"], list)
    assert entry["command"][0].endswith(("python", "python.exe"))


# ── generated entries must actually be executable ────────────────────────────


def _entry_argv(entry: dict) -> tuple[list[str], dict[str, str]]:
    """
    Reduce any host's entry shape back to the argv a harness would exec.

    Hosts disagree about shape — a separate ``args`` list, one argv array,
    ``env`` vs ``environment`` — so normalise before asserting or launching.
    """
    command = entry["command"]
    if isinstance(command, list):
        argv = [str(part) for part in command]
    else:
        argv = [str(command), *[str(a) for a in entry.get("args", [])]]
    env = entry.get("env") or entry.get("environment") or {}
    return argv, {str(k): str(v) for k, v in env.items()}


def test_every_host_entry_is_a_launchable_stdio_command(tmp_path: Path) -> None:
    """
    Every generated entry must reduce to the same executable stdio command.

    The shape tests above cover Zed and OpenCode individually; this covers the
    other 17, which are plain ``command`` + ``args`` entries. They are the easy
    ones to break silently — a dropped flag is still perfectly valid JSON, so
    the config looks right and only fails when a harness tries to spawn it.
    """
    for key, host in HOSTS.items():
        entry = host.build_entry(tmp_path)
        argv, env = _entry_argv(entry)
        assert argv[0].endswith(("python", "python.exe")), f"{key}: {argv[0]}"
        assert argv[1:3] == ["-X", "utf8"], f"{key}: lost the UTF-8 flag: {argv}"
        assert "tracera.main" in argv, f"{key}: not launching our module: {argv}"
        assert "mcp" in argv and "serve" in argv, f"{key}: {argv}"
        assert argv[-2:] == ["--workspace", str(tmp_path)], f"{key}: {argv}"
        assert env.get("PYTHONUTF8") == "1", f"{key}: missing PYTHONUTF8"


def test_documented_harness_table_matches_the_catalog() -> None:
    """
    The public harness table must list every harness the catalog knows.

    It drifted by one: `vscode-copilot-chat` was added to `HARNESSES` and never
    documented, so the docs quietly under-advertised what `integrate apply`
    will actually wire up. Cheap to check, invisible otherwise.
    """
    page = Path(__file__).resolve().parents[1] / "docs-site/content/docs/integrations.mdx"
    if not page.exists():
        pytest.skip("integrations.mdx not present")

    body = page.read_text(encoding="utf-8")
    missing = [key for key in HARNESSES if f"`{key}`" not in body]
    assert not missing, (
        f"integrations.mdx is missing {len(missing)} harness key(s): {missing}"
    )


@pytest.mark.parametrize("host_key", ["claude-code", "zed", "opencode"])
def test_generated_entry_serves_mcp(host_key: str, tmp_path: Path) -> None:
    """
    Spawn the generated entry exactly as written and complete a handshake.

    These three keys cover every distinct shape the generator emits: plain
    ``command`` + ``args``, Zed's flat ``context_servers`` entry, and OpenCode's
    single argv array. This is the end of the chain — a config file is
    worthless if the harness cannot exec it and get a tool result back.
    """
    entry = HOSTS[host_key].build_entry(tmp_path)
    argv, _entry_env = _entry_argv(entry)  # env is asserted in the test above

    replies = _mcp_handshake(tmp_path, argv=argv)

    init = (replies["initialize"] or {}).get("result", {})
    assert init, (
        f"{host_key}: server never replied to initialize; stderr="
        + "\n".join(getattr(_mcp_handshake, "last_errors", [])[-10:])
    )
    assert init.get("serverInfo", {}).get("name") == "tracera"
    assert init.get("protocolVersion") == "2024-11-05"

    tools = (replies["tools"] or {}).get("result", {}).get("tools", [])
    assert len(tools) >= 40, f"{host_key}: tools/list returned {len(tools)}"

    call_result = (replies["call"] or {}).get("result", {})
    assert "content" in call_result, f"{host_key}: bad tools/call reply: {call_result}"
    assert not replies["call"].get("error"), f"{host_key}: tool errored: {replies['call']}"
