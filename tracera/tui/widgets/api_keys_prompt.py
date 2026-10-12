"""Interactive screen for configuring provider API keys at runtime.

Use ``/api-keys`` in the TUI to open this screen, type a provider name
followed by ``=<value>`` (or ``done`` when finished), and the key pairs are
written straight into the process environment. They are then visible to the
agent without restarting.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.message import Message
from textual.widgets import Input, Static

from tracera.config.settings import get_settings
from tracera.tui import theme as tui_theme
from tracera.tui.widgets.agent_panel import AgentPanel

#: Environment-variable names for the providers the app supports.
_PROVIDER_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "SAMBANOVA_API_KEY",
    "MISTRAL_API_KEY",
    "CEREBRAS_API_KEY",
    "NVIDIA_API_KEY",
    "NEMOTRON_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "COHERE_API_KEY",
]

#: Explicit user responses that exit the prompt screen.
_SKIP_MARKER = "skip"



class ApiKeysPromptScreen(Screen):
    """Modal screen that lets a first-time user type provider keys in-place.

    Each line is ``name=value`` (the name is the UPPER_SNAKE env var, e.g.
    ``OPENAI_API_KEY``); an empty line or ``done`` closes the screen.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "submit", "Submit", show=False),
    ]

    class Submitted(Message):
        """Posted to the parent app with the configuration key pairs."""

        def __init__(self, pairs: dict[str, str]):
            super().__init__()
            self.pairs = pairs

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pairs: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        from tracera.tui.theme import get_theme as _gt

        th = _gt()
        yield Static(" Add configuration API keys ", id="api-keys-title")
        yield Static(
            "Type [bold]'OPENAI_API_KEY=my-key'[/] for a provider, "
            "or just [bold]done[/] when finished.",
            id="api-keys-hint",
        )
        yield Input(placeholder="e.g. GROQ_API_KEY=gsk_...", id="api-keys-input")
        yield Static("↑↓ navigate · enter submit · esc cancel · skip: move without key",
                     id="api-keys-hint")

    def on_mount(self) -> None:
        self.query_one("#api-keys-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Delegated so the single "enter" binding above matches the app's
        # focus/keystroke flow; avoids double-submission.
        self._submit(event.value)

    def action_submit(self) -> None:
        self._submit(self.query_one("#api-keys-input", Input).value)

    def action_cancel(self) -> None:
        self.post_message(Message("cancel"))

    def _submit(self, raw: str) -> None:
        value = raw.strip()
        if not value:
            return
        if value.lower() == _SKIP_MARKER:
            self._commit(skip=True)
            return
        if value.lower() == "done":
            self._commit(skip=False)
            return
        if "=" not in value:
            self.query_one("#api-keys-input", Input).value = ""
            self.query_one("#api-keys-input", Input).focus()
            return
        key, _, val = value.partition("=")
        key = key.strip().upper()
        val = val.strip()
        if not key or not val:
            self.query_one("#api-keys-input", Input).value = ""
            self.query_one("#api-keys-input", Input).focus()
            return
        self._pairs[key] = val
        self.query_one("#api-keys-input", Input).value = ""
        self._refresh_hint()
        self.query_one("#api-keys-input", Input).focus()

    def _refresh_hint(self) -> None:
        from tracera.tui.theme import get_theme as _gt

        th = _gt()
        self._hint = f"[bold thin]📋 {len(self._pairs)} configured[/]"
        self.query_one("#api-keys-hint", Static).update(self._hint)

    def _commit(self, *, skip: bool) -> None:
        added = {k: v for k, v in self._pairs.items() if k in _PROVIDER_ENV_VARS}

        if not added and not skip:
            self.post_message(self.Submitted({}))
            self._dismiss()
            return

        try:
            settings = get_settings()
        except Exception as e:
            self.query_one("#api-keys-input", Input).value = ""
            self.query_one("#api-keys-input", Input).focus()
            self.post_message(self.Submitted({}))
            self.app.push_screen(self, callback=lambda r: self._dismiss(e))
            return

        # 1) Persist to the workspace .env so the change survives a restart.
        env_path = Path(settings.tracera_workspace) / ".env"
        try:
            lines = []
            if env_path.exists():
                lines = env_path.read_text(encoding="utf-8").splitlines()

            written: set[str] = set()
            out: list[str] = []
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    out.append(line)
                    continue
                # Keep existing keys in place; overwrite on the next write.
                existing_key = stripped.split("=", 1)[0].strip()
                if existing_key in added:
                    out.append(f"{existing_key}={added[existing_key]}")
                    written.add(existing_key)
                else:
                    out.append(line)

            for key, val in added.items():
                if key not in written:
                    out.append(f"{key}={val}")

            env_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
        except Exception:
            # If .env can't be written, still propagate the in-memory update.
            pass

        # 2) Sync into the in-memory settings singleton so it is visible
        #    immediately to the running agent.
        try:
            refresh = get_settings()
            for key, val in added.items():
                try:
                    object.__setattr__(refresh, key.lower(), val)
                except Exception:
                    pass
        except Exception:
            pass

        if skip and not added:
            self.post_message(self.Submitted({}))
            self._dismiss()
            return

        self.post_message(self.Submitted(added))
        self._dismiss()

    def action_cancel(self) -> None:
        self.dismiss()

    def _dismiss(self) -> None:
        self.app.pop_screen()
