"""
TRACERA Command Input — premium multiline editor (Claude Code style).

A thin wrapper around ``TextArea`` that turns it into a chat prompt:

    enter      → submit
    shift+enter→ insert a newline (multi-line drafting)
    ctrl+up    → previous command (history)
    ctrl+down  → next command
    tab        → complete the slash command being typed
    esc        → close suggestions (handled by the app)

The widget posts two messages:
    - ``CommandInput.Submit`` when enter is pressed with non-empty text.
    - ``CommandInput.SuggestionsChanged`` whenever the typed text produces a
      (possibly empty) list of slash-command completions, so the panel can
      render a live autocomplete list above the prompt.
"""

from __future__ import annotations

import os

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea

from tracera.tui.widgets.command_registry import COMMAND_ORDER


class CommandInput(TextArea):
    """A multiline prompt with history and slash-command autocomplete."""

    class Submit(Message):
        """Posted when the user submits the buffer (enter)."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class SuggestionsChanged(Message):
        """Posted when slash-command completions change (may be empty)."""

        def __init__(self, matches: list[str]) -> None:
            super().__init__()
            self.matches = matches

    BINDINGS = [
        Binding("ctrl+up", "history_prev", "Previous", show=False),
        Binding("ctrl+down", "history_next", "Next", show=False),
    ]

    def __init__(
        self,
        *,
        commands: dict[str, str] | None = None,
        placeholder: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            "",
            language=None,
            show_line_numbers=False,
            soft_wrap=True,
            tab_behavior="indent",
            placeholder=placeholder or None,
            **kwargs,
        )
        self._commands: dict[str, str] = commands or {}
        self._history: list[str] = []
        self._history_index: int | None = None
        self._draft = ""
        self._matches: list[str] = []
        self._placeholder = placeholder

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_commands(self, commands: dict[str, str]) -> None:
        """Provide the slash-command name → description map for autocomplete."""
        self._commands = commands or {}

    @property
    def match_list(self) -> list[str]:
        return list(self._matches)

    # ── Key handling ──────────────────────────────────────────────────────────

    async def _on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "enter":
            event.stop()
            event.prevent_default()
            self._submit()
            return
        if key == "shift+enter":
            # On some terminals shift+enter is delivered as "enter"; the app
            # handles enter as submit, so a newline is still typed manually.
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if key == "tab":
            if self._matches:
                event.stop()
                event.prevent_default()
                self._complete()
                self._publish_suggestions()
                return
        await super()._on_key(event)

    def _submit(self) -> None:
        raw = self.text
        text = raw.strip()
        if text:
            if not self._history or self._history[-1] != text:
                self._history.append(text)
            self._history_index = None
            self._draft = ""
            self.load_text("")
            self.post_message(self.Submit(text))
        else:
            self.load_text("")

    def _move_to_end(self) -> None:
        try:
            row = max(self.document.line_count - 1, 0)
            col = len(self.document[row])
            self.move_cursor((row, col))
        except Exception:
            pass

    # ── History ───────────────────────────────────────────────────────────────

    def action_history_prev(self) -> None:
        if not self._history:
            return
        if self._history_index is None:
            self._draft = self.text
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return
        self.load_text(self._history[self._history_index])
        self._move_to_end()

    def action_history_next(self) -> None:
        if self._history_index is None:
            return
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self.load_text(self._history[self._history_index])
        else:
            self._history_index = None
            self.load_text(self._draft)
        self._move_to_end()

    # ── Autocomplete ──────────────────────────────────────────────────────────

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._publish_suggestions()

    def _compute_matches(self) -> list[str]:
        text = self.text
        if "\n" in text:
            return []
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return []
        query = stripped[1:]
        if " " in query:
            return []
        names = self._commands or {}
        # Prefer registry order so the most-used commands surface first.
        ordered = [n for n in COMMAND_ORDER if n in names]
        ordered += [n for n in names if n not in ordered]
        matches = [n for n in ordered if n.startswith(query)]
        if not matches:
            matches = [n for n in ordered if query in n]
        return matches[:8]

    def _publish_suggestions(self) -> None:
        self._matches = self._compute_matches()
        self.post_message(self.SuggestionsChanged(self._matches))

    def _complete(self) -> None:
        matches = self._matches
        if not matches:
            return
        if len(matches) == 1:
            replacement = "/" + matches[0] + " "
        else:
            prefix = os.path.commonprefix(matches)
            replacement = "/" + prefix
        self.load_text(replacement)
        self._move_to_end()