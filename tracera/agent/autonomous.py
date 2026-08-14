"""
Phases 35-38 — Autonomous Engineering Loop.

Phase 35: Retrieval-Driven Debugging — failure → retrieve → reason → patch
Phase 36: Autonomous Fix Loop — plan → edit → test → retry on failure
Phase 37: Self-Review — independent post-implementation code review
Phase 38: Regression Protection — pre/post diff + affected symbol detection
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tracera.tools.test_runner import FailureAnalyzer, TestFailure, TestReport, TestRunner
from tracera.logging import get_logger

log = get_logger("agent.autonomous")


# ── Phase 35: Retrieval-Driven Debugging ─────────────────────────────────────

@dataclass
class DebugPlan:
    """A plan for fixing a specific test failure."""
    failure: TestFailure
    retrieved_context: str
    hypothesis: str = ""
    proposed_patch: str = ""


class RetrievalDebugger:
    """
    Phase 35: Given a test failure, retrieves relevant code and constructs
    a structured debug plan with a proposed patch.
    """

    def __init__(self, retriever: Any, context_engine: Any) -> None:
        self._retriever = retriever
        self._context = context_engine

    def build_debug_plan(self, failure: TestFailure, provider: Any) -> DebugPlan:
        """
        For a given TestFailure:
        1. Search for the failing symbol in the retrieval index.
        2. Retrieve its implementation + dependencies.
        3. Build a DebugPlan with retrieved context.
        """
        query = f"{failure.test_name} {failure.error_type} {failure.error_message}"
        chunks = self._retriever.search(query, k=8)
        context_str = self._context.assemble(chunks, query=f"Fix: {failure.error_message}")

        return DebugPlan(
            failure=failure,
            retrieved_context=context_str,
            hypothesis=f"The failure '{failure.error_message}' likely originates in {failure.file_path}:{failure.line_number}",
        )


# ── Phase 36: Autonomous Fix Loop ────────────────────────────────────────────

@dataclass
class FixAttempt:
    """Record of a single fix attempt."""
    iteration: int
    test_report: TestReport
    success: bool
    patch_description: str = ""


@dataclass
class AutonomousFixResult:
    """Result of running the full autonomous fix loop."""
    task: str
    attempts: list[FixAttempt] = field(default_factory=list)
    final_success: bool = False
    total_iterations: int = 0


class AutonomousFixLoop:
    """
    Phase 36: Implements the autonomous fix loop.

    Plan → Retrieve → Edit → Test → (Failure? → Retrieve → Fix → Test) → Continue
    """

    def __init__(
        self,
        workspace_root: Path,
        test_runner: TestRunner,
        debugger: RetrievalDebugger,
        max_iterations: int = 5,
    ) -> None:
        self._workspace = workspace_root
        self._test_runner = test_runner
        self._debugger = debugger
        self._max_iterations = max_iterations

    async def run(self, task: str, provider: Any, agent: Any) -> AutonomousFixResult:
        """
        Run the autonomous fix loop.

        Args:
            task: The coding task description.
            provider: LLM provider for reasoning.
            agent: The ReAct agent instance for code editing.

        Returns:
            AutonomousFixResult with all attempts documented.
        """
        result = AutonomousFixResult(task=task)

        for i in range(self._max_iterations):
            log.info("Autonomous fix loop: iteration %d/%d", i + 1, self._max_iterations)

            # Run tests
            report = self._test_runner.run()
            attempt = FixAttempt(
                iteration=i + 1,
                test_report=report,
                success=report.success,
            )
            result.attempts.append(attempt)
            result.total_iterations = i + 1

            if report.success:
                log.info("All tests passing on iteration %d!", i + 1)
                result.final_success = True
                break

            if not report.failures:
                log.warning("Tests failed but no structured failures found. Stopping.")
                break

            # Build debug plans for each failure
            failure = report.failures[0]  # Focus on first failure
            log.info("Debugging failure: %s — %s", failure.test_name, failure.error_message)

            debug_plan = self._debugger.build_debug_plan(failure, provider)
            attempt.patch_description = debug_plan.hypothesis

            # Instruct the agent to fix based on debug plan
            fix_task = (
                f"Fix this test failure:\n"
                f"Test: {failure.test_name}\n"
                f"Error: {failure.error_type}: {failure.error_message}\n"
                f"File: {failure.file_path}:{failure.line_number}\n\n"
                f"Relevant code context:\n{debug_plan.retrieved_context}"
            )

            try:
                await agent.run(fix_task)
            except Exception as e:
                log.error("Agent fix attempt failed: %s", e)
                break

        return result


# ── Phase 37: Self-Review ─────────────────────────────────────────────────────

class SelfReviewer:
    """
    Phase 37: After implementation, runs an independent review of the changes.

    Gets a git diff of what was changed, retrieves related context, and asks
    the LLM to critique the implementation for bugs or design issues.
    """

    def __init__(self, workspace_root: Path, retriever: Any) -> None:
        self._workspace = workspace_root
        self._retriever = retriever

    def get_diff(self) -> str:
        """Get git diff of unstaged changes."""
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(self._workspace),
                capture_output=True, text=True,
            )
            return result.stdout[:8000]  # Limit diff size
        except Exception as e:
            return f"Could not get diff: {e}"

    async def review(self, provider: Any, implementation_summary: str = "") -> str:
        """
        Run an independent code review on the current changes.

        Returns the reviewer's assessment as a string.
        """
        diff = self.get_diff()
        if not diff.strip():
            return "No changes detected. Nothing to review."

        prompt = (
            "You are a senior engineer performing a critical code review.\n"
            "Review the following diff for: bugs, edge cases, security issues, "
            "missing error handling, and architectural concerns.\n\n"
            f"**Implementation Summary:** {implementation_summary}\n\n"
            f"**Git Diff:**\n```diff\n{diff}\n```\n\n"
            "Respond with:\n"
            "1. A severity assessment (PASS / MINOR_ISSUES / MAJOR_ISSUES)\n"
            "2. A bullet list of specific issues found\n"
            "3. Suggested fixes for each issue"
        )

        from tracera.providers.base import LLMMessage, Role
        messages = [LLMMessage(role=Role.USER, content=prompt)]

        try:
            response = await provider.complete(messages, temperature=0.1)
            return response.content or "No review response."
        except Exception as e:
            return f"Review failed: {e}"


# ── Phase 38: Regression Protection ─────────────────────────────────────────

class RegressionProtector:
    """
    Phase 38: Before and after a coding task, captures the test state
    and the set of changed symbols. Ensures no existing tests were broken.
    """

    def __init__(self, workspace_root: Path, test_runner: TestRunner) -> None:
        self._workspace = workspace_root
        self._test_runner = test_runner
        self._pre_report: TestReport | None = None

    def snapshot_before(self) -> TestReport:
        """Run tests before the task and capture the baseline."""
        self._pre_report = self._test_runner.run()
        log.info("Pre-task baseline: %s", self._pre_report.summary)
        return self._pre_report

    def get_changed_files(self) -> list[str]:
        """Get list of changed files from git."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=str(self._workspace),
                capture_output=True, text=True,
            )
            return [f.strip() for f in result.stdout.splitlines() if f.strip()]
        except Exception:
            return []

    def verify_after(self) -> dict:
        """
        Run tests after the task and compare against baseline.

        Returns a regression report dict.
        """
        post_report = self._test_runner.run()
        changed_files = self.get_changed_files()

        # Compare against baseline
        pre_passed = self._pre_report.passed if self._pre_report else 0
        regression_count = max(0, pre_passed - post_report.passed)

        report = {
            "pre_passed": pre_passed,
            "post_passed": post_report.passed,
            "post_failed": post_report.failed,
            "regressions": regression_count,
            "changed_files": changed_files,
            "overall_success": post_report.success and regression_count == 0,
            "summary": post_report.summary,
        }

        if regression_count > 0:
            log.warning(
                "REGRESSION DETECTED: %d previously passing tests now failing!",
                regression_count,
            )
        else:
            log.info("No regressions detected. %s", post_report.summary)

        return report
