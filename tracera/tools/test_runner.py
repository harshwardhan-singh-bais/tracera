"""
Phases 32-34 — Test Discovery, Execution, and Failure Analysis.

Provides tools for the autonomous agent to:
  - Phase 32: Discover test suites (pytest, unittest, npm, cargo)
  - Phase 33: Execute tests safely in the sandbox
  - Phase 34: Parse failures into structured FailureReport objects
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tracera.logging import get_logger

log = get_logger("tools.test_runner")

TestFramework = Literal["pytest", "unittest", "npm", "cargo", "unknown"]


@dataclass
class TestFailure:
    """Structured representation of a single test failure."""
    test_name: str
    error_type: str
    error_message: str
    file_path: str = ""
    line_number: int = 0
    stack_trace: str = ""


@dataclass
class TestReport:
    """Full test run report."""
    framework: TestFramework
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total: int = 0
    duration_seconds: float = 0.0
    failures: list[TestFailure] = field(default_factory=list)
    raw_output: str = ""
    success: bool = False

    @property
    def summary(self) -> str:
        icon = "✅" if self.success else "❌"
        return (
            f"{icon} {self.framework}: "
            f"{self.passed}/{self.total} passed"
            + (f", {self.failed} failed" if self.failed else "")
            + (f", {self.errors} errors" if self.errors else "")
        )


class TestDiscovery:
    """Phase 32: Discover which test framework(s) a project uses."""

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root

    def detect_framework(self) -> TestFramework:
        """Auto-detect the primary test framework."""
        root = self._root

        if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
            if (root / "tests").exists() or list(root.rglob("test_*.py")):
                return "pytest"

        if list(root.rglob("test_*.py")) or list(root.rglob("*_test.py")):
            return "pytest"

        if (root / "package.json").exists():
            try:
                pkg = (root / "package.json").read_text()
                if '"test"' in pkg:
                    return "npm"
            except Exception:
                pass

        if (root / "Cargo.toml").exists():
            return "cargo"

        return "unknown"

    def get_test_command(self, framework: TestFramework | None = None) -> list[str]:
        """Return the command to run tests for the detected framework."""
        fw = framework or self.detect_framework()
        commands = {
            "pytest": ["python", "-m", "pytest", "-v", "--tb=short"],
            "unittest": ["python", "-m", "unittest", "discover", "-v"],
            "npm": ["npm", "test"],
            "cargo": ["cargo", "test"],
            "unknown": [],
        }
        return commands.get(fw, [])


class TestRunner:
    """Phase 33: Execute tests safely within the workspace sandbox."""

    def __init__(
        self,
        workspace_root: Path,
        timeout: int = 120,
    ) -> None:
        self._root = workspace_root
        self._timeout = timeout
        self._discovery = TestDiscovery(workspace_root)

    def run(
        self,
        framework: TestFramework | None = None,
        test_paths: list[str] | None = None,
    ) -> TestReport:
        """
        Run the test suite and return a structured report.

        Args:
            framework: Override the auto-detected framework.
            test_paths: Specific test files/dirs to run.

        Returns:
            TestReport with parsed results.
        """
        fw = framework or self._discovery.detect_framework()
        cmd = self._discovery.get_test_command(fw)

        if not cmd:
            return TestReport(
                framework="unknown",
                raw_output="Could not detect a supported test framework.",
                success=False,
            )

        if test_paths:
            cmd.extend(test_paths)

        log.info("Running tests: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            output = result.stdout + result.stderr
            report = FailureAnalyzer.parse(output, fw)
            report.success = result.returncode == 0
            return report
        except subprocess.TimeoutExpired:
            return TestReport(
                framework=fw,
                raw_output=f"Test run timed out after {self._timeout}s.",
                success=False,
            )
        except Exception as e:
            return TestReport(
                framework=fw,
                raw_output=f"Test execution error: {e}",
                success=False,
            )


class FailureAnalyzer:
    """Phase 34: Parse raw test output into structured FailureReport objects."""

    @staticmethod
    def parse(output: str, framework: TestFramework) -> TestReport:
        """Parse raw test output into a structured TestReport."""
        if framework == "pytest":
            return FailureAnalyzer._parse_pytest(output)
        # Fallback for other frameworks
        return TestReport(framework=framework, raw_output=output, success="passed" in output.lower())

    @staticmethod
    def _parse_pytest(output: str) -> TestReport:
        report = TestReport(framework="pytest", raw_output=output)

        # Parse summary line: "5 passed, 2 failed in 1.23s"
        summary_match = re.search(
            r"(\d+) passed(?:,\s*(\d+) failed)?(?:,\s*(\d+) error)?.*?in ([\d.]+)s",
            output,
        )
        if summary_match:
            report.passed = int(summary_match.group(1) or 0)
            report.failed = int(summary_match.group(2) or 0)
            report.errors = int(summary_match.group(3) or 0)
            report.total = report.passed + report.failed + report.errors
            report.duration_seconds = float(summary_match.group(4) or 0)

        # Parse individual failures
        failure_blocks = re.split(r"_{20,}", output)
        for block in failure_blocks:
            if "FAILED" not in block and "ERROR" not in block:
                continue

            # Extract test name
            name_match = re.search(r"FAILED (.+?)(?:\s|$)", block)
            test_name = name_match.group(1).strip() if name_match else "unknown"

            # Extract error type and message
            error_match = re.search(r"([\w.]+Error|AssertionError|Exception):\s*(.+)", block)
            error_type = error_match.group(1) if error_match else "TestError"
            error_message = error_match.group(2).strip() if error_match else ""

            # Extract file path and line
            file_match = re.search(r"File \"(.+?)\", line (\d+)", block)
            file_path = file_match.group(1) if file_match else ""
            line_num = int(file_match.group(2)) if file_match else 0

            report.failures.append(TestFailure(
                test_name=test_name,
                error_type=error_type,
                error_message=error_message,
                file_path=file_path,
                line_number=line_num,
                stack_trace=block[:1000],
            ))

        return report
