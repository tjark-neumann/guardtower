"""Findings and report objects produced by an audit."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Severity(enum.IntEnum):
    """Ordered so that ``max(severities)`` gives the worst one."""

    INFO = 10
    WARN = 20
    ERROR = 30

    @property
    def label(self) -> str:
        return {Severity.INFO: "INFO", Severity.WARN: "WARN", Severity.ERROR: "FAIL"}[self]

    @property
    def glyph(self) -> str:
        return {Severity.INFO: "·", Severity.WARN: "▲", Severity.ERROR: "✗"}[self]


@dataclass
class Finding:
    """A single thing the auditor noticed."""

    check: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    catalog_id: str | None = None

    def __str__(self) -> str:
        return f"[{self.severity.label}] {self.check}: {self.message}"


class GuardtowerError(AssertionError):
    """Raised by :meth:`Report.raise_if_errors` when ERROR findings exist."""


class Report:
    """Collection of findings with convenience accessors and pretty printing."""

    def __init__(self, findings: list[Finding] | None = None, meta: dict[str, Any] | None = None):
        self.findings: list[Finding] = findings or []
        self.meta: dict[str, Any] = meta or {}

    # -- construction -------------------------------------------------------
    def add(self, check: str, severity: Severity, message: str,
            catalog_id: str | None = None, **details: Any) -> Finding:
        f = Finding(check=check, severity=severity, message=message,
                    details=details, catalog_id=catalog_id)
        self.findings.append(f)
        return f

    # -- views --------------------------------------------------------------
    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARN]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.INFO]

    @property
    def ok(self) -> bool:
        """True when there are no ERROR findings."""
        return not self.errors

    def by_check(self, name: str) -> list[Finding]:
        return [f for f in self.findings if f.check == name]

    # -- actions ------------------------------------------------------------
    def raise_if_errors(self) -> "Report":
        """Raise :class:`GuardtowerError` if any ERROR-level findings exist.

        Returns ``self`` when clean so it chains: ``audit(...).raise_if_errors()``.
        """
        if self.errors:
            lines = "\n".join(f"  ✗ {f.check}: {f.message}" for f in self.errors)
            raise GuardtowerError(
                f"guardtower found {len(self.errors)} blocking issue(s):\n{lines}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "meta": self.meta,
            "findings": [
                {
                    "check": f.check,
                    "severity": f.severity.label,
                    "message": f.message,
                    "catalog_id": f.catalog_id,
                    "details": f.details,
                }
                for f in self.findings
            ],
        }

    # -- display ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    def summary(self) -> str:
        n_e, n_w, n_i = len(self.errors), len(self.warnings), len(self.infos)
        verdict = "PASS" if self.ok else "FAIL"
        return f"guardtower: {verdict}  ({n_e} fail · {n_w} warn · {n_i} info)"

    def __repr__(self) -> str:
        width = 64
        bar = "─" * width
        header = self.summary()
        lines = [bar, header, bar]
        if not self.findings:
            lines.append("  no findings — nothing instrumented?")
        # worst first, stable within a severity
        order = sorted(
            self.findings, key=lambda f: -int(f.severity)
        )
        for f in order:
            tag = f"  [{f.catalog_id}]" if f.catalog_id else ""
            lines.append(f"  {f.severity.glyph} {f.severity.label:<4} {f.check}{tag}")
            lines.append(f"      {f.message}")
        lines.append(bar)
        return "\n".join(lines)
