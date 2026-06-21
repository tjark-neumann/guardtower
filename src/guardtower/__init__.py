"""guardtower: catch the silent bugs that waste GPU-hours before you launch.

Quickstart
----------
    import guardtower

    report = guardtower.audit(
        model,
        lambda: loss_fn(model(x), y),
        optimizer=optimizer,
        inputs=x,
    )
    print(report)
    report.raise_if_errors()   # fail fast in CI / at the top of train.py
"""

from __future__ import annotations

from .audit import ALL_CHECKS, audit
from .catalog import CATALOG, FailureMode, catalog, catalog_markdown
from .monitor import NonFiniteActivation, monitor
from .peft import is_peft_model, lora_summary
from .report import Finding, Report, Severity, GuardtowerError

__all__ = [
    "audit",
    "monitor",
    "lora_summary",
    "is_peft_model",
    "catalog",
    "catalog_markdown",
    "CATALOG",
    "FailureMode",
    "ALL_CHECKS",
    "Report",
    "Finding",
    "Severity",
    "GuardtowerError",
    "NonFiniteActivation",
]

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("guardtower")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"


def __getattr__(name):
    # Expose the HF callback lazily so transformers isn't imported at package load.
    if name == "GuardtowerCallback":
        from .integrations.huggingface import GuardtowerCallback
        return GuardtowerCallback
    raise AttributeError(f"module 'guardtower' has no attribute {name!r}")
