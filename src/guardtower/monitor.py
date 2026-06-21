"""Live monitoring during a real training loop (not just a pre-flight step)."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .audit import _iter_tensors


class NonFiniteActivation(RuntimeError):
    """Raised when a monitored module emits NaN/Inf and on_nonfinite='raise'."""


class monitor:
    """Context manager that watches forward activations for NaN/Inf live.

    Cheap enough to leave on for the first few hundred steps of a real run so a
    blow-up is pinpointed to the exact module the instant it happens, instead of
    surfacing as a useless ``loss=nan`` several layers downstream::

        with guardtower.monitor(model, on_nonfinite="raise"):
            for batch in loader:
                loss = train_step(batch)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        on_nonfinite: Literal["raise", "warn"] = "raise",
    ):
        self.model = model
        self.on_nonfinite = on_nonfinite
        self._handles: list = []
        self.hits: list[str] = []

    def __enter__(self) -> "monitor":
        for name, mod in self.model.named_modules():
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))
        return self

    def _make_hook(self, name: str):
        def hook(mod, inp, out):
            for t in _iter_tensors(out):
                if t.is_floating_point() and not torch.isfinite(t).all():
                    label = name or type(mod).__name__
                    self.hits.append(label)
                    msg = f"non-finite activation in module '{label}' ({type(mod).__name__})"
                    if self.on_nonfinite == "raise":
                        raise NonFiniteActivation(msg)
                    import warnings

                    warnings.warn(msg, stacklevel=2)
                    break
        return hook

    def __exit__(self, *exc) -> bool:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False
