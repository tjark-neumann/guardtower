"""Hugging Face ``Trainer`` integration.

Add the callback to a Trainer you already run and guardtower audits the model on
the first real batch, before the expensive part begins::

    from guardtower.integrations.huggingface import GuardtowerCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        callbacks=[GuardtowerCallback(raise_on_error=True)],
    )
    trainer.train()

Importing this module requires ``transformers``; the core ``guardtower`` package
does not.
"""

from __future__ import annotations

import torch

try:
    from transformers import TrainerCallback
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "GuardtowerCallback requires the 'transformers' package. "
        "Install it with: pip install transformers"
    ) from e

from ..audit import audit


def _to_device(batch, device):
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(v.to(device) if torch.is_tensor(v) else v for v in batch)
    return batch.to(device) if torch.is_tensor(batch) else batch


class GuardtowerCallback(TrainerCallback):
    """Run a guardtower pre-flight audit on the first training batch.

    Parameters
    ----------
    raise_on_error
        If True, abort training (raise ``GuardtowerError``) when blocking issues
        are found, so a misconfigured run never starts.
    checks
        Passed through to :func:`guardtower.audit`.
    verbose
        Print the report (default True).
    """

    def __init__(self, *, raise_on_error: bool = False, checks="all", verbose: bool = True):
        self.raise_on_error = raise_on_error
        self.checks = checks
        self.verbose = verbose
        self.report = None
        self._done = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self._done:
            return control
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        train_dataloader = kwargs.get("train_dataloader")
        if model is None or train_dataloader is None:
            if self.verbose:
                print("[guardtower] skipped: model or dataloader unavailable in callback")
            return control

        try:
            batch = next(iter(train_dataloader))
        except StopIteration:
            if self.verbose:
                print("[guardtower] skipped: empty dataloader")
            return control

        device = next(model.parameters()).device
        batch = _to_device(batch, device)

        def step():
            out = model(**batch) if isinstance(batch, dict) else model(batch)
            loss = getattr(out, "loss", None)
            if loss is None and isinstance(out, dict):
                loss = out.get("loss")
            if loss is None:
                raise RuntimeError(
                    "guardtower could not find a 'loss' in the model output. Ensure the "
                    "batch includes labels so the HF model returns a loss."
                )
            return loss

        self.report = audit(model, step, optimizer=optimizer, checks=self.checks)
        self._done = True
        if self.verbose:
            print("\n[guardtower] pre-flight audit on first batch:")
            print(self.report)
        if self.raise_on_error:
            self.report.raise_if_errors()
        return control
