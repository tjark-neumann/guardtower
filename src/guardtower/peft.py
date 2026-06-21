"""Checks specific to parameter-efficient fine-tuning (LoRA / PEFT / QLoRA).

These run automatically inside :func:`guardtower.audit` when adapter parameters
are detected, and are also exposed directly via :func:`guardtower.lora_summary`
for a quick, training-free sanity check.
"""

from __future__ import annotations

import torch
from torch import nn

from .report import Report, Severity

# Substrings that mark an adapter (LoRA, DoRA, etc.) parameter. Matches the
# real PEFT naming, e.g. "...q_proj.lora_A.default.weight".
ADAPTER_MARKERS = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",  # DoRA
)

# Class names PEFT uses for the wrapper model.
PEFT_CLASS_NAMES = (
    "PeftModel",
    "PeftModelForCausalLM",
    "PeftModelForSequenceClassification",
    "PeftModelForTokenClassification",
    "PeftMixedModel",
    "LoraModel",
)


def is_peft_model(model: nn.Module) -> bool:
    """True if the model looks like a PEFT/LoRA model (no peft import needed)."""
    if type(model).__name__ in PEFT_CLASS_NAMES:
        return True
    for name, _ in model.named_parameters():
        if any(m in name for m in ADAPTER_MARKERS):
            return True
    return False


def _partition(model: nn.Module):
    """Split params into (adapter, trainable_extra, base)."""
    adapter, extra, base = [], [], []
    for name, p in model.named_parameters():
        if any(m in name for m in ADAPTER_MARKERS):
            adapter.append((name, p))
        elif "modules_to_save" in name and "original_module" not in name:
            # PEFT's intentionally-trainable extra modules (e.g. a new head)
            extra.append((name, p))
        else:
            base.append((name, p))
    return adapter, extra, base


def _fmt_pct(trainable: int, total: int) -> str:
    pct = (100.0 * trainable / total) if total else 0.0
    return f"{trainable:,} / {total:,} ({pct:.3f}%)"


def peft_checks(
    model: nn.Module,
    rep: Report,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    static_only: bool = False,
) -> bool:
    """Run LoRA/PEFT checks, appending findings to ``rep``.

    Returns True if this looked like a PEFT model (so callers know whether the
    adapter-specific checks applied).
    """
    if not is_peft_model(model):
        return False

    adapter, extra, base = _partition(model)

    # 1) Are the adapters actually trainable? --------------------------------
    adapter_trainable = [(n, p) for n, p in adapter if p.requires_grad]
    if adapter and not adapter_trainable:
        rep.add(
            "lora_adapter", Severity.ERROR, catalog_id="LORA-002",
            message="LoRA/PEFT adapters are present but ALL frozen "
            "(requires_grad=False) — nothing will train. Did you call "
            "model.eval()/freeze everything, or forget to enable the adapter?",
            params=[n for n, _ in adapter],
        )
    elif adapter_trainable:
        rep.add(
            "lora_adapter", Severity.INFO,
            f"{len(adapter_trainable)} trainable adapter tensor(s) detected "
            f"(e.g. {adapter_trainable[0][0]})",
        )

    # 2) Is the base model accidentally unfrozen? ----------------------------
    base_trainable = [(n, p) for n, p in base if p.requires_grad and p.numel() > 0]
    if base_trainable:
        n_elems = sum(p.numel() for _, p in base_trainable)
        rep.add(
            "lora_base_frozen", Severity.WARN, catalog_id="LORA-001",
            message=f"the BASE model is not fully frozen: {len(base_trainable)} base "
            f"tensor(s) / {n_elems:,} weights still require grad. With LoRA you "
            f"usually want only the adapter trainable — this blows up memory and "
            f"defeats the point (e.g. {base_trainable[0][0]})",
            params=[n for n, _ in base_trainable],
        )

    # 3) Overall trainable% ---------------------------------------------------
    total = sum(p.numel() for _, p in model.named_parameters())
    trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    pct = (100.0 * trainable / total) if total else 0.0
    verdict = "looks like LoRA ✓" if pct < 5 else (
        "unusually high for LoRA — is the base unfrozen?" if pct > 30 else "ok"
    )
    sev = Severity.WARN if pct > 30 else Severity.INFO
    rep.add(
        "lora_trainable_pct", sev,
        f"trainable parameters: {_fmt_pct(trainable, total)} — {verdict}",
        trainable=trainable, total=total, pct=pct,
    )

    # 4) Are the trainable adapters actually in the optimizer? ---------------
    if optimizer is not None and adapter_trainable:
        opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        missing = [n for n, p in adapter_trainable if id(p) not in opt_ids]
        if missing:
            rep.add(
                "lora_optimizer", Severity.ERROR, catalog_id="LORA-003",
                message=f"{len(missing)} trainable adapter tensor(s) are NOT in the "
                f"optimizer and will never update (e.g. {missing[0]}). Build the "
                f"optimizer from the PEFT model AFTER get_peft_model(...)",
                params=missing,
            )
        else:
            rep.add("lora_optimizer", Severity.INFO, "optimizer covers the adapter params")

    # 5) Mixed dtype between base and adapter (QLoRA is fine; just inform) ----
    if not static_only:
        base_dtypes = {str(p.dtype) for _, p in base if p.is_floating_point()}
        adp_dtypes = {str(p.dtype) for _, p in adapter if p.is_floating_point()}
        if base_dtypes and adp_dtypes and base_dtypes != adp_dtypes:
            rep.add(
                "lora_dtype", Severity.INFO,
                f"base dtype(s) {sorted(base_dtypes)} differ from adapter "
                f"{sorted(adp_dtypes)} — expected for QLoRA, worth a glance otherwise",
            )

    return True


def lora_summary(
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> Report:
    """Training-free LoRA/PEFT sanity check.

        print(guardtower.lora_summary(model, optimizer=opt))
    """
    rep = Report(meta={"mode": "lora_summary"})
    found = peft_checks(model, rep, optimizer=optimizer, static_only=True)
    if not found:
        rep.add(
            "lora", Severity.WARN,
            "no LoRA/PEFT adapters detected — is this a PEFT model? "
            "(looked for lora_A/lora_B parameters and PeftModel wrappers)",
        )
    return rep
