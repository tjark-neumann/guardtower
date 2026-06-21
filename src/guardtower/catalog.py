"""The catalog of silent training-failure modes — the knowledge asset.

Each check in guardtower traces to a :class:`FailureMode` here (by ``catalog_id``),
so every finding carries *why* it matters, *how* to fix it, and a documented
reference. The catalog is the durable value: looping over parameters is trivial;
knowing the specific ways fine-tuning quietly fails — and keeping that list
current — is not.

Contributing a failure mode = add one entry here + a check that emits its id.
"""

from __future__ import annotations

from dataclasses import dataclass

from .report import Severity


@dataclass(frozen=True)
class FailureMode:
    id: str
    title: str
    severity: Severity
    summary: str
    why: str
    fix: str
    reference: str
    tags: tuple


def _m(**kw):
    return FailureMode(**kw)


CATALOG = {m.id: m for m in [
    # -- core wiring -------------------------------------------------------
    _m(id="LOSS-001", title="Loss is detached from the parameters",
       severity=Severity.ERROR, tags=("core", "autograd"),
       summary="The loss tensor does not require grad, so backward() updates nothing.",
       why="A stray .detach(), .item(), or torch.no_grad() severs the graph; "
           "training runs but no parameter ever changes.",
       fix="Compute the loss from model outputs without detaching it.",
       reference="https://pytorch.org/docs/stable/notes/autograd.html"),
    _m(id="LOSS-002", title="Loss is not a scalar",
       severity=Severity.ERROR, tags=("core",),
       summary="backward() needs a scalar; a non-reduced loss raises or misbehaves.",
       why="Forgetting reduction='mean' (or a .mean()) leaves a per-element loss.",
       fix="Reduce the loss to a single number before backward().",
       reference="https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html"),
    _m(id="INPUT-001", title="Loss does not depend on the input",
       severity=Severity.ERROR, tags=("core",),
       summary="Perturbing the input does not change the loss — the model ignores it.",
       why="A wiring bug (wrong tensor used, input overwritten) makes the model "
           "learn a constant; metrics can still move, hiding the fault.",
       fix="Trace the forward path from the input tensor to the loss.",
       reference="https://pytorch.org/docs/stable/notes/autograd.html"),
    _m(id="GRAD-001", title="Parameter receives no gradient",
       severity=Severity.ERROR, tags=("core", "gradients"),
       summary="A trainable parameter is disconnected from the loss (grad is None).",
       why="A sub-module is built but never called, or its output is dropped — it "
           "occupies optimizer state yet never learns. Invisible in the source.",
       fix="Trace the forward path; either use the module or stop marking it trainable.",
       reference="https://pytorch.org/docs/stable/notes/autograd.html"),
    _m(id="GRAD-004", title="Parameter has an all-zero gradient",
       severity=Severity.WARN, tags=("gradients",),
       summary="Gradient exists but is exactly zero this step.",
       why="Often a masking/detach bug or a saturated path upstream.",
       fix="Check for masks that zero the contribution, or a dead activation.",
       reference="https://pytorch.org/docs/stable/notes/autograd.html"),
    # -- optimizer ---------------------------------------------------------
    _m(id="OPT-001", title="Trainable parameter is missing from the optimizer",
       severity=Severity.ERROR, tags=("core", "optimizer"),
       summary="A requires_grad parameter is not in any optimizer param group.",
       why="Building the optimizer before adding/replacing a module (or filtering "
           "params wrong) means it never updates, even though its grad is computed.",
       fix="Construct the optimizer from the final model.parameters().",
       reference="https://pytorch.org/docs/stable/optim.html"),
    _m(id="OPT-002", title="Frozen parameter is registered with the optimizer",
       severity=Severity.WARN, tags=("optimizer",),
       summary="A requires_grad=False parameter sits in the optimizer.",
       why="Harmless for updates, but weight decay can still act on it and it "
           "wastes optimizer state memory.",
       fix="Pass only trainable params to the optimizer.",
       reference="https://pytorch.org/docs/stable/optim.html"),
    # -- numerics ----------------------------------------------------------
    _m(id="NAN-001", title="NaN/Inf in forward activations",
       severity=Severity.ERROR, tags=("numerics",),
       summary="A module emits non-finite activations.",
       why="Surfaces downstream as a useless loss=nan; the real culprit is the "
           "first module to go non-finite.",
       fix="Inspect the named module; common causes are log/div by zero, fp16 "
           "overflow, or bad init.",
       reference="https://pytorch.org/docs/stable/generated/torch.isfinite.html"),
    _m(id="NAN-002", title="NaN/Inf in gradients",
       severity=Severity.ERROR, tags=("numerics", "gradients"),
       summary="Gradients become non-finite during backward.",
       why="A finite forward can still produce non-finite grads (sqrt(0), log near "
           "0), silently poisoning the optimizer state.",
       fix="Add epsilons to unstable ops; consider gradient clipping.",
       reference="https://pytorch.org/docs/stable/amp.html"),
    # -- gradients ---------------------------------------------------------
    _m(id="GRAD-002", title="Exploding gradient norm",
       severity=Severity.WARN, tags=("gradients",),
       summary="A gradient norm is very large on the first step.",
       why="Predicts divergence/NaN within a few steps.",
       fix="Lower the learning rate or apply clip_grad_norm_.",
       reference="https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html"),
    _m(id="GRAD-003", title="Vanishing gradient norm",
       severity=Severity.WARN, tags=("gradients",),
       summary="The global gradient norm is near zero.",
       why="Little or no learning signal reaches the parameters.",
       fix="Check activations/normalization and that the loss is connected.",
       reference="https://pytorch.org/docs/stable/notes/autograd.html"),
    # -- activations -------------------------------------------------------
    _m(id="RELU-001", title="Dead ReLU units",
       severity=Severity.WARN, tags=("activations",),
       summary="A ReLU outputs almost all zeros for the batch.",
       why="Units stuck negative pass no gradient and never recover, wasting "
           "capacity — usually bad init or too-high learning rate.",
       fix="Check initialization/LR; consider LeakyReLU/GELU.",
       reference="https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html"),
    _m(id="TIE-001", title="Tied (shared-storage) weights",
       severity=Severity.INFO, tags=("core",),
       summary="Two parameters share the same storage.",
       why="Common and correct for LM embedding/output tying, but a surprise tie "
           "couples updates you meant to keep separate.",
       fix="Confirm the tie is intentional.",
       reference="https://arxiv.org/abs/1608.05859"),
    # -- LoRA / PEFT -------------------------------------------------------
    _m(id="LORA-001", title="Base model is not frozen under LoRA",
       severity=Severity.WARN, tags=("lora", "peft"),
       summary="Base (non-adapter) parameters still require grad.",
       why="You think you're doing cheap LoRA but are full-fine-tuning: memory "
           "blows up and the parameter efficiency is lost.",
       fix="Freeze the base; get_peft_model() does this — don't re-enable grad after.",
       reference="https://huggingface.co/docs/peft/main/en/developer_guides/troubleshooting"),
    _m(id="LORA-002", title="LoRA adapter is frozen",
       severity=Severity.ERROR, tags=("lora", "peft"),
       summary="Adapter params are present but all have requires_grad=False.",
       why="Everything was frozen (or the adapter disabled) — the run trains nothing.",
       fix="Enable the adapter; ensure its parameters require grad.",
       reference="https://huggingface.co/docs/peft/main/en/index"),
    _m(id="LORA-003", title="LoRA adapter is missing from the optimizer",
       severity=Severity.ERROR, tags=("lora", "peft", "optimizer"),
       summary="Trainable adapter params are not in any optimizer group.",
       why="Optimizer built from the wrong params (or before get_peft_model) — the "
           "adapter computes gradients that are never applied.",
       fix="Build the optimizer from the PEFT model after get_peft_model(...).",
       reference="https://huggingface.co/docs/peft/main/en/index"),
    _m(id="LORA-004", title="Adapter zero gradient at initialization (expected)",
       severity=Severity.INFO, tags=("lora", "peft"),
       summary="LoRA's A matrix has no gradient on the first step.",
       why="LoRA initializes B to zero, so A legitimately sees no gradient until B "
           "moves. guardtower reports this as expected, not a bug.",
       fix="None — this is correct LoRA behavior.",
       reference="https://arxiv.org/abs/2106.09685"),
    _m(id="LORA-005", title="Trainable percentage is high for LoRA",
       severity=Severity.WARN, tags=("lora", "peft"),
       summary="A large share of parameters is trainable.",
       why="Corroborates an unfrozen base or mis-targeted adapters; LoRA should "
           "leave only a small fraction trainable.",
       fix="Verify target_modules and that the base is frozen.",
       reference="https://arxiv.org/abs/2106.09685"),
    # -- distributed -------------------------------------------------------
    _m(id="DDP-001", title="Unused parameter hangs DDP",
       severity=Severity.ERROR, tags=("distributed", "ddp"),
       summary="A parameter gets no gradient while the model is wrapped in DDP.",
       why="DDP's gradient all-reduce expects every parameter to participate; an "
           "unused one deadlocks or errors unless find_unused_parameters=True.",
       fix="Remove the unused parameter, or set find_unused_parameters=True (slower).",
       reference="https://pytorch.org/docs/stable/notes/ddp.html"),
    _m(id="DDP-002", title="BatchNorm is not synchronized across ranks",
       severity=Severity.WARN, tags=("distributed", "ddp", "fsdp"),
       summary="Plain BatchNorm under DDP/FSDP computes per-GPU statistics.",
       why="Each rank normalizes with its local batch, changing behavior versus "
           "single-GPU and hurting small-per-GPU-batch training.",
       fix="nn.SyncBatchNorm.convert_sync_batchnorm(model) before wrapping.",
       reference="https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html"),
    _m(id="DDP-003", title="Process group initialized but model not wrapped",
       severity=Severity.WARN, tags=("distributed",),
       summary="torch.distributed is initialized yet the model is not DDP/FSDP.",
       why="Gradients won't be synchronized across ranks; each rank trains its own "
           "diverging copy.",
       fix="Wrap the model in DistributedDataParallel or FSDP.",
       reference="https://pytorch.org/docs/stable/notes/ddp.html"),
]}


def get(mode_id):
    """Look up a FailureMode by its catalog id (e.g. 'LORA-001')."""
    return CATALOG[mode_id]


def catalog():
    """All known failure modes, ordered by severity then id."""
    return sorted(CATALOG.values(), key=lambda m: (-int(m.severity), m.id))


def render_markdown():
    """Render the catalog as Markdown (used to generate CATALOG.md)."""
    section_order = ["lora", "distributed", "core", "optimizer", "gradients",
                     "numerics", "activations"]
    titles = {"lora": "LoRA / PEFT fine-tuning",
              "distributed": "Distributed training (DDP / FSDP)",
              "core": "Core wiring", "optimizer": "Optimizer",
              "gradients": "Gradient health", "numerics": "Numerical stability",
              "activations": "Activations"}
    badge = {Severity.ERROR: "FAIL", Severity.WARN: "WARN", Severity.INFO: "INFO"}
    seen = set()
    out = ["# Catalog of silent training-failure modes", "",
           "Every check in guardtower maps to one of these. Each entry says what the "
           "failure is, why it bites silently, how to fix it, and a reference. "
           f"({len(CATALOG)} modes and counting — contributions welcome.)", ""]
    for tag in section_order:
        members = [m for m in catalog() if tag in m.tags and m.id not in seen]
        for m in members:
            seen.add(m.id)
        if not members:
            continue
        out += [f"## {titles[tag]}", ""]
        for m in members:
            out += [f"### `{m.id}` — {m.title}  _({badge[m.severity]})_", "",
                    f"- **What:** {m.summary}",
                    f"- **Why it's silent:** {m.why}",
                    f"- **Fix:** {m.fix}",
                    f"- **Reference:** {m.reference}", ""]
    return "\n".join(out)


# public alias
catalog_markdown = render_markdown
