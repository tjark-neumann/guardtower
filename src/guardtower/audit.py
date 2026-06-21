"""The auditor: run one instrumented training step and report silent bugs."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import torch
from torch import nn

from .distributed import distributed_checks
from .peft import peft_checks
from .report import Report, Severity

# Default catalogue of checks. Each name maps to a private function below.
ALL_CHECKS = (
    "loss",
    "grad_connectivity",
    "optimizer_coverage",
    "frozen",
    "weight_tying",
    "nonfinite",
    "dead_relu",
    "grad_norms",
    "input_dependency",
    "device_dtype",
    "peft",
    "distributed",
)


def audit(
    model: nn.Module,
    step_fn: Callable[[], torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    inputs: torch.Tensor | Sequence[torch.Tensor] | None = None,
    checks: Iterable[str] | str = "all",
    dead_relu_threshold: float = 0.95,
    exploding_norm: float = 1e3,
    vanishing_norm: float = 1e-7,
) -> Report:
    """Run a single instrumented training step and report silent bugs.

    Parameters
    ----------
    model
        The module being trained.
    step_fn
        A zero-arg closure that performs the forward pass and returns a scalar
        loss, e.g. ``lambda: loss_fn(model(x), y)``. guardtower calls it once and
        runs ``.backward()`` for you.
    optimizer
        Optional. If given, guardtower checks that every trainable parameter is
        actually registered with it (and flags frozen params wastefully added).
    inputs
        Optional tensor or tuples of tensors used *inside* ``step_fn``. Float
        inputs are temporarily marked ``requires_grad`` so guardtower can verify
        the loss actually depends on them.
    checks
        ``"all"`` or an iterable of check names from :data:`ALL_CHECKS`.

    Returns
    -------
    Report
        Inspect it, print it, or call ``.raise_if_errors()`` to fail fast.
    """
    selected = ALL_CHECKS if checks == "all" else tuple(checks)
    rep = Report(meta={"checks": list(selected)})

    if not isinstance(model, nn.Module):
        rep.add("setup", Severity.ERROR, "model is not an nn.Module")
        return rep

    # Static checks that need no step ---------------------------------------
    if "frozen" in selected:
        _check_frozen(model, rep)
    if "weight_tying" in selected:
        _check_weight_tying(model, rep)
    if "device_dtype" in selected:
        _check_device_dtype(model, rep)

    # Mode sanity: backprop in eval() is usually a mistake.
    if not model.training:
        rep.add(
            "train_mode",
            Severity.WARN,
            "model is in eval() mode but you're auditing a training step "
            "(dropout/batchnorm will behave as at inference)",
        )

    # Mark inputs so we can test loss<-input dependency.
    tracked_inputs: list[torch.Tensor] = []
    if inputs is not None and "input_dependency" in selected:
        seq = [inputs] if isinstance(inputs, torch.Tensor) else list(inputs)
        for t in seq:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                t.requires_grad_(True)
                tracked_inputs.append(t)

    # Instrument forward/backward for activation + grad finiteness/stats.
    instr = _Instrumentation(model, want_dead_relu="dead_relu" in selected)
    instr.attach()

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    else:
        model.zero_grad(set_to_none=True)

    try:
        loss = step_fn()
    except Exception as e:  # pragma: no cover - surfaced to user
        instr.detach()
        rep.add("loss", Severity.ERROR, f"step_fn raised {type(e).__name__}: {e}")
        return rep

    # Loss sanity -----------------------------------------------------------
    loss_ok = _check_loss(loss, rep) if "loss" in selected else _is_scalar_loss(loss)
    if "nonfinite" in selected:
        instr.report_forward_nonfinite(rep)

    if loss_ok:
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message="Full backward hook is firing")
                loss.backward()
        except Exception as e:  # pragma: no cover
            instr.detach()
            rep.add("backward", Severity.ERROR, f"backward() raised {type(e).__name__}: {e}")
            return rep

        if "grad_connectivity" in selected:
            from .peft import is_peft_model
            _check_grad_connectivity(model, rep, lora_aware=is_peft_model(model))
        if "grad_norms" in selected:
            _check_grad_norms(model, rep, exploding_norm, vanishing_norm)
        if "nonfinite" in selected:
            instr.report_backward_nonfinite(rep)
            _check_grad_finite(model, rep)
        if "input_dependency" in selected and tracked_inputs:
            _check_input_dependency(tracked_inputs, rep)

    if "dead_relu" in selected:
        instr.report_dead_relu(rep, dead_relu_threshold)
    if "optimizer_coverage" in selected and optimizer is not None:
        _check_optimizer_coverage(model, optimizer, rep)

    # Stack-specific checks: auto-run when the relevant setup is detected.
    if "peft" in selected:
        peft_checks(model, rep, optimizer=optimizer)
    if "distributed" in selected:
        unused = any(
            p.requires_grad and p.grad is None for _, p in model.named_parameters()
        )
        distributed_checks(model, rep, has_unused_params=unused)

    instr.detach()
    # clean up the requires_grad we forced on user inputs
    for t in tracked_inputs:
        t.requires_grad_(False)
        t.grad = None
    return rep


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------
def _is_scalar_loss(loss) -> bool:
    return isinstance(loss, torch.Tensor) and loss.dim() == 0 and loss.requires_grad


def _check_loss(loss, rep: Report) -> bool:
    if not isinstance(loss, torch.Tensor):
        rep.add("loss", Severity.ERROR, f"step_fn returned {type(loss).__name__}, not a tensor")
        return False
    if loss.dim() != 0:
        rep.add(
            "loss", Severity.ERROR,
            f"loss is not a scalar (shape {tuple(loss.shape)}); reduce it (e.g. .mean())",
        )
        return False
    if not loss.requires_grad:
        rep.add(
            "loss", Severity.ERROR, catalog_id="LOSS-001",
            message="loss does not require grad — it is detached from the parameters; "
            "nothing will train",
        )
        return False
    if not torch.isfinite(loss).item():
        rep.add("loss", Severity.ERROR, f"loss is {loss.item()} (non-finite) on the very first step")
        return False
    return True


def _check_grad_connectivity(model: nn.Module, rep: Report, lora_aware: bool = False) -> None:
    from .peft import ADAPTER_MARKERS

    no_grad, zero_grad = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            no_grad.append(name)
        elif float(p.grad.abs().sum()) == 0.0:
            zero_grad.append(name)
    if no_grad:
        rep.add(
            "grad_connectivity", Severity.ERROR, catalog_id="GRAD-001",
            message=f"{len(no_grad)} trainable parameter(s) received NO gradient — "
            f"they are disconnected from the loss and will never update "
            f"(e.g. {no_grad[0]})",
            params=no_grad,
        )
    if zero_grad:
        # LoRA's B matrix is zero-initialized, so A has no gradient on step 1.
        # That's expected, not a bug — don't cry wolf about it.
        if lora_aware:
            adapter_zero = [n for n in zero_grad if any(m in n for m in ADAPTER_MARKERS)]
            other_zero = [n for n in zero_grad if n not in adapter_zero]
        else:
            adapter_zero, other_zero = [], zero_grad
        if adapter_zero:
            rep.add(
                "grad_connectivity", Severity.INFO, catalog_id="LORA-004",
                message=f"{len(adapter_zero)} adapter tensor(s) had a zero gradient this "
                f"step — expected at initialization (LoRA B is zero-init, so A "
                f"sees no gradient on the first step)",
                params=adapter_zero,
            )
        if other_zero:
            rep.add(
                "grad_connectivity", Severity.WARN,
                f"{len(other_zero)} trainable parameter(s) had an all-zero gradient "
                f"this step (may be legitimate, often a masking/detach bug) "
                f"(e.g. {other_zero[0]})",
                params=other_zero,
            )
    if not no_grad and not zero_grad:
        rep.add("grad_connectivity", Severity.INFO, "all trainable parameters receive gradient")


def _check_optimizer_coverage(model, optimizer, rep: Report) -> None:
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    missing, frozen_in_opt = [], []
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in opt_ids:
            missing.append(name)
        if (not p.requires_grad) and id(p) in opt_ids:
            frozen_in_opt.append(name)
    if missing:
        rep.add(
            "optimizer_coverage", Severity.ERROR, catalog_id="OPT-001",
            message=f"{len(missing)} trainable parameter(s) are NOT in the optimizer and "
            f"won't be updated (e.g. {missing[0]}). Did you build the optimizer "
            f"before adding/replacing a module?",
            params=missing,
        )
    if frozen_in_opt:
        rep.add(
            "optimizer_coverage", Severity.WARN,
            f"{len(frozen_in_opt)} frozen parameter(s) are registered with the optimizer "
            f"(harmless but wasteful, and weight decay may still apply) (e.g. {frozen_in_opt[0]})",
            params=frozen_in_opt,
        )
    if not missing and not frozen_in_opt:
        rep.add("optimizer_coverage", Severity.INFO, "optimizer covers exactly the trainable params")


def _check_frozen(model: nn.Module, rep: Report) -> None:
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    total = sum(1 for _ in model.named_parameters())
    if frozen:
        n_frozen_elems = sum(
            p.numel() for n, p in model.named_parameters() if not p.requires_grad
        )
        rep.add(
            "frozen", Severity.INFO,
            f"{len(frozen)}/{total} parameter tensors are frozen "
            f"({n_frozen_elems:,} elements). Confirm this is intentional (e.g. {frozen[0]})",
            params=frozen,
        )


def _check_weight_tying(model: nn.Module, rep: Report) -> None:
    by_ptr: dict[int, list[str]] = {}
    # remove_duplicate=False so tied (shared-storage) params show up more than once
    for name, p in model.named_parameters(remove_duplicate=False):
        if p.numel() == 0:
            continue
        by_ptr.setdefault(p.data_ptr(), []).append(name)
    tied = [names for names in by_ptr.values() if len(names) > 1]
    if tied:
        groups = "; ".join(" = ".join(g) for g in tied)
        rep.add(
            "weight_tying", Severity.INFO,
            f"{len(tied)} group(s) of tied (shared-storage) parameters: {groups}",
            groups=tied,
        )


def _check_grad_norms(model, rep: Report, exploding: float, vanishing: float) -> None:
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms[name] = float(p.grad.norm())
    if not norms:
        return
    total = math.sqrt(sum(v * v for v in norms.values()))
    big = {n: v for n, v in norms.items() if v > exploding}
    if big:
        worst = max(big, key=big.get)
        rep.add(
            "grad_norms", Severity.WARN,
            f"large gradient norm(s): {worst} = {big[worst]:.2e} "
            f"(total {total:.2e}). Consider clipping or a lower LR",
            per_param=norms,
        )
    elif total < vanishing:
        rep.add(
            "grad_norms", Severity.WARN,
            f"global gradient norm is tiny ({total:.2e}); signal may be vanishing",
            per_param=norms,
        )
    else:
        rep.add("grad_norms", Severity.INFO, f"global gradient norm = {total:.3e}", per_param=norms)


def _check_grad_finite(model, rep: Report) -> None:
    bad = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    if bad:
        rep.add(
            "nonfinite", Severity.ERROR,
            f"{len(bad)} parameter(s) have NaN/Inf in their gradient (e.g. {bad[0]})",
            params=bad,
        )


def _check_input_dependency(tracked_inputs, rep: Report) -> None:
    dead = []
    for i, t in enumerate(tracked_inputs):
        if t.grad is None or float(t.grad.abs().sum()) == 0.0:
            dead.append(i)
    if dead:
        rep.add(
            "input_dependency", Severity.ERROR, catalog_id="INPUT-001",
            message=f"the loss does not depend on input tensor(s) at position {dead} — "
            f"the model is ignoring its input (wiring bug?)",
            positions=dead,
        )
    else:
        rep.add("input_dependency", Severity.INFO, "loss depends on all provided inputs")


def _check_device_dtype(model: nn.Module, rep: Report) -> None:
    devices, dtypes = set(), set()
    for p in model.parameters():
        devices.add(str(p.device))
        if p.is_floating_point():
            dtypes.add(str(p.dtype))
    if len(devices) > 1:
        rep.add(
            "device_dtype", Severity.WARN,
            f"parameters span multiple devices {sorted(devices)} — intentional sharding?",
        )
    if len(dtypes) > 1:
        rep.add(
            "device_dtype", Severity.INFO,
            f"mixed floating dtypes across parameters {sorted(dtypes)}",
        )


# --------------------------------------------------------------------------
# forward/backward instrumentation
# --------------------------------------------------------------------------
class _Instrumentation:
    """Registers hooks to capture activation finiteness and ReLU sparsity."""

    def __init__(self, model: nn.Module, want_dead_relu: bool):
        self.model = model
        self.want_dead_relu = want_dead_relu
        self.handles: list = []
        self.fwd_nonfinite: list[str] = []
        self.bwd_nonfinite: list[str] = []
        self.relu_zero_frac: dict[str, float] = {}

    def attach(self) -> None:
        for name, mod in self.model.named_modules():
            # include the root module: non-finite values are often produced in
            # the top-level forward, after the last submodule returns.
            self.handles.append(mod.register_forward_hook(self._fwd_hook(name)))
            self.handles.append(mod.register_full_backward_hook(self._bwd_hook(name)))

    def _fwd_hook(self, name):
        def hook(mod, inp, out):
            for t in _iter_tensors(out):
                if t.is_floating_point() and not torch.isfinite(t).all():
                    self.fwd_nonfinite.append(name or type(mod).__name__)
                    break
            if self.want_dead_relu and isinstance(mod, (nn.ReLU, nn.ReLU6)):
                for t in _iter_tensors(out):
                    if t.numel():
                        self.relu_zero_frac[name or "relu"] = float((t == 0).float().mean())
                    break
        return hook

    def _bwd_hook(self, name):
        def hook(mod, grad_in, grad_out):
            for t in _iter_tensors(grad_out):
                if t is not None and t.is_floating_point() and not torch.isfinite(t).all():
                    self.bwd_nonfinite.append(name or type(mod).__name__)
                    break
        return hook

    def report_forward_nonfinite(self, rep: Report) -> None:
        if self.fwd_nonfinite:
            rep.add(
                "nonfinite", Severity.ERROR,
                f"NaN/Inf appeared in the forward activations of module(s): "
                f"{self.fwd_nonfinite[:3]}",
                modules=self.fwd_nonfinite,
            )

    def report_backward_nonfinite(self, rep: Report) -> None:
        if self.bwd_nonfinite:
            rep.add(
                "nonfinite", Severity.ERROR,
                f"NaN/Inf appeared in the backward pass of module(s): {self.bwd_nonfinite[:3]}",
                modules=self.bwd_nonfinite,
            )

    def report_dead_relu(self, rep: Report, threshold: float) -> None:
        dead = {n: f for n, f in self.relu_zero_frac.items() if f >= threshold}
        if dead:
            worst = max(dead, key=dead.get)
            rep.add(
                "dead_relu", Severity.WARN,
                f"{len(dead)} ReLU module(s) output mostly zeros this batch "
                f"(e.g. {worst}: {dead[worst]*100:.0f}% dead) — possible dead units / bad init",
                fractions=dead,
            )

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _iter_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _iter_tensors(x)
