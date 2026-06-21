"""Checks for distributed training setups (DDP / FSDP).

Single-process-friendly: detection is by wrapper class and process-group state,
so these checks are useful even on a single rank during development.
"""

from __future__ import annotations

from torch import nn

from .report import Report, Severity

DDP_NAMES = ("DistributedDataParallel", "DataParallel")
FSDP_NAMES = ("FullyShardedDataParallel", "FSDPModule")

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _wrapper_kind(model: nn.Module) -> str | None:
    for cls in type(model).__mro__:
        if cls.__name__ in DDP_NAMES:
            return "ddp"
        if cls.__name__ in FSDP_NAMES:
            return "fsdp"
    return None


def _dist_initialized() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def distributed_checks(
    model: nn.Module,
    rep: Report,
    *,
    has_unused_params: bool = False,
) -> bool:
    """Append distributed-training findings. Returns True if a distributed
    setup was detected (wrapped model or an initialized process group)."""
    kind = _wrapper_kind(model)
    initialized = _dist_initialized()
    if kind is None and not initialized:
        return False

    if initialized and kind is None:
        rep.add(
            "dist_wrap", Severity.WARN, catalog_id="DDP-003", message=
            "a process group is initialized but the model is NOT wrapped in "
            "DDP or FSDP — gradients will not be synchronized across ranks",
        )

    if kind == "ddp":
        rep.add("dist_setup", Severity.INFO, "DistributedDataParallel detected")
        # Plain BatchNorm under DDP computes stats per-GPU, not globally.
        bns = [
            n for n, m in model.named_modules()
            if isinstance(m, _BN) and not isinstance(m, nn.SyncBatchNorm)
        ]
        if bns:
            rep.add(
                "dist_batchnorm", Severity.WARN, catalog_id="DDP-002", message=
                f"{len(bns)} plain BatchNorm layer(s) under DDP — convert with "
                f"nn.SyncBatchNorm.convert_sync_batchnorm(model) for correct "
                f"multi-GPU statistics (e.g. {bns[0]})",
                modules=bns,
            )
        # Unused params hang DDP unless find_unused_parameters=True.
        if has_unused_params:
            rep.add(
                "dist_unused", Severity.ERROR, catalog_id="DDP-001", message=
                "some parameters received no gradient AND the model is under "
                "DDP — this will hang or crash collective sync unless the model "
                "is created with find_unused_parameters=True (better: remove the "
                "unused parameters)",
            )

    if kind == "fsdp":
        rep.add("dist_setup", Severity.INFO, "FSDP detected")
        bns = [n for n, m in model.named_modules()
               if isinstance(m, _BN) and not isinstance(m, nn.SyncBatchNorm)]
        if bns:
            rep.add(
                "dist_batchnorm", Severity.WARN, catalog_id="DDP-002", message=
                f"{len(bns)} plain BatchNorm layer(s) under FSDP — running stats "
                f"are not synchronized across shards; prefer SyncBatchNorm or a "
                f"norm without running stats (e.g. {bns[0]})",
                modules=bns,
            )

    return True
