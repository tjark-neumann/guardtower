"""Each test plants a real, silent bug and asserts guardtower catches it."""

import torch
from torch import nn

import guardtower


def _xy(n=8, d=4, c=3):
    return torch.randn(n, d), torch.randint(0, c, (n,))


# --------------------------------------------------------------------------
def test_clean_model_passes():
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = _xy()
    rep = guardtower.audit(model, lambda: nn.functional.cross_entropy(model(x), y),
                         optimizer=opt, inputs=x)
    assert rep.ok, rep
    assert rep.errors == []


def test_disconnected_parameter_is_caught():
    # A second head that never participates in the loss -> no gradient.
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(4, 3)
            self.orphan = nn.Linear(4, 3)  # never called

        def forward(self, x):
            return self.used(x)

    model = Net()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = _xy()
    rep = guardtower.audit(model, lambda: nn.functional.cross_entropy(model(x), y), optimizer=opt)
    assert not rep.ok
    msgs = [f.message for f in rep.by_check("grad_connectivity")]
    assert any("NO gradient" in m for m in msgs)
    params = rep.by_check("grad_connectivity")[0].details["params"]
    assert any("orphan" in p for p in params)


def test_param_missing_from_optimizer_is_caught():
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    # Optimizer only sees the first layer.
    opt = torch.optim.SGD(model[0].parameters(), lr=0.1)
    x, y = _xy()
    rep = guardtower.audit(model, lambda: nn.functional.cross_entropy(model(x), y), optimizer=opt)
    assert not rep.ok
    assert any("NOT in the optimizer" in f.message for f in rep.by_check("optimizer_coverage"))


def test_detached_loss_is_caught():
    model = nn.Linear(4, 3)
    x, y = _xy()
    rep = guardtower.audit(model, lambda: nn.functional.cross_entropy(model(x), y).detach())
    assert not rep.ok
    assert any("does not require grad" in f.message for f in rep.by_check("loss"))


def test_nonscalar_loss_is_caught():
    model = nn.Linear(4, 3)
    x, y = _xy()
    rep = guardtower.audit(
        model,
        lambda: nn.functional.cross_entropy(model(x), y, reduction="none"),
    )
    assert not rep.ok
    assert any("not a scalar" in f.message for f in rep.by_check("loss"))


def test_input_ignored_is_caught():
    # Loss built from a constant; model output (and input) unused.
    model = nn.Linear(4, 3)
    x, y = _xy()
    const = model.bias.sum()  # depends on params but NOT on x

    rep = guardtower.audit(model, lambda: (const - const + model.bias.pow(2).sum()),
                         inputs=x)
    assert any(
        "does not depend on input" in f.message for f in rep.by_check("input_dependency")
    )


def test_nan_in_forward_is_caught():
    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 3)

        def forward(self, x):
            out = self.lin(x)
            out = out / 0.0  # -> inf/nan
            return out

    model = Bad()
    x, y = _xy()
    rep = guardtower.audit(model, lambda: model(x).sum())
    assert not rep.ok
    assert any("NaN/Inf" in f.message for f in rep.by_check("nonfinite"))


def test_frozen_layer_reported():
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    for p in model[0].parameters():
        p.requires_grad_(False)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    x, y = _xy()
    rep = guardtower.audit(model, lambda: nn.functional.cross_entropy(model(x), y), optimizer=opt)
    assert any("frozen" in f.message for f in rep.by_check("frozen"))
    # frozen params correctly excluded -> still passes
    assert rep.ok, rep


def test_weight_tying_detected():
    emb = nn.Embedding(10, 8)
    head = nn.Linear(8, 10, bias=False)
    head.weight = emb.weight  # classic tied LM head
    model = nn.ModuleDict({"emb": emb, "head": head})
    rep = guardtower.audit(model, lambda: model["emb"](torch.tensor([0, 1])).sum())
    tie = rep.by_check("weight_tying")
    assert tie and len(tie[0].details["groups"]) == 1


def test_exploding_grad_norm_warns():
    model = nn.Linear(4, 3)
    x, y = _xy()
    # huge target -> huge gradient
    rep = guardtower.audit(model, lambda: (model(x) - 1e6).pow(2).mean())
    assert any("gradient norm" in f.message for f in rep.by_check("grad_norms"))


def test_raise_if_errors():
    model = nn.Linear(4, 3)
    x, y = _xy()
    rep = guardtower.audit(model, lambda: nn.functional.cross_entropy(model(x), y).detach())
    try:
        rep.raise_if_errors()
        assert False, "should have raised"
    except guardtower.GuardtowerError as e:
        assert "blocking issue" in str(e)


def test_monitor_raises_on_nan():
    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 3)

        def forward(self, x):
            return self.lin(x) / 0.0

    model = Bad()
    x, _ = _xy()
    raised = False
    try:
        with guardtower.monitor(model, on_nonfinite="raise"):
            model(x)
    except guardtower.NonFiniteActivation:
        raised = True
    assert raised


# -- LoRA / PEFT (mocked so the test suite needs no peft install) -----------
class _FakeLoRALinear(nn.Module):
    """Mimics PEFT's parameter naming: base_layer + lora_A/lora_B."""

    def __init__(self, d=8, r=2):
        super().__init__()
        self.base_layer = nn.Linear(d, d)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d, bias=False)})
        for p in self.base_layer.parameters():
            p.requires_grad_(False)            # PEFT freezes the base
        nn.init.zeros_(self.lora_B["default"].weight)  # B is zero-initialized

    def forward(self, x):
        return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))


class _FakeLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _FakeLoRALinear(8, 2)
        self.head = nn.Linear(8, 3)
        for p in self.head.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.head(self.layer(x))


def test_lora_is_detected():
    assert guardtower.is_peft_model(_FakeLoRAModel())
    assert not guardtower.is_peft_model(nn.Linear(4, 3))


def test_lora_summary_clean():
    model = _FakeLoRAModel()
    rep = guardtower.lora_summary(model)
    assert any("trainable adapter" in f.message for f in rep.by_check("lora_adapter"))
    # base correctly frozen -> no base-frozen warning
    assert rep.by_check("lora_base_frozen") == []


def test_lora_base_unfrozen_warns():
    model = _FakeLoRAModel()
    for p in model.parameters():
        p.requires_grad_(True)               # oops: unfroze the base
    rep = guardtower.lora_summary(model)
    assert any("BASE model is not fully frozen" in f.message
               for f in rep.by_check("lora_base_frozen"))


def test_lora_adapter_all_frozen_errors():
    model = _FakeLoRAModel()
    for p in model.parameters():
        p.requires_grad_(False)              # froze everything incl. adapter
    rep = guardtower.lora_summary(model)
    assert any("ALL frozen" in f.message for f in rep.by_check("lora_adapter"))
    assert not rep.ok


def test_lora_adapter_missing_from_optimizer():
    model = _FakeLoRAModel()
    opt = torch.optim.SGD(nn.Linear(8, 8).parameters(), lr=0.1)  # wrong params
    x = torch.randn(8, 8)
    rep = guardtower.audit(model, lambda: model(x).pow(2).mean(), optimizer=opt)
    assert any("NOT in the optimizer" in f.message for f in rep.by_check("lora_optimizer"))
    assert not rep.ok


def test_lora_correct_setup_has_no_false_zero_grad_warning():
    model = _FakeLoRAModel()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    x = torch.randn(8, 8)
    rep = guardtower.audit(model, lambda: model(x).pow(2).mean(), optimizer=opt)
    # The zero grad on lora_A is expected at init and must NOT be a warning.
    assert rep.ok, rep
    assert rep.warnings == [], rep


# -- distributed (mock the wrapper class name; no process group needed) -----
class DistributedDataParallel(nn.Module):  # name matches the real DDP class
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def test_ddp_unused_param_and_batchnorm():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)
            self.bn = nn.BatchNorm1d(16)
            self.head = nn.Linear(16, 4)
            self.unused = nn.Linear(16, 16)   # never used -> hangs DDP

        def forward(self, x):
            return self.head(self.bn(self.fc(x)))

    model = DistributedDataParallel(Net())
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    rep = guardtower.audit(model, lambda: nn.functional.cross_entropy(model(x), y), optimizer=opt)
    assert any("DistributedDataParallel detected" in f.message for f in rep.by_check("dist_setup"))
    assert any("hang or crash" in f.message for f in rep.by_check("dist_unused"))
    assert any("SyncBatchNorm" in f.message for f in rep.by_check("dist_batchnorm"))


# -- catalog integrity ------------------------------------------------------
def test_catalog_is_well_formed():

    assert len(guardtower.CATALOG) >= 15
    for mode in guardtower.catalog():
        assert mode.id and mode.title and mode.summary
        assert mode.why and mode.fix and mode.reference
        assert mode.reference.startswith("http")
        assert mode.tags


def test_emitted_catalog_ids_all_resolve():
    # Drive several failing scenarios and confirm every catalog_id a finding
    # carries actually exists in the catalog (no dangling references).

    x = torch.randn(8, 8)
    scenarios = []

    # disconnected param + DDP + batchnorm
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8); self.head = nn.Linear(8, 4); self.unused = nn.Linear(8, 8)
        def forward(self, x):
            return self.head(self.fc(x))
    m1 = DistributedDataParallel(Net())
    scenarios.append(guardtower.audit(
        m1, lambda: nn.functional.cross_entropy(m1(x), torch.randint(0, 4, (8,))),
        optimizer=torch.optim.SGD(m1.parameters(), lr=0.1)))

    # detached loss
    m2 = nn.Linear(8, 4)
    scenarios.append(guardtower.audit(m2, lambda: m2(x).sum().detach()))

    # LoRA optimizer bug
    m3 = _FakeLoRAModel()
    scenarios.append(guardtower.audit(
        m3, lambda: m3(x).pow(2).mean(),
        optimizer=torch.optim.SGD(nn.Linear(8, 8).parameters(), lr=0.1)))

    seen_ids = set()
    for rep in scenarios:
        for f in rep:
            if f.catalog_id is not None:
                seen_ids.add(f.catalog_id)
                assert f.catalog_id in guardtower.CATALOG, f"dangling id {f.catalog_id}"
    # we should have exercised a decent spread
    assert len(seen_ids) >= 4, seen_ids


# -- Hugging Face callback (skips if transformers absent) -------------------
def test_hf_callback_runs_and_raises():
    try:
        from guardtower.integrations.huggingface import GuardtowerCallback
    except ImportError:
        print("    (skipped: transformers not installed)")
        return
    from types import SimpleNamespace

    class FakeOut:
        def __init__(self, loss):
            self.loss = loss

    class FakeHFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 2)
        def forward(self, input_ids=None, labels=None):
            return FakeOut(nn.functional.cross_entropy(self.lin(input_ids), labels))

    model = FakeHFModel()
    dataloader = [{"input_ids": torch.randn(4, 8), "labels": torch.randint(0, 2, (4,))}]

    # clean: optimizer covers the model
    good_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cb = GuardtowerCallback(raise_on_error=True, verbose=False)
    cb.on_train_begin(None, None, SimpleNamespace(),
                      model=model, optimizer=good_opt, train_dataloader=dataloader)
    assert cb.report is not None and cb.report.ok

    # bug: optimizer built from the wrong params -> should raise
    bad_opt = torch.optim.SGD(nn.Linear(8, 2).parameters(), lr=0.1)
    cb2 = GuardtowerCallback(raise_on_error=True, verbose=False)
    raised = False
    try:
        cb2.on_train_begin(None, None, SimpleNamespace(),
                           model=model, optimizer=bad_opt, train_dataloader=dataloader)
    except guardtower.GuardtowerError:
        raised = True
    assert raised and not cb2.report.ok


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception as e:
            failures += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
