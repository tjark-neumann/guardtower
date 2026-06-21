"""A real, silent LoRA bug guardtower catches in one step.

A common mistake: you build the optimizer from the model and then wrap it for
LoRA. The optimizer captured the original weights (now frozen), and the adapter
tensors it should be training didn't exist yet, so it never sees them. Nothing
crashes, the loss even drifts a little, and six GPU-hours later the adapter is
still at its initialization.

Same shape as building the optimizer before ``get_peft_model(...)``.

    python examples/lora_optimizer_bug.py
"""

import torch
import torch.nn as nn

import guardtower


class LoRALinear(nn.Module):
    """A frozen linear with a trainable LoRA adapter, named the way PEFT names them."""

    def __init__(self, base: nn.Linear, r=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False  # base is frozen under LoRA
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)  # real LoRA: B is zero-initialised

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


class TinyModel(nn.Module):
    def __init__(self, d=128, n_classes=4):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.act = nn.ReLU()
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        x = self.act(self.q_proj(x))
        x = self.v_proj(x)
        return self.head(x.mean(dim=1))


def main():
    torch.manual_seed(0)
    model = TinyModel()

    # --- the bug -----------------------------------------------------------
    # Optimizer built here, from the model, before the adapters exist.
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # The LoRA adapters are added afterwards: new tensors the optimizer never
    # sees, while the base weights it did capture are now frozen.
    model.q_proj = LoRALinear(model.q_proj)
    model.v_proj = LoRALinear(model.v_proj)
    for p in model.head.parameters():
        p.requires_grad = False
    # -----------------------------------------------------------------------

    x = torch.randn(8, 16, 128)            # (batch, seq, dim)
    y = torch.randint(0, 4, (8,))

    report = guardtower.audit(
        model,
        lambda: nn.functional.cross_entropy(model(x), y),
        optimizer=optimizer,
        inputs=x,
    )
    print(report)

    # In a real script this aborts the run before any GPU time is spent.
    try:
        report.raise_if_errors()
    except guardtower.GuardtowerError:
        print("\n→ guardtower aborted the run: fix the optimizer, then relaunch.")


if __name__ == "__main__":
    main()
