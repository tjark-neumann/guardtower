"""Run me:  python examples/quickstart.py"""

import torch
import torch.nn as nn

import guardtower


def main():
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(16, 32)
    y = torch.randint(0, 10, (16,))

    # One call, on a single batch, BEFORE you launch the real run.
    report = guardtower.audit(
        model,
        lambda: nn.functional.cross_entropy(model(x), y),
        optimizer=optimizer,
        inputs=x,
    )
    print(report)

    # In CI or at the top of train.py, make it fail fast:
    report.raise_if_errors()
    print("pre-flight clean — safe to launch \U0001F680")

    # Optionally keep a cheap NaN guardtower on for the first steps of real training:
    with guardtower.monitor(model, on_nonfinite="raise"):
        for _ in range(3):
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
