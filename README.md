# guardtower

**Catch the silent fine-tuning bugs that waste GPU-hours — before you launch the run.**

Most training bugs don't crash. The model runs, the loss goes down, and hours
later your results are quietly wrong because the part you *meant* to train never
did. `guardtower` runs one instrumented step on a single batch and tells you, in
plain language, exactly what's wrong — and points each finding at a documented
[catalog of failure modes](CATALOG.md).

Built for how people fine-tune today: **LoRA / PEFT / QLoRA** and **multi-GPU
(DDP / FSDP)**. It auto-detects your setup and runs the right checks.

## Drop it into the training loop you already run

One line in a Hugging Face `Trainer` and guardtower audits the model on the first
real batch — and can abort the run before the expensive part starts:

```python
from guardtower.integrations.huggingface import GuardtowerCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    callbacks=[GuardtowerCallback(raise_on_error=True)],
)
trainer.train()
```

If the adapter isn't wired up, the run stops immediately instead of training
nothing for six hours:

```
[guardtower] pre-flight audit on first batch:
────────────────────────────────────────────────────────────────
guardtower: FAIL  (1 fail · 0 warn · 2 info)
────────────────────────────────────────────────────────────────
  ✗ FAIL lora_optimizer  [LORA-003]
      4 trainable adapter tensor(s) are NOT in the optimizer and will
      never update. Build the optimizer from the PEFT model AFTER
      get_peft_model(...)
────────────────────────────────────────────────────────────────
```

## Or call it directly

```python
import guardtower

# Quick, training-free LoRA check:
print(guardtower.lora_summary(model, optimizer=optimizer))

# Full pre-flight (runs one step):
report = guardtower.audit(
    model,
    lambda: loss_fn(model(x), y),   # closure returning a scalar loss
    optimizer=optimizer,
    inputs=x,
)
report.raise_if_errors()            # fail fast in CI / before launching
```

## What it catches

The full list lives in **[CATALOG.md](CATALOG.md)** — every check maps to an
entry there with *what it is, why it's silent, how to fix it, and a reference.*
Highlights:

- **LoRA/PEFT:** base model accidentally left unfrozen, adapter frozen so nothing
  trains, adapter missing from the optimizer, trainable-% sanity. Knows that
  LoRA's zero-initialized `B` matrix makes `A` show a zero gradient on step 1 —
  so it **doesn't** false-alarm on correct setups.
- **Distributed (DDP/FSDP):** an unused parameter that will *hang* DDP, plain
  BatchNorm that should be SyncBatchNorm, a process group with an unwrapped model.
- **Core wiring:** parameter that gets no gradient, trainable param missing from
  the optimizer, loss detached from the graph, loss that doesn't depend on the
  input.
- **Numerics:** NaN/Inf in forward **and** backward, pinpointed to the module;
  dead ReLUs; exploding / vanishing gradient norms.

## Live NaN monitoring

Leave a cheap guardtower on for the first steps of a real run so a blow-up is
pinpointed to the exact module the instant it happens:

```python
with guardtower.monitor(model, on_nonfinite="raise"):
    for batch in loader:
        train_step(batch)
```

## Install

```bash
pip install guardtower            # core (PyTorch only)
pip install guardtower[hf]        # + Hugging Face Trainer integration
```

PEFT/transformers are **not** required for the core checks — LoRA detection works
by inspecting parameters, so it's safe on any model.

## The catalog is the point

The value isn't the code that loops over parameters — that's easy to regenerate.
It's the curated, referenced list of *every silent way fine-tuning fails*, kept
current. Hit a new one? Add an entry to `guardtower/catalog.py` and a check that
emits its id. `guardtower.catalog()` returns the list; `guardtower.catalog_markdown()`
regenerates [CATALOG.md](CATALOG.md).

## The `Report` object

| attribute / method          | meaning                                         |
|-----------------------------|-------------------------------------------------|
| `report.ok`                 | `True` if there are no blocking (FAIL) findings |
| `report.errors / warnings / infos` | findings by severity                     |
| `report.by_check(name)`     | findings from a specific check                  |
| `report.raise_if_errors()`  | raise `GuardtowerError` on any FAIL (chains)      |
| `report.to_dict()`          | JSON-friendly dump for logging                  |

Each finding carries a `catalog_id` linking to its catalog entry.

## Test

```bash
python tests/test_checks.py      # or: pytest -q   (22 tests, no GPU needed)
```

## License

Apache-2.0. See [LICENSE](LICENSE).
