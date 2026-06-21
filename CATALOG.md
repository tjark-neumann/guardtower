# Catalog of silent training-failure modes

Every check in guardtower maps to one of these. Each entry says what the failure is, why it bites silently, how to fix it, and a reference. (21 modes and counting. Contributions welcome.)

## LoRA / PEFT fine-tuning

### `LORA-002` — LoRA adapter is frozen  _(FAIL)_

- **What:** Adapter params are present but all have requires_grad=False.
- **Why it's silent:** Everything was frozen (or the adapter disabled) — the run trains nothing.
- **Fix:** Enable the adapter; ensure its parameters require grad.
- **Reference:** https://huggingface.co/docs/peft/main/en/index

### `LORA-003` — LoRA adapter is missing from the optimizer  _(FAIL)_

- **What:** Trainable adapter params are not in any optimizer group.
- **Why it's silent:** Optimizer built from the wrong params (or before get_peft_model) — the adapter computes gradients that are never applied.
- **Fix:** Build the optimizer from the PEFT model after get_peft_model(...).
- **Reference:** https://huggingface.co/docs/peft/main/en/index

### `LORA-001` — Base model is not frozen under LoRA  _(WARN)_

- **What:** Base (non-adapter) parameters still require grad.
- **Why it's silent:** You think you're doing cheap LoRA but are full-fine-tuning: memory blows up and the parameter efficiency is lost.
- **Fix:** Freeze the base; get_peft_model() does this — don't re-enable grad after.
- **Reference:** https://huggingface.co/docs/peft/main/en/developer_guides/troubleshooting

### `LORA-005` — Trainable percentage is high for LoRA  _(WARN)_

- **What:** A large share of parameters is trainable.
- **Why it's silent:** Corroborates an unfrozen base or mis-targeted adapters; LoRA should leave only a small fraction trainable.
- **Fix:** Verify target_modules and that the base is frozen.
- **Reference:** https://arxiv.org/abs/2106.09685

### `LORA-004` — Adapter zero gradient at initialization (expected)  _(INFO)_

- **What:** LoRA's A matrix has no gradient on the first step.
- **Why it's silent:** LoRA initializes B to zero, so A legitimately sees no gradient until B moves. guardtower reports this as expected, not a bug.
- **Fix:** None — this is correct LoRA behavior.
- **Reference:** https://arxiv.org/abs/2106.09685

## Distributed training (DDP / FSDP)

### `DDP-001` — Unused parameter hangs DDP  _(FAIL)_

- **What:** A parameter gets no gradient while the model is wrapped in DDP.
- **Why it's silent:** DDP's gradient all-reduce expects every parameter to participate; an unused one deadlocks or errors unless find_unused_parameters=True.
- **Fix:** Remove the unused parameter, or set find_unused_parameters=True (slower).
- **Reference:** https://pytorch.org/docs/stable/notes/ddp.html

### `DDP-002` — BatchNorm is not synchronized across ranks  _(WARN)_

- **What:** Plain BatchNorm under DDP/FSDP computes per-GPU statistics.
- **Why it's silent:** Each rank normalizes with its local batch, changing behavior versus single-GPU and hurting small-per-GPU-batch training.
- **Fix:** nn.SyncBatchNorm.convert_sync_batchnorm(model) before wrapping.
- **Reference:** https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html

### `DDP-003` — Process group initialized but model not wrapped  _(WARN)_

- **What:** torch.distributed is initialized yet the model is not DDP/FSDP.
- **Why it's silent:** Gradients won't be synchronized across ranks; each rank trains its own diverging copy.
- **Fix:** Wrap the model in DistributedDataParallel or FSDP.
- **Reference:** https://pytorch.org/docs/stable/notes/ddp.html

## Core wiring

### `GRAD-001` — Parameter receives no gradient  _(FAIL)_

- **What:** A trainable parameter is disconnected from the loss (grad is None).
- **Why it's silent:** A sub-module is built but never called, or its output is dropped — it occupies optimizer state yet never learns. Invisible in the source.
- **Fix:** Trace the forward path; either use the module or stop marking it trainable.
- **Reference:** https://pytorch.org/docs/stable/notes/autograd.html

### `INPUT-001` — Loss does not depend on the input  _(FAIL)_

- **What:** Perturbing the input does not change the loss — the model ignores it.
- **Why it's silent:** A wiring bug (wrong tensor used, input overwritten) makes the model learn a constant; metrics can still move, hiding the fault.
- **Fix:** Trace the forward path from the input tensor to the loss.
- **Reference:** https://pytorch.org/docs/stable/notes/autograd.html

### `LOSS-001` — Loss is detached from the parameters  _(FAIL)_

- **What:** The loss tensor does not require grad, so backward() updates nothing.
- **Why it's silent:** A stray .detach(), .item(), or torch.no_grad() severs the graph; training runs but no parameter ever changes.
- **Fix:** Compute the loss from model outputs without detaching it.
- **Reference:** https://pytorch.org/docs/stable/notes/autograd.html

### `LOSS-002` — Loss is not a scalar  _(FAIL)_

- **What:** backward() needs a scalar; a non-reduced loss raises or misbehaves.
- **Why it's silent:** Forgetting reduction='mean' (or a .mean()) leaves a per-element loss.
- **Fix:** Reduce the loss to a single number before backward().
- **Reference:** https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html

### `OPT-001` — Trainable parameter is missing from the optimizer  _(FAIL)_

- **What:** A requires_grad parameter is not in any optimizer param group.
- **Why it's silent:** Building the optimizer before adding/replacing a module (or filtering params wrong) means it never updates, even though its grad is computed.
- **Fix:** Construct the optimizer from the final model.parameters().
- **Reference:** https://pytorch.org/docs/stable/optim.html

### `TIE-001` — Tied (shared-storage) weights  _(INFO)_

- **What:** Two parameters share the same storage.
- **Why it's silent:** Common and correct for LM embedding/output tying, but a surprise tie couples updates you meant to keep separate.
- **Fix:** Confirm the tie is intentional.
- **Reference:** https://arxiv.org/abs/1608.05859

## Optimizer

### `OPT-002` — Frozen parameter is registered with the optimizer  _(WARN)_

- **What:** A requires_grad=False parameter sits in the optimizer.
- **Why it's silent:** Harmless for updates, but weight decay can still act on it and it wastes optimizer state memory.
- **Fix:** Pass only trainable params to the optimizer.
- **Reference:** https://pytorch.org/docs/stable/optim.html

## Gradient health

### `NAN-002` — NaN/Inf in gradients  _(FAIL)_

- **What:** Gradients become non-finite during backward.
- **Why it's silent:** A finite forward can still produce non-finite grads (sqrt(0), log near 0), silently poisoning the optimizer state.
- **Fix:** Add epsilons to unstable ops; consider gradient clipping.
- **Reference:** https://pytorch.org/docs/stable/amp.html

### `GRAD-002` — Exploding gradient norm  _(WARN)_

- **What:** A gradient norm is very large on the first step.
- **Why it's silent:** Predicts divergence/NaN within a few steps.
- **Fix:** Lower the learning rate or apply clip_grad_norm_.
- **Reference:** https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html

### `GRAD-003` — Vanishing gradient norm  _(WARN)_

- **What:** The global gradient norm is near zero.
- **Why it's silent:** Little or no learning signal reaches the parameters.
- **Fix:** Check activations/normalization and that the loss is connected.
- **Reference:** https://pytorch.org/docs/stable/notes/autograd.html

### `GRAD-004` — Parameter has an all-zero gradient  _(WARN)_

- **What:** Gradient exists but is exactly zero this step.
- **Why it's silent:** Often a masking/detach bug or a saturated path upstream.
- **Fix:** Check for masks that zero the contribution, or a dead activation.
- **Reference:** https://pytorch.org/docs/stable/notes/autograd.html

## Numerical stability

### `NAN-001` — NaN/Inf in forward activations  _(FAIL)_

- **What:** A module emits non-finite activations.
- **Why it's silent:** Surfaces downstream as a useless loss=nan; the real culprit is the first module to go non-finite.
- **Fix:** Inspect the named module; common causes are log/div by zero, fp16 overflow, or bad init.
- **Reference:** https://pytorch.org/docs/stable/generated/torch.isfinite.html

## Activations

### `RELU-001` — Dead ReLU units  _(WARN)_

- **What:** A ReLU outputs almost all zeros for the batch.
- **Why it's silent:** Units stuck negative pass no gradient and never recover, wasting capacity — usually bad init or too-high learning rate.
- **Fix:** Check initialization/LR; consider LeakyReLU/GELU.
- **Reference:** https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
