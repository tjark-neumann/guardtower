"""One-line pre-flight inside a Hugging Face Trainer.

    pip install guardtower-ml[hf]
"""
from guardtower.integrations.huggingface import GuardtowerCallback

# trainer = Trainer(
#     model=model,                 # e.g. a get_peft_model(...) LoRA model
#     args=training_args,
#     train_dataset=train_ds,
#     callbacks=[GuardtowerCallback(raise_on_error=True)],
# )
# trainer.train()
#
# On the first batch guardtower audits the model and, if something is
# misconfigured (adapter not in the optimizer, base unfrozen, no gradient
# path, ...), prints a report and aborts before the expensive part begins.
