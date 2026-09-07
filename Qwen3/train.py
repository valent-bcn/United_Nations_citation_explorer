"""
Usage:
    python train.py \
        --config config/train.yaml \
        --reset 
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # paths (from yaml)
    data_path: str = "/srv/scratch/branes/Qwen_finetune/data"
    eval_data_path: Optional[str] = None
    output_dir: str = "/srv/scratch/branes/Qwen_finetune/checkpoints"
    model_name_or_path: str = "Qwen/Qwen3-0.6B"

    # tokenisation
    max_length: int = 3024         # hard cap; sequences longer get truncated
    target_length: int = 512       # expected typical length

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )

    # training
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-3
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    eval_steps: int = 500 # check evaluation loss
    save_steps: int = 500 # save checkpoint + write logs
    logging_steps: int = 500 # print in terminal
    seed: int = 42
    bf16: bool = True


def load_config(yaml_path: str = "config/train.yaml") -> TrainConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    cfg = TrainConfig()
    
    for k, v in (raw or {}).items():
        if hasattr(cfg, k):
            current = getattr(cfg, k)
            if current is not None:
                try:
                    v = type(current)(v)
                except Exception:
                    pass  # fallback if conversion fails
                    
            setattr(cfg, k, v)
    return cfg


    


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl_or_json(path: str) -> list[dict]:
    with open(path) as f:
        first = f.read(1)
    with open(path) as f:
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def apply_chat_template_and_mask(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict:
    """
    Tokenise a conversation and build labels that are -100 everywhere
    except on assistant turns, so loss only trains on assistant outputs.

    Expected format of example["messages"]:
        [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    """
    messages = example["messages"]

    input_ids: list[int] = []
    labels:    list[int] = []

    for msg in messages:
        role = msg["role"]

        # Render just this single turn as text using the model's chat template
        turn_text = tokenizer.apply_chat_template(
            [msg],
            tokenize=False,
            add_generation_prompt=False,
        )
        turn_ids = tokenizer(turn_text, add_special_tokens=False)["input_ids"]

        input_ids.extend(turn_ids)
        if role == "assistant":
            labels.extend(turn_ids)          # train on assistant tokens
        else:
            labels.extend([-100] * len(turn_ids))   # mask everything else

    # Truncate from the LEFT so the most recent turns are always kept
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels    = labels[-max_length:]

    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         labels,
    }


def build_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    raw = load_jsonl_or_json(data_path)
    ds  = Dataset.from_list(raw)
    ds  = ds.map(
        lambda ex: apply_chat_template_and_mask(ex, tokenizer, max_length),
        remove_columns=ds.column_names,
        desc="Tokenising",
    )
    # drop samples with no assistant tokens to train on
    ds = ds.filter(lambda ex: any(l != -100 for l in ex["labels"]))
    return ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: TrainConfig):
    dtype = (
        torch.bfloat16
        if (cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False  # required for gradient checkpointing

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, eval_loader, device) -> dict:
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for batch in eval_loader:
        batch    = {k: v.to(device) for k, v in batch.items()}
        out      = model(**batch)
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss   += out.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss   = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))   # cap to avoid overflow
    model.train()
    return {"eval_loss": avg_loss, "perplexity": perplexity}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, reset: bool = False):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(cfg)

    train_ds = build_dataset(cfg.data_path, tokenizer, cfg.max_length)
    eval_ds  = (
        build_dataset(cfg.eval_data_path, tokenizer, cfg.max_length)
        if cfg.eval_data_path else None
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
    )
    eval_loader = (
        DataLoader(
            eval_ds,
            batch_size=cfg.per_device_train_batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
        )
        if eval_ds else None
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_optimizer_steps = (
        math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
        * cfg.num_epochs
    )
    warmup_steps = int(total_optimizer_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    # ---- optional resume ----
    start_epoch  = 0
    global_step  = 0
    accum_count  = 0
    running_loss = 0.0

    latest_ckpt = os.path.join(cfg.output_dir, "latest_checkpoint.pt")
    
    if not reset and os.path.exists(latest_ckpt):
        
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
    elif reset:
        print("--reset flag set, starting from scratch.")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- main loop ----
    model.train()
    optimizer.zero_grad()

    step_log = []
    epoch_log = []
    train_loss_log = []
    eval_loss_log = []
    perplexity_log = []

    for epoch in range(start_epoch, cfg.num_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            loss = model(**batch).loss / cfg.gradient_accumulation_steps
            loss.backward()
            avg = 0
            running_loss += loss.item()
            accum_count  += 1

            if accum_count % cfg.gradient_accumulation_steps != 0:
                continue

            # --- optimizer step ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % cfg.logging_steps == 0: 
                avg = running_loss / cfg.logging_steps
                lr  = scheduler.get_last_lr()[0]
                print(f"epoch {epoch+1} | step {global_step} | loss {avg:.4f} | lr {lr:.2e}")
                running_loss = 0.0

            if eval_loader and global_step % cfg.eval_steps == 0:
                m = evaluate(model, eval_loader, device)
                eval_loss = m['eval_loss']
                perplexity = m['perplexity']                
                print(f"  [eval] loss {eval_loss:.4f} | ppl {perplexity:.2f}")

                step_log.append(global_step)
                epoch_log.append(epoch+1)
                train_loss_log.append(avg) # Same as avg on the logging chunk
                eval_loss_log.append(eval_loss)
                perplexity_log.append(perplexity)

            if global_step % cfg.save_steps == 0:
                ckpt_dir = os.path.join(cfg.output_dir, f"{global_step}")
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                torch.save(
                    {
                        "epoch":           epoch,
                        "global_step":     global_step,
                        "model_state":     model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                    },
                    latest_ckpt,
                )
                print(f"  Saved checkpoint → {ckpt_dir}")
                
    log = pd.DataFrame({
        'epoch': epoch_log,
        'step': step_log,
        'train_loss': train_loss_log,
        'eval_loss': eval_loss_log,
        'perplexity': perplexity_log
    })

    log_path = os.path.join(cfg.output_dir, "train_log.csv")
    log.to_csv(log_path, index=False)
    print(f"Training log saved → {log_path}")

    
    final_dir = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Done. Final adapter saved → {final_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--reset", action="store_true", help="Ignore any existing checkpoint and start fresh")

    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, reset=args.reset)
