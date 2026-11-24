import json
import sys
import time
from pathlib import Path

TEACHER = "Qwen/Qwen3-1.7B"
TEACHER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
FINEWEB = "HuggingFaceFW/fineweb-edu"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
MMLU = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
GSM8K = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"

FULL = {
    "drive_dir": "MyDrive/looped-distill",
    "resume": True,
    "unique_layers": 4,
    "loops": 2,
    "init": "average",
    "every_nth_offset": 0,
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_init": "svd",
    "messages": False,
    "message_rank": 64,
    "trainable": ["unique", "shared", "lora", "messages"],
    "fineweb_config": "sample-10BT",
    "held_out_every": 100,
    "seq_len": 1024,
    "micro_batch": 4,
    "accumulation": 8,
    "token_budget": 150_000_000,
    "lr": 5e-5,
    "adapter_lr": 2e-4,
    "weight_decay": 0.0,
    "warmup_steps": 200,
    "grad_clip": 1.0,
    "temperature": 1.0,
    "ce_weight": 0.0,
    "gradient_checkpointing": False,
    "compile": True,
    "log_every": 10,
    "checkpoint_minutes": 20,
    "progress_minutes": 5,
    "budget_units": 100,
    "gpu_units_per_hour": 6.73,
    "units_spent_before": 0,
    "reserve_minutes": 30,
    "ppl_sequences": 64,
    "mmlu_samples": 200,
    "gsm8k_samples": 100,
    "gsm8k_max_new_tokens": 384,
    "eval_batch": 16,
    "seed": 0,
}

SMOKE = {
    "accumulation": 2,
    "token_budget": 250_000,
    "warmup_steps": 5,
    "log_every": 5,
    "budget_units": 10,
    "reserve_minutes": 20,
    "ppl_sequences": 4,
    "mmlu_samples": 8,
    "gsm8k_samples": 4,
    "gsm8k_max_new_tokens": 64,
    "eval_batch": 4,
}

CRASH_TEST = {
    **SMOKE,
    "token_budget": 6_000_000,
    "checkpoint_minutes": 3,
    "progress_minutes": 1,
}

MODES = {"full": {}, "smoke": SMOKE, "crash-test": CRASH_TEST}


def make_config(mode, run_name=None, run_dir=None, **overrides):
    config = {**FULL, **MODES[mode], **overrides}
    config["mode"] = mode
    config["session_start"] = time.time()
    config["run_dir"] = run_dir or f"{config['drive_dir']}/{run_name or mode}"
    return config


def load_config():
    return json.loads(Path(sys.argv[1]).read_text())


def units_used(config):
    hours = (time.time() - config["session_start"]) / 3600
    return config["units_spent_before"] + hours * config["gpu_units_per_hour"]


def out_of_budget(config):
    reserve = config["reserve_minutes"] / 60 * config["gpu_units_per_hour"]
    return units_used(config) >= config["budget_units"] - reserve


def log(config, stage, message):
    minutes = (time.time() - config["session_start"]) / 60
    print(f"[{stage:7} {minutes:6.1f} min {units_used(config):5.1f} units] {message}", flush=True)
