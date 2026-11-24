import json
import math
import os
import random
import re
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from config import (
    GSM8K,
    GSM8K_REVISION,
    MMLU,
    MMLU_REVISION,
    TEACHER,
    TEACHER_REVISION,
    load_config,
    log,
)
from data import PackedStream
from distill import CHECKPOINT, WEIGHTS
from model import load_qwen, load_student
from storage import DriveFolder

LETTERS = ("A", "B", "C", "D")
GSM8K_INSTRUCTION = "Solve the problem step by step, then end with 'The answer is N.'"


def held_out_sequences(tokenizer, config):
    stream = PackedStream(tokenizer, config, held_out=True)
    try:
        return [stream.next_batch(1, config["seq_len"]) for _ in range(config["ppl_sequences"])]
    finally:
        stream.close()


def perplexity(model, sequences, device):
    total_nll, total_tokens = 0.0, 0
    for tokens in sequences:
        tokens = tokens.to(device)
        logits = model(input_ids=tokens[:, :-1], use_cache=False).logits
        total_nll += F.cross_entropy(logits.float().flatten(0, 1), tokens[:, 1:].flatten(), reduction="sum").item()
        total_tokens += tokens[:, 1:].numel()
    return math.exp(total_nll / total_tokens)


def sample_rows(dataset, count, seed):
    indices = random.Random(seed).sample(range(len(dataset)), count)
    return [dataset[index] for index in indices]


def mmlu_prompt(row):
    subject = row["subject"].replace("_", " ")
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"]))
    return f"The following is a multiple choice question about {subject}.\n\n{row['question']}\n{options}\nAnswer:"


def mmlu_accuracy(model, tokenizer, rows, device):
    letter_ids = [tokenizer.encode(f" {letter}")[0] for letter in LETTERS]
    correct = 0
    for row in rows:
        input_ids = tokenizer(mmlu_prompt(row), return_tensors="pt").input_ids.to(device)
        last = model(input_ids=input_ids, use_cache=False).logits[0, -1]
        correct += int(last[letter_ids].argmax().item() == row["answer"])
    return correct / len(rows)


def gsm8k_prompt(tokenizer, row):
    messages = [{"role": "user", "content": f"{row['question']}\n\n{GSM8K_INSTRUCTION}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def last_number(text):
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "").rstrip(".") if numbers else None


def gsm8k_replies(model, tokenizer, rows, config, device):
    replies = []
    for start in range(0, len(rows), config["eval_batch"]):
        prompts = [gsm8k_prompt(tokenizer, row) for row in rows[start : start + config["eval_batch"]]]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        output = model.generate(**batch, max_new_tokens=config["gsm8k_max_new_tokens"], do_sample=False)
        replies += tokenizer.batch_decode(output[:, batch.input_ids.shape[1] :], skip_special_tokens=True)
    return replies


def gsm8k_accuracy(replies, rows):
    correct = 0
    for reply, row in zip(replies, rows):
        expected = row["answer"].split("####")[-1].strip().replace(",", "")
        correct += int(last_number(reply) == expected)
    return correct / len(rows)


def eval_tasks(tokenizer, config):
    mmlu = load_dataset(MMLU, "all", split="test", revision=MMLU_REVISION)
    gsm8k = load_dataset(GSM8K, "main", split="test", revision=GSM8K_REVISION)
    return {
        "sequences": held_out_sequences(tokenizer, config),
        "mmlu": sample_rows(mmlu, config["mmlu_samples"], config["seed"]),
        "gsm8k": sample_rows(gsm8k, config["gsm8k_samples"], config["seed"]),
    }


def load_trained_student(config, device, folder):
    student = load_student(config, device)
    artifact = folder.download(WEIGHTS) or folder.download(CHECKPOINT)
    if artifact is None:
        log(config, "eval", "no checkpoint yet, evaluating the student as built")
        return student.eval(), 0
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    student.load_state_dict(payload["student"], strict=False)
    return student.eval(), payload["state"]["step"]


def evaluate(label, model, tokenizer, tasks, config, device):
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        replies = gsm8k_replies(model, tokenizer, tasks["gsm8k"], config, device)
        result = {
            "perplexity": perplexity(model, tasks["sequences"], device),
            "mmlu": mmlu_accuracy(model, tokenizer, tasks["mmlu"], device),
            "gsm8k": gsm8k_accuracy(replies, tasks["gsm8k"]),
            "gsm8k_first_reply": replies[0],
        }
    log(config, "eval", f"{label}: ppl {result['perplexity']:.3f} mmlu {result['mmlu']:.3f} gsm8k {result['gsm8k']:.3f}")
    return result


def markdown_table(results, config, step):
    lines = [
        f"# Looped student vs teacher (step {step})",
        "",
        (
            f"K={config['unique_layers']}, L={config['loops']}, init={config['init']}, "
            f"LoRA rank {config['lora_rank']}, messages {'on' if config['messages'] else 'off'}"
        ),
        "",
        (
            f"| | Held-out PPL ({config['ppl_sequences']} x {config['seq_len']}) | "
            f"MMLU ({config['mmlu_samples']}) | GSM8K ({config['gsm8k_samples']}) |"
        ),
        "|---|---:|---:|---:|",
    ]
    for label, result in results.items():
        lines.append(f"| {label} | {result['perplexity']:.3f} | {100 * result['mmlu']:.1f}% | {100 * result['gsm8k']:.1f}% |")
    return "\n".join(lines) + "\n"


def main():
    config = load_config()
    folder = DriveFolder(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER, revision=TEACHER_REVISION, padding_side="left")
    tasks = eval_tasks(tokenizer, config)
    teacher = load_qwen(device).eval()
    student, step = load_trained_student(config, device, folder)
    results = {
        "teacher": evaluate("teacher", teacher, tokenizer, tasks, config, device),
        "student": evaluate("student", student, tokenizer, tasks, config, device),
    }
    report = {"step": step, "settings": {key: config[key] for key in config if key != "session_start"}, "results": results}
    table = markdown_table(results, config, step)
    print(table)
    folder.save_text("eval.json", json.dumps(report, indent=1))
    folder.save_text("eval.md", table)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
