import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import (
    TEACHER,
    TEACHER_REVISION,
    load_config,
    log,
    out_of_budget,
    units_used,
)
from data import PackedStream, PrefetchStream
from model import (
    category,
    load_qwen,
    load_student,
    select_trainable,
    share_outer,
    trainable_state,
)
from storage import DriveFolder, utc_now

CHECKPOINT = "checkpoint.pt"
WEIGHTS = "weights.pt"
STATE = "distill-state.json"
PROGRESS = "progress.json"
ADAPTER_CATEGORIES = ("lora", "messages")
LOSS_CHUNK = 512


class Compiled:
    def __init__(self, target, config, label, dynamic=False):
        self.eager = target
        self.config = config
        self.label = label
        self.active = target
        if config["compile"]:
            try:
                self.active = torch.compile(target, dynamic=dynamic)
            except Exception as error:
                log(config, "compile", f"{label} compile setup failed ({error}), using eager")

    def __call__(self, *args, **kwargs):
        if self.active is self.eager:
            return self.eager(*args, **kwargs)
        try:
            return self.active(*args, **kwargs)
        except Exception as error:
            log(self.config, "compile", f"{self.label} compile failed ({error}), falling back to eager")
            self.active = self.eager
            return self.eager(*args, **kwargs)


def chunk_kl_and_grad(
    student_hidden, teacher_hidden, targets, student_weight, teacher_weight, total_tokens, temperature, ce_weight
):
    student_logits = F.linear(student_hidden.to(student_weight.dtype), student_weight).float()
    teacher_logits = F.linear(teacher_hidden.to(teacher_weight.dtype), teacher_weight).float()
    if temperature == 1.0:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        kl_sum = F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction="sum")
        ce_sum = F.nll_loss(student_log_probs, targets, reduction="sum")
        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()
        if ce_weight == 0.0:
            grad_logits = (student_probs - teacher_probs) / total_tokens
        else:
            grad_logits = (student_probs - (1 - ce_weight) * teacher_probs) / total_tokens
            delta = torch.full(
                (targets.shape[0], 1), -ce_weight / total_tokens, device=grad_logits.device, dtype=grad_logits.dtype
            )
            grad_logits = grad_logits.scatter_add(1, targets.unsqueeze(1), delta)
    else:
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        kl_sum = F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction="sum") * (
            temperature * temperature
        )
        grad_logits = (student_log_probs.exp() - teacher_log_probs.exp()) * (
            (1 - ce_weight) * temperature / total_tokens
        )
        ce_log_probs = F.log_softmax(student_logits, dim=-1)
        ce_sum = F.nll_loss(ce_log_probs, targets, reduction="sum")
        if ce_weight != 0.0:
            grad_logits = grad_logits + ce_log_probs.exp() * (ce_weight / total_tokens)
            delta = torch.full(
                (targets.shape[0], 1), -ce_weight / total_tokens, device=grad_logits.device, dtype=grad_logits.dtype
            )
            grad_logits = grad_logits.scatter_add(1, targets.unsqueeze(1), delta)
    grad_logits_cast = grad_logits.to(student_weight.dtype)
    grad_hidden = (grad_logits_cast @ student_weight).to(student_hidden.dtype)
    return kl_sum, ce_sum, grad_hidden, grad_logits_cast


class ChunkedDistillLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_hidden, teacher_hidden, targets, student_weight, teacher_weight, temperature, ce_weight, chunk_fn):
        flat_student = student_hidden.reshape(-1, student_hidden.shape[-1])
        flat_teacher = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
        flat_targets = targets.reshape(-1)
        total_tokens = flat_targets.numel()
        grad_student = torch.empty_like(flat_student)
        grad_weight = torch.zeros_like(student_weight) if student_weight.requires_grad else None
        kl_total = student_hidden.new_zeros((), dtype=torch.float32)
        ce_total = student_hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, total_tokens, LOSS_CHUNK):
            end = min(start + LOSS_CHUNK, total_tokens)
            s_chunk = flat_student[start:end]
            t_chunk = flat_teacher[start:end]
            y_chunk = flat_targets[start:end]
            kl_sum, ce_sum, grad_hidden, grad_logits = chunk_fn(
                s_chunk, t_chunk, y_chunk, student_weight, teacher_weight, total_tokens, temperature, ce_weight
            )
            kl_total = kl_total + kl_sum
            ce_total = ce_total + ce_sum
            grad_student[start:end] = grad_hidden
            if grad_weight is not None:
                grad_weight.addmm_(grad_logits.T, s_chunk.to(student_weight.dtype))
        kl = kl_total / total_tokens
        ce = ce_total / total_tokens
        loss = (1 - ce_weight) * kl + ce_weight * ce
        ctx.save_for_backward(grad_student.view_as(student_hidden), grad_weight)
        ctx.mark_non_differentiable(kl, ce)
        return loss, kl, ce

    @staticmethod
    def backward(ctx, grad_loss, _grad_kl, _grad_ce):
        grad_student, grad_weight = ctx.saved_tensors
        scale = grad_loss.to(grad_student.dtype)
        scaled_student = grad_student * scale
        scaled_weight = grad_weight * grad_loss.to(grad_weight.dtype) if grad_weight is not None else None
        return scaled_student, None, None, scaled_weight, None, None, None, None


def chunked_distill_loss(
    student_hidden, teacher_hidden, targets, student_weight, teacher_weight, temperature, ce_weight, chunk_fn=chunk_kl_and_grad
):
    return ChunkedDistillLoss.apply(
        student_hidden, teacher_hidden, targets, student_weight, teacher_weight, temperature, ce_weight, chunk_fn
    )


def warmup_cosine(warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def parameter_groups(student, config):
    weights, adapters = [], []
    for name, param in student.named_parameters():
        if param.requires_grad:
            (adapters if category(name) in ADAPTER_CATEGORIES else weights).append(param)
    return [
        {"params": weights, "lr": config["lr"]},
        {"params": adapters, "lr": config["adapter_lr"]},
    ]


def busy(upload):
    return upload is not None and not upload.done()


def wait_for(upload):
    if upload is not None:
        upload.result()


class Distillation:
    def __init__(self, config, device, folder):
        self.config = config
        self.device = device
        self.folder = folder
        self.tokens_per_step = config["micro_batch"] * config["accumulation"] * config["seq_len"]
        self.total_steps = config["token_budget"] // self.tokens_per_step
        self.tokenizer = AutoTokenizer.from_pretrained(TEACHER, revision=TEACHER_REVISION)
        self.teacher = load_qwen(device).eval().requires_grad_(False)
        self.student = load_student(config, device)
        if "outer" not in config["trainable"]:
            share_outer(self.student, self.teacher)
        select_trainable(self.student, config["trainable"])
        if config["gradient_checkpointing"]:
            self.student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.student.train()
        self.teacher_backbone = Compiled(self.teacher.model, config, "teacher")
        self.student_backbone = Compiled(self.student.model, config, "student")
        self.loss_chunk = Compiled(chunk_kl_and_grad, config, "loss", dynamic=True)
        self.trainable = [param for param in self.student.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(
            parameter_groups(self.student, config), weight_decay=config["weight_decay"], fused=device.type == "cuda"
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, warmup_cosine(config["warmup_steps"], self.total_steps)
        )
        self.state = {"step": 0, "tokens": 0, "done": False}
        self.latest = {"loss": None, "kl": None, "ce": None, "tokens_per_second": None}
        self.stream = None
        self.checkpoint_upload = None
        self.progress_upload = None

    def open_stream(self, state=None):
        raw = PackedStream(self.tokenizer, self.config, held_out=False, state=state)
        return PrefetchStream(
            raw, self.config["micro_batch"], self.config["seq_len"], pin_memory=self.device.type == "cuda"
        )

    def restore(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.student.load_state_dict(checkpoint["student"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.state = checkpoint["state"]
        self.stream = self.open_stream(state=checkpoint["stream"])
        log(self.config, "distill", f"resumed at step {self.state['step']}, {self.state['tokens']:,} tokens seen")
        print(f"RESUMED step={self.state['step']}", flush=True)

    def snapshot_checkpoint(self):
        checkpoint = {
            "student": trainable_state(self.student),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state": self.state,
            "stream": self.stream.state(),
        }
        torch.save(checkpoint, self.folder.local(CHECKPOINT))
        return self.state["step"], json.dumps(self.state, indent=1)

    def upload_checkpoint(self, step, state_text):
        checkpoint_md5 = self.folder.save(CHECKPOINT)
        self.folder.save_text(STATE, state_text)
        print(f"CHECKPOINT VERIFIED step={step} md5={checkpoint_md5}", flush=True)

    def save_weights(self):
        weights = {
            "student": {name: param.to(torch.bfloat16) for name, param in trainable_state(self.student).items()},
            "state": self.state,
        }
        torch.save(weights, self.folder.local(WEIGHTS))
        self.folder.save(WEIGHTS)
        self.folder.save_text(STATE, json.dumps(self.state, indent=1))

    def progress(self):
        return {
            "step": self.state["step"],
            "total_steps": self.total_steps,
            "tokens": self.state["tokens"],
            **self.latest,
            "elapsed_seconds": round(time.time() - self.config["session_start"]),
            "units_used": round(units_used(self.config), 2),
            "time": utc_now(),
        }

    def announce_progress(self):
        text = json.dumps(self.progress())
        print(f"PROGRESS {text}", flush=True)
        return text

    def report_progress(self, uploads):
        text = self.announce_progress()
        if busy(self.progress_upload):
            log(self.config, "distill", f"previous {PROGRESS} upload still running, skipping this one")
        else:
            self.progress_upload = uploads.submit(self.folder.save_text, PROGRESS, text)

    def start_checkpoint(self, uploads):
        if busy(self.checkpoint_upload):
            log(self.config, "distill", f"checkpoint at step {self.state['step']} skipped, previous upload still running")
        else:
            self.checkpoint_upload = uploads.submit(self.upload_checkpoint, *self.snapshot_checkpoint())

    def raise_upload_errors(self):
        for upload in (self.checkpoint_upload, self.progress_upload):
            if not busy(upload):
                wait_for(upload)

    def finish(self):
        wait_for(self.checkpoint_upload)
        if self.state["done"]:
            self.save_weights()
        else:
            self.upload_checkpoint(*self.snapshot_checkpoint())
        wait_for(self.progress_upload)
        self.folder.save_text(PROGRESS, self.announce_progress())

    def micro_step(self):
        tokens = self.stream.next_batch().to(self.device, non_blocking=True)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        with torch.no_grad():
            teacher_hidden = self.teacher_backbone(input_ids=inputs, use_cache=False).last_hidden_state
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            student_hidden = self.student_backbone(input_ids=inputs, use_cache=False).last_hidden_state
        loss, kl, ce = chunked_distill_loss(
            student_hidden,
            teacher_hidden,
            targets,
            self.student.lm_head.weight,
            self.teacher.lm_head.weight,
            self.config["temperature"],
            self.config["ce_weight"],
            self.loss_chunk,
        )
        (loss / self.config["accumulation"]).backward()
        self.state["tokens"] += tokens.numel()
        return kl.detach(), ce.detach()

    def optimizer_step(self):
        parts = [self.micro_step() for _ in range(self.config["accumulation"])]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, self.config["grad_clip"])
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state["step"] += 1
        kl = torch.stack([part[0] for part in parts]).mean()
        ce = torch.stack([part[1] for part in parts]).mean()
        return kl, ce, grad_norm

    def log_progress(self, kl, ce, grad_norm, tokens_per_second):
        loss = (1 - self.config["ce_weight"]) * kl + self.config["ce_weight"] * ce
        self.latest = {"loss": loss, "kl": kl, "ce": ce, "tokens_per_second": round(tokens_per_second)}
        lr = self.scheduler.get_last_lr()[0]
        log(
            self.config,
            "distill",
            f"step {self.state['step']}/{self.total_steps} tokens {self.state['tokens']:,} "
            f"loss {loss:.4f} kl {kl:.4f} ce {ce:.4f} grad {grad_norm:.3f} lr {lr:.2e} "
            f"{tokens_per_second:,.0f} tok/s, {units_used(self.config):.1f} of {self.config['budget_units']} units",
        )

    def run(self):
        if self.stream is None:
            self.stream = self.open_stream()
        uploads = ThreadPoolExecutor(max_workers=2)
        try:
            log(self.config, "distill", f"{self.total_steps} steps of {self.tokens_per_step:,} tokens")
            last_progress = window_start = time.time()
            last_checkpoint = None
            window_tokens = self.state["tokens"]
            while self.state["step"] < self.total_steps:
                if out_of_budget(self.config):
                    log(self.config, "distill", "unit budget reached, saving and stopping")
                    break
                kl, ce, grad_norm = self.optimizer_step()
                self.raise_upload_errors()
                if last_checkpoint is None:
                    last_checkpoint = time.time()
                if self.state["step"] % self.config["log_every"] == 0:
                    kl_val, ce_val, grad_val = kl.item(), ce.item(), grad_norm.item()
                    tokens_per_second = (self.state["tokens"] - window_tokens) / (time.time() - window_start)
                    self.log_progress(kl_val, ce_val, grad_val, tokens_per_second)
                    window_start, window_tokens = time.time(), self.state["tokens"]
                if time.time() - last_progress > self.config["progress_minutes"] * 60:
                    self.report_progress(uploads)
                    last_progress = time.time()
                if time.time() - last_checkpoint > self.config["checkpoint_minutes"] * 60:
                    self.start_checkpoint(uploads)
                    last_checkpoint = time.time()
            self.state["done"] = self.state["step"] >= self.total_steps
            self.finish()
        finally:
            self.stream.close()
            uploads.shutdown(wait=True)


def main():
    config = load_config()
    folder = DriveFolder(config)
    state_path = folder.download(STATE) if config["resume"] else None
    if state_path is not None and json.loads(state_path.read_text())["done"]:
        log(config, "distill", f"{STATE} on Drive says distillation finished, skipping")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    if out_of_budget(config):
        log(config, "distill", "no unit budget left, skipping")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distillation = Distillation(config, device, folder)
    checkpoint_path = folder.download(CHECKPOINT) if config["resume"] else None
    if checkpoint_path is not None:
        distillation.restore(checkpoint_path)
    distillation.run()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
