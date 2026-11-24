import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from config import TEACHER, TEACHER_REVISION, load_config, log

LINEAR_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
NORM_NAMES = ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")
OUTER_NAMES = ("model.embed_tokens.", "model.norm.", "lm_head.")
CATEGORIES = ("outer", "unique", "shared", "lora", "messages")


class FastRMSNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.eps = norm.variance_epsilon

    def forward(self, x):
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(dtype)


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(base.weight.new_zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))

    def forward(self, x):
        base_out = self.base(x)
        low_rank = F.linear(x, self.lora_a) * self.scale
        shape = base_out.shape
        return torch.addmm(
            base_out.reshape(-1, shape[-1]),
            low_rank.reshape(-1, low_rank.shape[-1]),
            self.lora_b.T,
        ).reshape(shape)

    @torch.no_grad()
    def start_fresh(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lora_b.zero_()

    @torch.no_grad()
    def start_from(self, original_weight):
        residual = original_weight.float() - self.base.weight.float()
        if residual.abs().max() == 0:
            self.start_fresh()
            return
        rank = self.lora_a.shape[0]
        left, singular, right = torch.svd_lowrank(residual, q=min(rank + 8, *residual.shape), niter=4)
        root = (singular[:rank] / self.scale).sqrt()
        self.lora_b.copy_(left[:, :rank] * root)
        self.lora_a.copy_(root[:, None] * right[:, :rank].T)


class PassMessage(nn.Module):
    def __init__(self, hidden_size, rank, eps):
        super().__init__()
        self.eps = eps
        self.read_query = nn.Parameter(torch.zeros(hidden_size))
        self.down = nn.Linear(2 * hidden_size, rank, bias=False)
        self.forget = nn.Linear(rank, hidden_size, bias=False)
        self.gain = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.forget.weight)
        nn.init.zeros_(self.gain.weight)

    def rms(self, x):
        return F.rms_norm(x, (x.shape[-1],), eps=self.eps)

    def read(self, earlier_outputs):
        stacked = torch.stack(earlier_outputs)
        scores = (self.rms(stacked) * self.read_query).sum(-1) / math.sqrt(stacked.shape[-1])
        weights = scores.softmax(dim=0).unsqueeze(-1)
        return (weights * stacked).sum(dim=0)

    def forward(self, hidden_states, earlier_outputs):
        message = self.read(earlier_outputs)
        mixed = self.down(torch.cat([self.rms(hidden_states), self.rms(message)], dim=-1))
        keep = 1 - torch.tanh(self.forget(mixed))
        return keep * hidden_states + torch.tanh(self.gain(mixed)) * message


class LoopCore(nn.Module):
    def __init__(self, shared, passes, messages):
        super().__init__()
        self.shared = nn.ModuleList(shared)
        self.passes = nn.ModuleList(nn.ModuleList(layers) for layers in passes)
        self.messages = nn.ModuleList(messages)

    def forward(self, hidden_states, **layer_kwargs):
        pass_outputs = [hidden_states]
        for loop, layers in enumerate(self.passes):
            if self.messages and loop > 0:
                hidden_states = self.messages[loop - 1](hidden_states, pass_outputs)
            for layer in layers:
                hidden_states = layer(hidden_states, **layer_kwargs)
            pass_outputs.append(hidden_states)
        return hidden_states


def average_layers(layers):
    shared = copy.deepcopy(layers[0])
    states = [layer.state_dict() for layer in layers]
    averaged = {name: torch.stack([state[name].float() for state in states]).mean(0) for name in states[0]}
    shared.load_state_dict(averaged)
    return shared


def shared_sources(middle, position, shared_size, config):
    loops = config["loops"]
    if config["init"] == "average":
        return [middle[loop * shared_size + position] for loop in range(loops)]
    return [middle[position * loops + config["every_nth_offset"]]]


def pass_layer(model_config, shared, original, layer_idx, config):
    with torch.device("meta"):
        view = Qwen3DecoderLayer(model_config, layer_idx)
    for name in NORM_NAMES:
        view.set_submodule(name, shared.get_submodule(name))
    for name in LINEAR_NAMES:
        base = shared.get_submodule(name)
        if config["lora_rank"] == 0:
            view.set_submodule(name, base)
            continue
        lora = LoRALinear(base, config["lora_rank"], config["lora_alpha"])
        if config["lora_init"] == "svd":
            lora.start_from(original.get_submodule(name).weight)
        else:
            lora.start_fresh()
        view.set_submodule(name, lora)
    return view


def pass_messages(model_config, shared, config):
    if not config["messages"]:
        return []
    weight = shared[0].self_attn.q_proj.weight
    return [
        PassMessage(model_config.hidden_size, config["message_rank"], model_config.rms_norm_eps).to(weight)
        for _ in range(config["loops"] - 1)
    ]


def wrap_norms(layer):
    for name in NORM_NAMES:
        layer.set_submodule(name, FastRMSNorm(layer.get_submodule(name)))
    return layer


def build_student(model, config):
    layers = list(model.model.layers)
    unique, loops = config["unique_layers"], config["loops"]
    middle = layers[unique : len(layers) - unique]
    if len(middle) % loops:
        raise ValueError(f"{len(middle)} middle layers do not split into {loops} loops")
    shared_size = len(middle) // loops
    shared = [
        wrap_norms(average_layers(shared_sources(middle, position, shared_size, config)))
        for position in range(shared_size)
    ]
    passes = []
    for loop in range(loops):
        layers_in_pass = []
        for position in range(shared_size):
            depth = loop * shared_size + position
            layers_in_pass.append(pass_layer(model.config, shared[position], middle[depth], unique + depth, config))
        passes.append(layers_in_pass)
    core = LoopCore(shared, passes, pass_messages(model.config, shared, config))
    head = [wrap_norms(layer) for layer in layers[:unique]]
    tail = [wrap_norms(layer) for layer in layers[len(layers) - unique :]]
    model.model.layers = nn.ModuleList([*head, core, *tail])
    return model


def category(name):
    if name.startswith(OUTER_NAMES):
        return "outer"
    if ".lora_" in name:
        return "lora"
    if ".messages." in name:
        return "messages"
    if ".shared." in name:
        return "shared"
    return "unique"


def parameter_counts(model):
    counts = dict.fromkeys(CATEGORIES, 0)
    for name, param in model.named_parameters():
        counts[category(name)] += param.numel()
    counts["total"] = sum(counts.values())
    return counts


def select_trainable(student, trainable):
    for name, param in student.named_parameters():
        param.requires_grad_(category(name) in trainable)


def trainable_state(student):
    return {name: param.detach() for name, param in student.named_parameters() if param.requires_grad}


def load_qwen(device):
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, revision=TEACHER_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    return model.to(device)


def share_outer(student, teacher):
    student.model.embed_tokens = teacher.model.embed_tokens
    student.model.norm = teacher.model.norm
    student.lm_head = teacher.lm_head


def load_student(config, device):
    torch.manual_seed(config["seed"])
    with torch.no_grad():
        model = load_qwen(device)
        if "outer" in config["trainable"]:
            model = model.float()
        else:
            model.model.layers.float()
        return build_student(model, config)


def print_counts(config, teacher_counts, student_counts):
    log(config, "model", f"teacher: {teacher_counts['total'] / 1e6:8.1f}M parameters")
    for name in (*CATEGORIES, "total"):
        log(config, "model", f"student {name:8}: {student_counts[name] / 1e6:8.1f}M")
    layers = student_counts["unique"] + student_counts["shared"]
    teacher_layers = teacher_counts["total"] - teacher_counts["outer"]
    log(config, "model", f"student decoder weights are {100 * layers / teacher_layers:.1f}% of the teacher's")


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    model = load_qwen(device)
    teacher_counts = parameter_counts(model)
    with torch.no_grad():
        student = build_student(model, config)
    shared_size = len(student.model.layers[config["unique_layers"]].shared)
    log(
        config,
        "model",
        f"{config['unique_layers']} unique + {shared_size} shared x {config['loops']} loops + "
        f"{config['unique_layers']} unique = {student.config.num_hidden_layers} layer passes and KV cache slots",
    )
    print_counts(config, teacher_counts, parameter_counts(student))


if __name__ == "__main__":
    main()
