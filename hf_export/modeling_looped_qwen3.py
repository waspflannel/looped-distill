import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM

from .configuration_looped_qwen3 import LoopedQwen3Config

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


class SharedNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.shared = (norm,)

    def forward(self, x):
        return self.shared[0](x)


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.shared = (base,)
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        base_out = self.shared[0](x)
        low_rank = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return base_out + low_rank * self.scale


def pass_layer(config, shared, layer_idx):
    view = Qwen3DecoderLayer(config, layer_idx)
    for name in NORM_NAMES:
        view.set_submodule(name, SharedNorm(shared.get_submodule(name)))
    for name in LINEAR_NAMES:
        view.set_submodule(name, LoRALinear(shared.get_submodule(name), config.lora_rank, config.lora_alpha))
    return view


class LoopCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        unique, loops = config.unique_layers, config.loops
        shared_size = (config.num_hidden_layers - 2 * unique) // loops
        self.shared = nn.ModuleList(Qwen3DecoderLayer(config, unique + position) for position in range(shared_size))
        self.passes = nn.ModuleList(
            nn.ModuleList(
                pass_layer(config, self.shared[position], unique + loop * shared_size + position)
                for position in range(shared_size)
            )
            for loop in range(loops)
        )

    def forward(self, hidden_states, **layer_kwargs):
        for layers in self.passes:
            for layer in layers:
                hidden_states = layer(hidden_states, **layer_kwargs)
        return hidden_states


class LoopedQwen3ForCausalLM(Qwen3ForCausalLM):
    config_class = LoopedQwen3Config

    def __init__(self, config):
        super().__init__(config)
        unique = config.unique_layers
        layers = self.model.layers
        head = list(layers[:unique])
        tail = list(layers[len(layers) - unique :])
        self.model.layers = nn.ModuleList([*head, LoopCore(config), *tail])
        self.post_init()
