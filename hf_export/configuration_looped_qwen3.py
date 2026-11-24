from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class LoopedQwen3Config(Qwen3Config):
    model_type = "looped_qwen3"

    def __init__(self, unique_layers=4, loops=2, lora_rank=64, lora_alpha=128, **kwargs):
        self.unique_layers = unique_layers
        self.loops = loops
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        super().__init__(**kwargs)
