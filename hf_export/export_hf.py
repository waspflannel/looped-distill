import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

TEACHER = "Qwen/Qwen3-1.7B"
TEACHER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
OUTER = ("model.embed_tokens.weight", "model.norm.weight")
COPIED = ("generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
CODE = ("configuration_looped_qwen3.py", "modeling_looped_qwen3.py")
HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Turn a distill weights.pt into a Hugging Face model folder")
    parser.add_argument("weights")
    parser.add_argument("out")
    parser.add_argument("--teacher-dir", help="local copy of Qwen3-1.7B; downloaded when omitted")
    parser.add_argument("--unique-layers", type=int, default=4)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    return parser.parse_args()


def outer_weights(teacher_dir):
    index = json.loads((teacher_dir / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for name in OUTER:
        with safe_open(teacher_dir / index[name], framework="pt") as shard:
            tensors[name] = shard.get_tensor(name)
    return tensors


def model_config(teacher_dir, settings):
    config = json.loads((teacher_dir / "config.json").read_text())
    config.pop("transformers_version", None)
    config["architectures"] = ["LoopedQwen3ForCausalLM"]
    config["model_type"] = "looped_qwen3"
    config["auto_map"] = {
        "AutoConfig": "configuration_looped_qwen3.LoopedQwen3Config",
        "AutoModelForCausalLM": "modeling_looped_qwen3.LoopedQwen3ForCausalLM",
    }
    config["unique_layers"] = settings["unique_layers"]
    config["loops"] = settings["loops"]
    config["lora_rank"] = settings["lora_rank"]
    config["lora_alpha"] = settings["lora_alpha"]
    return config


def main():
    args = parse_args()
    teacher_dir = Path(args.teacher_dir or snapshot_download(TEACHER, revision=TEACHER_REVISION))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.weights, map_location="cpu", weights_only=True)
    if any(".messages." in name for name in payload["student"]):
        raise SystemExit("pass messages are not supported by the exported model")
    tensors = {name: tensor.to(torch.bfloat16).contiguous() for name, tensor in payload["student"].items()}
    tensors.update(outer_weights(teacher_dir))
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    settings = {
        "unique_layers": args.unique_layers,
        "loops": args.loops,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    }
    (out / "config.json").write_text(json.dumps(model_config(teacher_dir, settings), indent=2) + "\n")
    for name in COPIED:
        shutil.copy(teacher_dir / name, out / name)
    for name in CODE:
        shutil.copy(HERE / name, out / name)
    print(f"wrote {len(tensors)} tensors to {out / 'model.safetensors'}, step {payload['state']['step']}")


if __name__ == "__main__":
    main()
