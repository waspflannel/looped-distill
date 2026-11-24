# looped-distill

Rebuild an open-weights model as a **looped** transformer and distill it from the original, on one Colab GPU.

This repo turns Qwen3-1.7B into a smaller model where the 20 middle layers are replaced by one shared 10-layer block that runs twice. A small LoRA per pass lets the two passes behave differently. The looped student is then trained to match the original model's next-token distribution on FineWeb-Edu.

The question behind it: can existing open models be rebuilt this way so that bigger models fit into less VRAM, without training from scratch?

Trained weights: https://huggingface.co/waspflannel/Qwen3-1.7B-Looped-K4L2

## Results

One run with the defaults: 150M tokens, 4,577 steps, about 4.2 hours on one A100 40GB.

| | Params | Held-out FineWeb-Edu perplexity (64 x 1,024) | MMLU (200) | GSM8K (100) |
|---|---:|---:|---:|---:|
| Qwen3-1.7B (teacher) | 1,720.6M | 17.87 | 62.0% | 84.0% |
| Looped student | 1,267.0M | 20.07 | 29.0% | 0.0% |

On the same kind of text it trained on, the student is within about 12% of the teacher's perplexity while being 26% smaller. Knowledge and math did not transfer, because the training data was only web articles. Adding math and instruction data to the mix is the next thing to try.

## How the model is built

```
teacher:  [0 1 2 3] [4 ... 13 | 14 ... 23] [24 25 26 27]
student:  [0 1 2 3] [shared block S0..S9, pass 1 | same block, pass 2] [24 25 26 27]
```

- `unique_layers` (K, default 4): the first and last K layers stay as they are.
- `loops` (L, default 2): the middle 28 - 2K layers become one shared block of (28 - 2K) / L layers, run L times. Each token still runs 28 layer passes, so looping saves memory, not compute.
- `init`: `average` makes each shared layer the mean of the teacher layers it replaces. `every_nth` copies one of them instead.
- `lora_rank`, `lora_alpha` (default 64, 128): each pass gets its own LoRA on q, k, v, o, gate, up and down. With `lora_init = "svd"` it starts as the best rank-r approximation of the gap between the original layer and the shared one.
- `messages` (off by default): an optional gated mix of earlier passes' outputs before each new pass, inspired by Inkling 2. Not used in the run above.
- The KV cache keeps one slot per pass, so `generate` works unchanged.

## How it is trained

- The teacher (original Qwen3-1.7B) is frozen and runs forward only.
- The loss is the exact full-vocabulary KL from teacher to student, computed in 512-token chunks so the full logit tensors never sit in memory at once. `ce_weight` can mix in ordinary cross-entropy (default 0).
- Embeddings, the final norm and the output layer are shared with the teacher and frozen. The unique layers, the shared block and the LoRAs are trained with AdamW, warmup and cosine decay.
- FineWeb-Edu `sample-10BT` is streamed and packed into 1,024-token sequences. Every 100th document is held out for evaluation.
- `eval.py` compares teacher and student on held-out perplexity, MMLU and GSM8K.

## Running it on Colab

You need a Google account with Colab and enough Drive space for the run folder (about 13 GB for a full run with its checkpoint).

1. Upload this repo's files to `MyDrive/looped-distill/code/` in Google Drive.
2. Open `looped.ipynb` in Colab and pick an **A100** runtime (a full run refuses GPUs under 39 GB).
3. Set `RUN_ARGS` in the first cell. Start with `["--smoke"]`, which finishes in a few minutes.
4. Run all cells and approve the Google sign-in and the Drive mount.

`run.py` runs `model.py`, `distill.py`, `eval.py` and a final `storage.py` check in turn, and prints `RUN COMPLETE` when every output file on Drive matches `manifest.json`.

Useful options (`python run.py --help` lists all of them):

- `--smoke`, `--crash-test`: short runs to check the setup and the resume path.
- `--tokens N`: the token budget (default 150M).
- `--run-name NAME` or `--run-dir MyDrive/...`: where the outputs go.
- `--no-resume`: start fresh instead of continuing from the last checkpoint in the run folder (resuming is on by default).
- `--checkpoint-minutes N`: how often to save a resumable checkpoint (default 20).
- `--set KEY=VALUE`: override any setting in `config.py`, for example `--set loops=4` or `--set messages=true`.

### Compute budget

`budget_units` (default 100) stops training and saves a checkpoint before the estimated Colab compute units used reach that number. The estimate is `gpu_units_per_hour` (6.73, about what an A100 costs on Colab) times the session time, plus `units_spent_before` if you are resuming in a new session. A full 150M-token run takes about 30 to 35 units. Check your own plan's rates and set these to match.

### Outputs

Everything lands in the run folder on Drive:

| File | Contents |
|---|---|
| `weights.pt` | Final bfloat16 trainable weights (1.78 GiB) and the step count |
| `checkpoint.pt` | Resumable checkpoint: float32 weights, optimizer, scheduler and data position (about 10.7 GiB) |
| `distill-state.json`, `progress.json` | Step, tokens seen and the latest training stats |
| `eval.json`, `eval.md` | Teacher vs student results |
| `manifest.json` | Size and md5 of every file above |

Every file is written locally first, uploaded with the Drive API and checked against Drive's md5 before it counts as saved.

## Exporting to Hugging Face format

`hf_export/` turns `weights.pt` into a folder that loads with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`:

```
pip install torch transformers safetensors huggingface_hub
python hf_export/export_hf.py weights.pt hf-out
```

It adds the frozen embeddings and final norm from Qwen3-1.7B, writes `model.safetensors` in bfloat16, and copies the config, tokenizer and model code.

## Requirements

`requirements.txt` pins the versions the run used (torch 2.14, transformers 5.20). The notebook installs them.

## References

- Relaxed Recursive Transformers: https://arxiv.org/abs/2410.20672
- Ouro / LoopLM: https://arxiv.org/abs/2510.25741
- Huginn, recurrent depth: https://arxiv.org/abs/2502.05171
- Mixture-of-Recursions: https://arxiv.org/abs/2507.10524

## License

Apache-2.0. See `LICENSE`. Qwen3-1.7B is also Apache-2.0.
