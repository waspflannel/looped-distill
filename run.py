import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from config import FULL, make_config, units_used
from storage import check_drive_access, drive_folder_names, local_stage_dir

CODE_DIR = Path(__file__).resolve().parent
STAGES = ("model.py", "distill.py", "eval.py", "storage.py")
MIN_GPU_MIB = 39_000
COLAB_AUTH_CREDENTIALS = "/content/.config/legacy_credentials/*/adc.json"


def setting(text):
    key, value = text.split("=", 1)
    if key not in FULL:
        raise argparse.ArgumentTypeError(f"{key} is not a config.py setting")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def parse_args():
    parser = argparse.ArgumentParser(description="Build, distill, evaluate and verify one looped Qwen3-1.7B run.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_const", const="smoke", dest="mode")
    mode.add_argument("--crash-test", action="store_const", const="crash-test", dest="mode")
    parser.set_defaults(mode="full")
    parser.add_argument("--run-name", help="folder under drive_dir, default the mode name")
    parser.add_argument("--run-dir", help="Drive path of the run folder, e.g. MyDrive/looped/runs/k4-l2")
    parser.add_argument("--tokens", type=int, help="token_budget")
    parser.add_argument("--checkpoint-minutes", type=float)
    parser.add_argument("--progress-minutes", type=float)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--set", action="append", default=[], type=setting, metavar="KEY=VALUE")
    return parser.parse_args()


def overrides(args):
    named = {
        "token_budget": args.tokens,
        "checkpoint_minutes": args.checkpoint_minutes,
        "progress_minutes": args.progress_minutes,
    }
    given = {key: value for key, value in named.items() if value is not None}
    return {**dict(args.set), **given, "resume": args.resume}


def check_gpu(mode):
    query = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    name, mib = subprocess.check_output(query, text=True).splitlines()[0].rsplit(",", 1)
    print(f"GPU {name.strip()} {int(mib)} MiB", flush=True)
    if mode == "full" and int(mib) < MIN_GPU_MIB:
        raise SystemExit(f"this GPU has {int(mib)} MiB; pick an A100 and run again (--smoke and --crash-test allow less)")


def use_colab_auth_credentials():
    found = sorted(glob.glob(COLAB_AUTH_CREDENTIALS))
    if found and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = found[0]


def write_config(config):
    path = local_stage_dir(drive_folder_names(config["run_dir"])) / "config.json"
    path.write_text(json.dumps(config, indent=1))
    return path


def run_command(*command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"{' '.join(command)} failed with exit code {process.returncode}")


def run_stage(script, config_path, config):
    started = time.time()
    print(f"== {script} starting", flush=True)
    run_command(sys.executable, "-u", str(CODE_DIR / script), str(config_path))
    minutes = (time.time() - started) / 60
    units = f"about {units_used(config):.1f} of {config['budget_units']} units used"
    print(f"== {script} done in {minutes:.1f} min, {units}", flush=True)


def main():
    args = parse_args()
    config = make_config(args.mode, args.run_name, args.run_dir, **overrides(args))
    check_gpu(args.mode)
    use_colab_auth_credentials()
    check_drive_access()
    config_path = write_config(config)
    print(json.dumps(config, indent=1), flush=True)
    for script in STAGES:
        run_stage(script, config_path, config)
    print("RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
