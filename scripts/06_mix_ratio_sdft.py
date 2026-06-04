import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path


DEFAULT_RATIOS = "0,0.25,0.5,0.75,1.0"


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_ratios(text):
    ratios = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ratio = float(part)
        if ratio < 0 or ratio > 1:
            raise ValueError(f"Mix ratio must be in [0, 1], got {ratio}")
        ratios.append(ratio)
    if not ratios:
        raise ValueError("At least one mix ratio is required")
    return ratios


def ratio_tag(ratio):
    return f"{ratio:.2f}".replace(".", "p")


def response_is_changed(item, original_field, mixed_field):
    return item.get(mixed_field, "") != item.get(original_field, "")


def build_mixed_dataset(
    data,
    ratio,
    seed,
    original_field,
    sdft_field,
    response_field,
    selection_mode,
):
    n_items = len(data)
    n_sdft = round(n_items * ratio)

    indices = list(range(n_items))
    if selection_mode == "random":
        rng = random.Random(seed)
        rng.shuffle(indices)

    sdft_indices = set(indices[:n_sdft])
    mixed = []
    n_selected_sdft = 0
    n_verified_sdft = 0
    n_changed_response = 0

    for idx, item in enumerate(data):
        use_sdft = idx in sdft_indices
        source_field = sdft_field if use_sdft else original_field

        if source_field not in item:
            raise KeyError(
                f"Missing field '{source_field}' in item {item.get('id', idx)}"
            )

        output_item = dict(item)
        output_item[response_field] = item[source_field]
        output_item["mix_ratio"] = ratio
        output_item["mix_source"] = "sdft" if use_sdft else "original"
        output_item["mix_source_field"] = source_field
        mixed.append(output_item)

        if use_sdft:
            n_selected_sdft += 1
            n_verified_sdft += int(bool(item.get("is_distilled_used")))
        n_changed_response += int(response_is_changed(output_item, original_field, response_field))

    stats = {
        "ratio": ratio,
        "total": n_items,
        "selected_sdft": n_selected_sdft,
        "selected_original": n_items - n_selected_sdft,
        "verified_sdft_selected": n_verified_sdft,
        "changed_response": n_changed_response,
        "actual_selected_sdft_ratio": n_selected_sdft / n_items if n_items else 0.0,
        "actual_verified_sdft_ratio": n_verified_sdft / n_items if n_items else 0.0,
        "actual_changed_response_ratio": n_changed_response / n_items if n_items else 0.0,
    }
    return mixed, stats


def run_command(cmd, dry_run):
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_train_cmd(args, train_file, output_dir):
    return [
        sys.executable,
        "scripts/04_train_qlora.py",
        "--model_name",
        args.model_name,
        "--train_file",
        str(train_file),
        "--response_field",
        args.response_field,
        "--output_dir",
        str(output_dir),
        "--max_seq_length",
        str(args.max_seq_length),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--logging_steps",
        str(args.logging_steps),
        "--save_steps",
        str(args.save_steps),
    ]


def build_eval_cmd(args, adapter_dir, output_file):
    return [
        sys.executable,
        "scripts/05_eval_gsm8k.py",
        "--model_name",
        args.model_name,
        "--adapter_path",
        str(adapter_dir),
        "--eval_file",
        args.eval_file,
        "--output_file",
        str(output_file),
        "--limit",
        str(args.eval_limit),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--batch_size",
        str(args.batch_size),
    ]


def summarize_eval(output_file):
    if not Path(output_file).exists():
        return None

    rows = load_jsonl(output_file)
    if not rows:
        return {"eval_total": 0, "eval_correct": 0, "eval_accuracy": 0.0}

    correct = sum(1 for row in rows if row.get("is_correct"))
    return {
        "eval_total": len(rows),
        "eval_correct": correct,
        "eval_accuracy": correct / len(rows),
    }


def write_summary(rows, csv_path, json_path):
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ratio",
        "train_file",
        "adapter_dir",
        "eval_output_file",
        "total",
        "selected_sdft",
        "selected_original",
        "verified_sdft_selected",
        "changed_response",
        "actual_selected_sdft_ratio",
        "actual_verified_sdft_ratio",
        "actual_changed_response_ratio",
        "eval_total",
        "eval_correct",
        "eval_accuracy",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build MixRatio ablation datasets for SDFT and optionally train/evaluate "
            "one LoRA adapter per ratio."
        )
    )
    parser.add_argument(
        "--distilled_file",
        default="data/distilled/gsm8k_train_500_sdft_v2.jsonl",
        help="JSONL file containing original_response and SDFT responses.",
    )
    parser.add_argument(
        "--ratios",
        default=DEFAULT_RATIOS,
        help="Comma-separated SDFT mix ratios, for example: 0,0.25,0.5,0.75,1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection_mode",
        choices=["random", "prefix"],
        default="random",
        help="How to choose examples that use the SDFT response for each ratio.",
    )
    parser.add_argument("--original_field", default="original_response")
    parser.add_argument(
        "--sdft_field",
        default="final_response",
        help="Use final_response to keep SDFT's verified fallback mechanism.",
    )
    parser.add_argument(
        "--response_field",
        default="mixed_response",
        help="Field written into each mixed JSONL and passed to the trainer.",
    )
    parser.add_argument("--data_dir", default="data/mix_ratio")
    parser.add_argument("--output_dir", default="checkpoints/mix_ratio")
    parser.add_argument("--eval_output_dir", default="outputs/mix_ratio")
    parser.add_argument("--summary_csv", default="outputs/mix_ratio/summary.csv")
    parser.add_argument("--summary_json", default="outputs/mix_ratio/summary.json")

    parser.add_argument("--run_train", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print train/eval commands without executing them.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip train/eval steps whose expected output already exists.",
    )

    parser.add_argument("--model_name", default="checkpoints/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)

    parser.add_argument("--eval_file", default="data/processed/gsm8k_eval_200.jsonl")
    parser.add_argument("--eval_limit", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_jsonl(args.distilled_file)
    if not data:
        raise ValueError(f"No data loaded from {args.distilled_file}")

    ratios = parse_ratios(args.ratios)
    rows = []

    for ratio in ratios:
        tag = ratio_tag(ratio)
        train_file = Path(args.data_dir) / f"gsm8k_train_mix_{tag}.jsonl"
        adapter_dir = Path(args.output_dir) / f"ratio_{tag}"
        eval_output_file = Path(args.eval_output_dir) / f"gsm8k_eval_ratio_{tag}.jsonl"

        mixed, stats = build_mixed_dataset(
            data=data,
            ratio=ratio,
            seed=args.seed,
            original_field=args.original_field,
            sdft_field=args.sdft_field,
            response_field=args.response_field,
            selection_mode=args.selection_mode,
        )
        save_jsonl(mixed, train_file)

        row = {
            **stats,
            "train_file": str(train_file),
            "adapter_dir": str(adapter_dir),
            "eval_output_file": str(eval_output_file),
        }

        print(
            f"[ratio={ratio:.2f}] saved {train_file} | "
            f"SDFT selected {stats['selected_sdft']}/{stats['total']} | "
            f"verified {stats['verified_sdft_selected']}/{stats['total']} | "
            f"changed {stats['changed_response']}/{stats['total']}"
        )

        if args.run_train:
            marker = adapter_dir / "adapter_model.safetensors"
            if args.skip_existing and marker.exists():
                print(f"Skip training because {marker} exists")
            else:
                run_command(build_train_cmd(args, train_file, adapter_dir), args.dry_run)

        if args.run_eval:
            if args.skip_existing and eval_output_file.exists():
                print(f"Skip eval because {eval_output_file} exists")
            else:
                run_command(build_eval_cmd(args, adapter_dir, eval_output_file), args.dry_run)

            eval_summary = summarize_eval(eval_output_file)
            if eval_summary is not None:
                row.update(eval_summary)

        rows.append(row)

    write_summary(rows, args.summary_csv, args.summary_json)
    print(f"Summary CSV: {args.summary_csv}")
    print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
