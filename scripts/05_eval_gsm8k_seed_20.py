import argparse
import json
import os
import re
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEFAULT_LOCAL_MODEL_PATH = "checkpoints/Qwen2.5-1.5B-Instruct"
DEFAULT_REMOTE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MODEL_NAME = os.environ.get(
    "MODEL_PATH",
    DEFAULT_LOCAL_MODEL_PATH
    if os.path.isdir(DEFAULT_LOCAL_MODEL_PATH)
    else DEFAULT_REMOTE_MODEL_NAME,
)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_number(text):
    text = text.replace(",", "")
    if "####" in text:
        tail = text.split("####")[-1]
        nums = re.findall(r"-?\d+\.?\d*", tail)
        if nums:
            return nums[-1]
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def build_prompt(instruction):
    format_instruction = (
        "\n\nImportant formatting requirement:\n"
        "The final line of your response must be exactly '#### <number>'.\n"
        "Only put the final numeric answer after ####.\n"
        "Do not add units, explanations, Markdown, or LaTeX after ####."
    )
    messages = [{"role": "user", "content": instruction + format_instruction}]
    return messages


def iter_batches(data, batch_size):
    for start in range(0, len(data), batch_size):
        yield data[start:start + batch_size]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/gsm8k_eval_200.jsonl")
    parser.add_argument("--output-path", default="outputs/seed_outputs/gsm8k_seed_20.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    eval_data = load_jsonl(args.data_path)[:args.limit]
    if not eval_data:
        raise ValueError("No evaluation data loaded; check --data-path and --limit")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    correct = 0

    progress = tqdm(
        iter_batches(eval_data, args.batch_size),
        total=(len(eval_data) + args.batch_size - 1) // args.batch_size,
    )
    for batch in progress:
        texts = [
            tokenizer.apply_chat_template(
                build_prompt(item["instruction"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        for item, response in zip(batch, responses):
            pred = extract_number(response)
            gold = str(item["final_answer"]).replace(",", "").strip()
            is_correct = pred == gold

            if is_correct:
                correct += 1

            results.append({
                "id": item["id"],
                "question": item["question"],
                "gold": gold,
                "prediction": pred,
                "is_correct": is_correct,
                "response": response,
            })

    acc = correct / len(eval_data)
    print(f"Seed GSM8K-20 Accuracy: {acc:.4f} ({correct}/{len(eval_data)})")

    save_jsonl(results, args.output_path)


if __name__ == "__main__":
    main()
