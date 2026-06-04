import argparse
import json
import re
from pathlib import Path
from decimal import Decimal, InvalidOperation

import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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


def normalize_number(x):
    if x is None:
        return ""
    x = str(x).replace(",", "").replace("$", "").strip()
    x = x.rstrip(".")
    try:
        d = Decimal(x)
        if d == d.to_integral():
            return str(int(d))
        return str(d.normalize())
    except InvalidOperation:
        return x


def extract_number(text):
    text = text.replace(",", "")

    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return normalize_number(m.group(1))

    m = re.search(r"\\boxed\{?\s*(-?\d+(?:\.\d+)?)\s*\}?", text)
    if m:
        return normalize_number(m.group(1))

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return normalize_number(nums[-1]) if nums else ""


def build_prompt(tokenizer, instruction):
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def iter_batches(data, batch_size):
    for start in range(0, len(data), batch_size):
        yield data[start:start + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="checkpoints/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default="data/processed/gsm8k_eval_200.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    eval_data = load_jsonl(args.eval_file)[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter_path is not None:
        print(f"Loading LoRA adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    results = []
    correct = 0

    progress = tqdm(
        iter_batches(eval_data, args.batch_size),
        total=(len(eval_data) + args.batch_size - 1) // args.batch_size,
    )
    for batch in progress:
        prompt_texts = [
            build_prompt(tokenizer, item["instruction"])
            for item in batch
        ]
        inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        for item, response in zip(batch, responses):
            response = response.strip()
            pred = extract_number(response)
            gold = normalize_number(item["final_answer"])
            is_correct = pred == gold

            correct += int(is_correct)

            results.append({
                "id": item["id"],
                "question": item["question"],
                "gold": gold,
                "prediction": pred,
                "is_correct": is_correct,
                "response": response,
            })

    acc = correct / len(eval_data)

    save_jsonl(results, args.output_file)

    print("=" * 80)
    print(f"Evaluated examples: {len(eval_data)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved to: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
