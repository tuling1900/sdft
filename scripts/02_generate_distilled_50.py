import json
import os
import re
import torch
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_NAME = os.environ.get("MODEL_PATH", "checkpoints/Qwen2.5-1.5B-Instruct")


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


def is_same_answer(pred, gold):
    return normalize_number(pred) == normalize_number(gold)


def build_distill_prompt(question, original_response, gold_answer):
    return f"""Rewrite the reference solution in a concise GSM8K style.

Important rules:
1. Do not solve the problem independently.
2. Preserve the exact reasoning and arithmetic from the reference solution.
3. Do not use Markdown headings, bullet points, LaTeX display equations, or boxed answers.
4. Keep the rewritten solution short, no more than 6 lines.
5. The final line must be exactly:
#### {gold_answer}
6. Do not write anything after the final line.

Problem:
{question}

Reference solution:
{original_response}

Rewritten solution:
"""


def main():
    train_data = load_jsonl("data/processed/gsm8k_train_500.jsonl")[:50]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

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
    accepted = 0
    fallback = 0

    for item in tqdm(train_data):
        prompt = build_distill_prompt(
            question=item["question"],
            original_response=item["original_response"],
            gold_answer=item["final_answer"],
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        distilled_response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        gold_answer = str(item["final_answer"]).replace(",", "").strip()
        distilled_answer = extract_number(distilled_response)

        if is_same_answer(distilled_answer, gold_answer):
            final_response = distilled_response
            is_distilled_used = True
            accepted += 1
        else:
            final_response = item["original_response"]
            is_distilled_used = False
            fallback += 1

        results.append({
            "id": item["id"],
            "question": item["question"],
            "instruction": item["instruction"],
            "original_response": item["original_response"],
            "distilled_response": distilled_response,
            "final_response": final_response,
            "gold_answer": gold_answer,
            "distilled_answer": distilled_answer,
            "is_distilled_used": is_distilled_used,
        })

    save_jsonl(results, "data/distilled/gsm8k_train_50_sdft_v2.jsonl")

    print("Saved to data/distilled/gsm8k_train_50_sdft_v2.jsonl")
    print(f"Accepted: {accepted}")
    print(f"Fallback: {fallback}")
    print(f"Acceptance rate: {accepted / len(results):.4f}")


if __name__ == "__main__":
    main()
