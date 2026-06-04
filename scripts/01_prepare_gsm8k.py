import json
import re
from datasets import load_dataset
from pathlib import Path


def extract_final_answer(answer: str) -> str:
    """
    GSM8K answer usually contains '#### final_answer'.
    """
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    
    numbers = re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))
    return numbers[-1] if numbers else ""


def save_jsonl(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    dataset = load_dataset("openai/gsm8k", "main")

    train_raw = dataset["train"]
    test_raw = dataset["test"]

    train_samples = []
    for i, item in enumerate(train_raw.select(range(500))):
        question = item["question"]
        answer = item["answer"]
        train_samples.append({
            "id": f"gsm8k_train_{i:06d}",
            "question": question,
            "instruction": (
                "Solve the following math problem step by step. "
                "Give the final answer after ####.\n\n"
                f"Problem: {question}"
            ),
            "original_response": answer,
            "final_answer": extract_final_answer(answer)
        })

    eval_samples = []
    for i, item in enumerate(test_raw.select(range(200))):
        question = item["question"]
        answer = item["answer"]
        eval_samples.append({
            "id": f"gsm8k_eval_{i:06d}",
            "question": question,
            "instruction": (
                "Solve the following math problem step by step. "
                "Give the final answer after ####.\n\n"
                f"Problem: {question}"
            ),
            "original_response": answer,
            "final_answer": extract_final_answer(answer)
        })

    save_jsonl(train_samples, "data/processed/gsm8k_train_500.jsonl")
    save_jsonl(eval_samples, "data/processed/gsm8k_eval_200.jsonl")

    print("Saved:")
    print("data/processed/gsm8k_train_500.jsonl")
    print("data/processed/gsm8k_eval_200.jsonl")
    print("Example:")
    print(json.dumps(train_samples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
