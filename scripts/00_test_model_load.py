import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


DEFAULT_LOCAL_MODEL_PATH = "checkpoints/Qwen2.5-1.5B-Instruct"
DEFAULT_REMOTE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MODEL_NAME = os.environ.get(
    "MODEL_PATH",
    DEFAULT_LOCAL_MODEL_PATH
    if os.path.isdir(DEFAULT_LOCAL_MODEL_PATH)
    else DEFAULT_REMOTE_MODEL_NAME,
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except OSError as exc:
        if (
            "does not appear to have a file named" in str(exc)
            or "Can't load the model" in str(exc)
        ):
            raise RuntimeError(
                "Model weights are missing or the Hugging Face cache is incomplete. "
                "Download the model first, for example:\n"
                f"huggingface-cli download {DEFAULT_REMOTE_MODEL_NAME} "
                f"--local-dir {DEFAULT_LOCAL_MODEL_PATH} "
                "--local-dir-use-symlinks False\n"
                "Then run:\n"
                f"MODEL_PATH={DEFAULT_LOCAL_MODEL_PATH} "
                "python scripts/00_test_model_load.py"
            ) from exc
        raise

    prompt = "Solve the following math problem step by step. Problem: If Tom has 3 apples and buys 4 more, how many apples does he have?"

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()
