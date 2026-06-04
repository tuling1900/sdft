## Environment configuration

This project uses Python 3.10 and Hugging Face Transformers to run
Qwen2.5-1.5B-Instruct on GSM8K. The model loading scripts use
bitsandbytes 4-bit quantization, so an NVIDIA GPU with CUDA is recommended.

### 1. Create and activate the conda environment

```bash
conda create -n sdft python=3.10 -y
conda activate sdft
```

### 2. Install PyTorch

For an NVIDIA GPU with CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only environments, data preparation can run normally, but the current
model loading and evaluation scripts may need code changes because they use
bitsandbytes 4-bit quantization:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If your CUDA version is different, install the matching PyTorch wheel from
https://pytorch.org/get-started/locally/.

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the model

The scripts use the local checkpoint below by default when it exists:

```text
checkpoints/Qwen2.5-1.5B-Instruct
```

If the checkpoint is missing or incomplete, download it with:

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir checkpoints/Qwen2.5-1.5B-Instruct \
  --local-dir-use-symlinks False
```

You can also point the scripts at another local model path:

```bash
export MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
```

### 5. Prepare GSM8K data

```bash
python scripts/01_prepare_gsm8k.py
```

This creates:

```text
data/processed/gsm8k_train_500.jsonl
data/processed/gsm8k_eval_200.jsonl
```

### 6. Verify the environment

Test model loading and generation:

```bash
python scripts/00_test_model_load.py
```

Run the GSM8K seed evaluation:

```bash
python scripts/05_eval_gsm8k_seed_20.py
```
