## 06_mix_ratio_sdft.py

Build MixRatio ablation data for checking the SDFT mechanism. The ratio is the
fraction of training examples whose training target comes from the SDFT field
(`final_response` by default); the rest use `original_response`.

Generate mixed datasets only:

```bash
python scripts/06_mix_ratio_sdft.py --ratios 0,0.25,0.5,0.75,1
```

Run the full train/eval sweep:

```bash
python scripts/06_mix_ratio_sdft.py \
  --ratios 0,0.25,0.5,0.75,1 \
  --run_train \
  --run_eval
```

Outputs:

```text
data/mix_ratio/gsm8k_train_mix_*.jsonl
checkpoints/mix_ratio/ratio_*/
outputs/mix_ratio/gsm8k_eval_ratio_*.jsonl
outputs/mix_ratio/summary.csv
outputs/mix_ratio/summary.json
```
