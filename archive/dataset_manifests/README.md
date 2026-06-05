# Dataset Manifests (archived)

These 8 `.txt` files each list training-sample file paths for one class:

- `violent.txt` — violent clips
- `neutral.txt` — neutral clips
- `carrying.txt`, `cupping.txt`, `grasping.txt`, `gripping.txt`, `holding.txt`, `resting.txt` — hand-gesture / pose classes

## Why archived, not deleted

They were originally consumed by the now-removed `train_model.py` (the old minimal LSTM trainer). The current trainer (`train_model_enhanced.py`) reads samples from a `--data-dir` argument with validation instead, so these manifests are no longer wired in.

Kept here in case the curation work that produced them is still valuable, or in case you want to regenerate / port to a new format later.

## To restore one (example)

```bash
git mv archive/dataset_manifests/violent.txt ./violent.txt
```
