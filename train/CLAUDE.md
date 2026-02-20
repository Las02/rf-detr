# train/ — Custom Colony Training Scripts

Custom training scripts that use RF-DETR internals directly,
bypassing `model.train()` to inject domain-specific augmentations online.

## Scripts

### `train_colony.py`

Trains an `RFDETRLarge` model on a colony-counting (petri dish) COCO dataset with an
augmentation pipeline equivalent to `augment_data/augment.py` applied **online** per batch.

**Augmentation pipeline:**

| Transform | Probability | Notes |
|-----------|------------|-------|
| HorizontalFlip | 0.5 | |
| VerticalFlip | 0.5 | |
| Rotate ±180° | 0.75 | BORDER_CONSTANT fill |
| RandomBrightnessContrast | 0.3 | |
| HueSaturationValue | 0.3 | |
| ISONoise | 0.3 | |
| ImageCompression | 0.3 | |
| ChannelDropout | 0.3 | |
| *Perspective* | 0.05 | `--extra` only |
| *Defocus* | 0.1 | `--extra` only |
| *EdgeFlareTransform* | 0.1 | `--extra` only — A.OneOf of 4 edge-positioned sun flares |
| *ReflectColoniesTransform* | 0.5 | `--extra` only — reads boxes, adds unlabeled pixel-level distractors |

**Usage:**

```bash
# Standard run
uv run --with typer python train/train_colony.py \
    --dataset-dir /path/to/dataset \
    --output-dir output/colony \
    --epochs 100 \
    --batch-size 4 \
    --resolution 704

# With extra augmentations (reflect_colonies, edge_flare, perspective, defocus)
uv run --with typer python train/train_colony.py \
    --dataset-dir /path/to/dataset \
    --extra

# Resume from checkpoint
uv run --with typer python train/train_colony.py \
    --dataset-dir /path/to/dataset \
    --resume output/colony/checkpoint.pth
```

**Dataset format (Roboflow COCO):**

```
dataset_dir/
    train/
        _annotations.coco.json
        image1.jpg ...
    valid/
        _annotations.coco.json
        image1.jpg ...
    test/           ← optional; evaluation runs if present
        _annotations.coco.json
        image1.jpg ...
```

**Outputs:**
- `output/colony/checkpoint.pth` — checkpoint saved every epoch
- `output/colony/best_model.pth` — checkpoint with highest val mAP

## Dependencies

`typer` is required but not in the main `pyproject.toml` (it's a script-level dependency).
Install it with:

```bash
uv add typer           # adds to pyproject.toml
# or for a one-off run:
uv run --with typer python train/train_colony.py ...
```

## Key internals used

| Symbol | Source |
|--------|--------|
| `RFDETRLarge` | `rfdetr` — downloads & loads pretrained weights |
| `populate_args` | `rfdetr.main` — builds the args namespace |
| `build_criterion_and_postprocessors` | `rfdetr.models` |
| `get_param_dict` | `rfdetr.util.get_param_dicts` — layer-wise LR groups |
| `train_one_epoch`, `evaluate` | `rfdetr.engine` |
| `CocoDetection` | `rfdetr.datasets.coco` |
| `AlbumentationsWrapper`, `ComposeAugmentations` | `rfdetr.datasets.transforms` |
| `ModelEma` | `rfdetr.util.utils` |
