# train/ — Custom Colony Training Scripts

Custom training scripts that use RF-DETR internals directly,
bypassing `model.train()` to inject domain-specific augmentations online.

## Scripts

### `simple_train.py`

Trains an `RFDETRMedium` model using `rfdetr.train()` with an `aug_config` dict.
Simpler than `train_colony.py` — no custom training loop.

W&B logging is always enabled. Pass `--project` and `--run` to configure.

**Augmentation pipeline (`AUG_COLONY`):**

| Transform | Probability | Notes |
|-----------|------------|-------|
| LongestMaxSize | 1.0 | Scales down to max 2024px; small images unchanged |
| PadIfNeeded | 1.0 | Pads to 2024×2024 with black border |
| HorizontalFlip | 0.5 | |
| VerticalFlip | 0.5 | |
| Rotate ±180° | 0.5 | BORDER_CONSTANT fill |
| RandomSizedCrop | 0.8 | Crops 778–1124px from 2024px base — simulates 4-slice (1124px) to 9-slice (778px) SAHI with 20% overlap |
| RandomBrightnessContrast | 0.3 | |
| HueSaturationValue | 0.3 | |
| ISONoise | 0.3 | |
| ImageCompression | 0.3 | |
| ChannelDropout | 0.3 | |

**Extra augmentations (`AUG_COLONY_EXTRA`, `--extra` flag):**

| Transform | Probability | Notes |
|-----------|------------|-------|
| Defocus | 0.05 | |
| RandomSunFlare | 0.05 | |
| RandomShadow | 0.05 | |
| Lambda (_handwritten_text) | 0.3 | Draws realistic lab annotations on the image |

**SAHI crop rationale:**
Images are normalised to 2024×2024 before cropping. Crop sizes derived from
`slice_size = 2024 / (1 + (n-1) × 0.8)`:
- 4 slices → 1124px, 9 slices → 778px. `RandomSizedCrop` samples continuously
between these, covering both inference modes.

**Usage:**

```bash
uv run --with typer python train/simple_train.py \
    --dataset-dir /path/to/dataset \
    --project colony-counter \
    --run my-run

# With extra augmentations
uv run --with typer python train/simple_train.py \
    --dataset-dir /path/to/dataset \
    --extra

# Resume from checkpoint
uv run --with typer python train/simple_train.py \
    --dataset-dir /path/to/dataset \
    --resume output/colony/checkpoint.pth
```

---

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
