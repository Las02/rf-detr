# ------------------------------------------------------------------------
# Colony Counter — Simple RF-DETR Medium Training Script
# Uses rfdetr.train() with an aug_config dict — no custom training loop.
# ------------------------------------------------------------------------

from pathlib import Path

import typer
from rfdetr.detr import RFDETRMedium

import colony_transforms  # noqa: F401 — registers A.SahiStyleCrop and exports handwritten_text
from colony_transforms import handwritten_text

app = typer.Typer(add_completion=False)

AUG_COLONY: dict = {
    "LongestMaxSize":           {"max_size": 2024},
    "PadIfNeeded":              {"min_height": 2024, "min_width": 2024, "border_mode": 0, "p": 1.0},
    "HorizontalFlip":           {"p": 0.5},
    "VerticalFlip":             {"p": 0.5},
    "Rotate":                   {"limit": 180, "border_mode": 0, "value": 0, "p": 0.5},
    "Lambda":                   {"image": handwritten_text, "p": 0.3},
    "SahiStyleCrop":            {"sizes": (1124, 778), "weights": (0.6, 0.4), "p": 0.8},
    "RandomBrightnessContrast": {"p": 0.3},
    "HueSaturationValue":       {"p": 0.3},
    "ISONoise":                 {"p": 0.3},
    "ImageCompression":         {"p": 0.3},
    "ChannelDropout":           {"p": 0.3},
    "Defocus":                  {"p": 0.05},
    "RandomSunFlare":           {"p": 0.05},
    "RandomShadow":             {"p": 0.05},
}


@app.command()
def train(
    dataset_dir: Path = typer.Option(...,              help="Root of the Roboflow COCO dataset"),
    output_dir:  Path = typer.Option("output/colony",  help="Where to save checkpoints"),
    epochs:      int  = typer.Option(100,              help="Total training epochs"),
    batch_size:  int  = typer.Option(4,                help="Per-GPU batch size"),
    lr:          float = typer.Option(1e-4,            help="Base learning rate"),
    num_workers: int  = typer.Option(2,                help="DataLoader worker count"),
    device:      str  = typer.Option("auto",           help="Device: auto | cuda | mps | cpu"),
    resume:      str  = typer.Option("",              help="Path to checkpoint to resume from"),
    project:     str  = typer.Option("colony-counter", help="W&B project name"),
    run:         str  = typer.Option("",              help="W&B run name (defaults to auto)"),
) -> None:
    """Train RFDETRMedium on a colony-counting COCO dataset with online augmentation."""
    aug_config = AUG_COLONY

    rfdetr = RFDETRMedium()
    rfdetr.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        # lr_encoder=lr * 1.5,
        num_workers=num_workers,
        device=device,
        num_select=300,   # TrainConfig default is 300, but explicit for clarity
        num_queries=300,  # not a TrainConfig field — flows via **kwargs → populate_args
        aug_config=aug_config,
        resume=resume or None,
        wandb=True,
        project=project,
        run=run or None,
    )


if __name__ == "__main__":
    app()
