# ------------------------------------------------------------------------
# Colony Counter — Simple RF-DETR Large Training Script
# Uses rfdetr.train() with an aug_config dict — no custom training loop.
# ------------------------------------------------------------------------
"""Train an RFDETRLarge model on a colony-counting COCO dataset.

Augmentation pipeline:
  - HorizontalFlip, VerticalFlip, Rotate(±180°)
  - RandomBrightnessContrast, HueSaturationValue, ISONoise
  - ImageCompression, ChannelDropout
  --extra mode additionally enables:
  - Perspective, Defocus, RandomSunFlare
"""

from pathlib import Path

import typer
from rfdetr import RFDETRLarge

app = typer.Typer(add_completion=False)

import cv2
import numpy as np

def add_colony_metadata(image, **kwargs):
    # Copy to avoid modifying the original array in-place
    img = image.copy()
    text = "Colony-Scanner-v1"
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Place text at bottom-left
    cv2.putText(img, text, (10, img.shape[0] - 10), font, 0.6, (255, 255, 255), 1)
    return img

AUG_COLONY: dict = {
    "HorizontalFlip":           {"p": 0.5},
    "VerticalFlip":             {"p": 0.5},
    "Rotate":                   {"limit": 180, "border_mode":0, "value": 0, "p": 0.5},
    "RandomBrightnessContrast": {"p": 0.3},
    "HueSaturationValue":       {"p": 0.3},
    "ISONoise":                 {"p": 0.3},
    "ImageCompression":         {"p": 0.3},
    "ChannelDropout":           {"p": 0.3},
}

AUG_COLONY_EXTRA: dict = {
    "Defocus":        {"p": 0.8},
    "RandomSunFlare": {"p":0.05},
    "RandomShadow": {"p":0.05},
    "Lambda": {"image": add_colony_metadata, "p": 1.0},
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
    extra:       bool = typer.Option(False,            help="Add Perspective, Defocus, RandomSunFlare"),
    resume:      str  = typer.Option("",              help="Path to checkpoint to resume from"),
) -> None:
    """Train RFDETRLarge on a colony-counting COCO dataset with online augmentation."""
    aug_config = {**AUG_COLONY, **(AUG_COLONY_EXTRA if extra else {})}

    rfdetr = RFDETRLarge()
    rfdetr.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lr_encoder=lr * 1.5,
        num_workers=num_workers,
        device=device,
        num_select=300,   # TrainConfig default is 300, but explicit for clarity
        num_queries=300,  # not a TrainConfig field — flows via **kwargs → populate_args
        aug_config=aug_config,
        resume=resume or None,
    )


if __name__ == "__main__":
    app()
