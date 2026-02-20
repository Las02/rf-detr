"""Visualise colony augmentations on a real training image.

Picks one random image from dataset_dir/train/, applies the augmentation
pipeline --copies times, and saves a grid montage to --output.

Usage:
    uv run --with typer python train/test_aug.py --dataset-dir /path/to/dataset
    uv run --with typer python train/test_aug.py --dataset-dir /path/to/dataset --extra --copies 12
"""

import random
from pathlib import Path

import albumentations as A
import numpy as np
import typer
from PIL import Image

from simple_train import AUG_COLONY, AUG_COLONY_EXTRA

app = typer.Typer(add_completion=False)

COLS = 4  # images per row in the grid


def _build_grid(images: list[np.ndarray], cols: int) -> Image.Image:
    """Arrange a list of HWC numpy images into a grid."""
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (w * cols, h * rows), color=(30, 30, 30))
    for i, img in enumerate(images):
        col = i % cols
        row = i // cols
        grid.paste(Image.fromarray(img), (col * w, row * h))
    return grid


@app.command()
def test_aug(
    dataset_dir: Path = typer.Option(...,                  help="Root of the Roboflow COCO dataset"),
    copies:      int  = typer.Option(8,                    help="Number of augmented copies to generate"),
    extra:       bool = typer.Option(False,                help="Include Perspective, Defocus, RandomSunFlare"),
    output:      Path = typer.Option("aug_preview.png",    help="Output image path"),
) -> None:
    """Generate an augmentation preview grid from one random training image."""
    train_dir = dataset_dir / "train"
    image_files = sorted(
        p for p in train_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_files:
        typer.echo(f"No images found in {train_dir}", err=True)
        raise typer.Exit(1)

    img_path = random.choice(image_files)
    typer.echo(f"Source image: {img_path.name}")

    original = np.array(Image.open(img_path).convert("RGB"))

    aug_cfg = {**AUG_COLONY, **(AUG_COLONY_EXTRA if extra else {})}
    pipeline = A.Compose([getattr(A, name)(**params) for name, params in aug_cfg.items()])

    tiles = [original]
    for _ in range(copies):
        tiles.append(pipeline(image=original)["image"])

    grid = _build_grid(tiles, cols=COLS)
    grid.save(output)
    typer.echo(f"Saved {len(tiles)}-tile preview ({COLS} per row) → {output}")


if __name__ == "__main__":
    app()
