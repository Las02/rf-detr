"""Visualise colony augmentations on a real training image.

Picks one random image from dataset_dir/train/, applies the augmentation
pipeline --copies times, and saves a grid montage to --output.
Bounding boxes from the COCO annotation are drawn on each tile.

Usage:
    uv run --with typer python train/test_aug.py --dataset-dir /path/to/dataset
    uv run --with typer python train/test_aug.py --dataset-dir /path/to/dataset --copies 12
"""

import json
import random
from pathlib import Path

import albumentations as A
import numpy as np
import typer
from PIL import Image, ImageDraw

from simple_train import AUG_COLONY

app = typer.Typer(add_completion=False)

COLS = 4  # images per row in the grid
BOX_COLOR = (255, 80, 80)
BOX_WIDTH = 2


def _load_bboxes(ann_file: Path, image_name: str) -> list[tuple[float, float, float, float]]:
    """Return normalised (x_min, y_min, x_max, y_max) boxes for the given image."""
    data = json.loads(ann_file.read_text())
    img_id = next(
        (img["id"] for img in data["images"] if img["file_name"] == image_name),
        None,
    )
    if img_id is None:
        return []
    img_meta = next(img for img in data["images"] if img["id"] == img_id)
    iw, ih = img_meta["width"], img_meta["height"]
    boxes = []
    for ann in data["annotations"]:
        if ann["image_id"] != img_id:
            continue
        x, y, w, h = ann["bbox"]
        boxes.append((x / iw, y / ih, (x + w) / iw, (y + h) / ih))
    return boxes


def _draw_boxes(image: np.ndarray, bboxes: list[tuple]) -> np.ndarray:
    """Draw normalised bboxes onto an HWC numpy image."""
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    h, w = image.shape[:2]
    for x_min, y_min, x_max, y_max in bboxes:
        draw.rectangle(
            [x_min * w, y_min * h, x_max * w, y_max * h],
            outline=BOX_COLOR,
            width=BOX_WIDTH,
        )
    return np.array(pil)


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
    dataset_dir: Path = typer.Option(...,               help="Root of the Roboflow COCO dataset"),
    copies:      int  = typer.Option(8,                 help="Number of augmented copies to generate"),
    output:      Path = typer.Option("aug_preview.png", help="Output image path"),
) -> None:
    """Generate an augmentation preview grid from one random training image."""
    train_dir = dataset_dir / "train"
    ann_file = train_dir / "_annotations.coco.json"
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
    bboxes = _load_bboxes(ann_file, img_path.name) if ann_file.exists() else []
    typer.echo(f"  {len(bboxes)} annotations")

    pipeline = A.Compose(
        [getattr(A, name)(**params) for name, params in AUG_COLONY.items()],
        bbox_params=A.BboxParams(format="albumentations", label_fields=["labels"], min_visibility=0.1),
    )

    tiles = [_draw_boxes(original, bboxes)]
    for _ in range(copies):
        result = pipeline(image=original, bboxes=bboxes, labels=[0] * len(bboxes))
        tiles.append(_draw_boxes(result["image"], result["bboxes"]))

    grid = _build_grid(tiles, cols=COLS)
    grid.save(output)
    typer.echo(f"Saved {len(tiles)}-tile preview ({COLS} per row) → {output}")


if __name__ == "__main__":
    app()
