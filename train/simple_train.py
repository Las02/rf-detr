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

import random
from pathlib import Path

import numpy as np
import typer
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRLarge

app = typer.Typer(add_completion=False)

_FONT_PATH = Path(__file__).parent / "HomemadeApple-Regular.ttf"

# Realistic lab annotation fragments written on petri dish lids
_HANDWRITTEN_TEXTS = [
    # Dilution series (most common annotation on petri dishes)
    "10-1", "10-2", "10-3", "10-4", "10-5", "10-6",
    "1:10", "1:100", "1:1000",
    # Controls
    "neg", "pos", "ctrl", "blank", "wt",
    # Selective media / antibiotic conditions
    "+Amp", "-Amp", "+Kan", "Cm", "Tet",
    # Organism shorthands
    "E.coli", "S.aur", "B.sub", "K.pneu",
    # Colony count results
    "TNTC", ">300", "<10", "0",
    # Sample / plate IDs
    "S1", "S2", "S3", "P1", "P2",
    "A1", "A2", "B1", "B2",
    # Initials + date fragments
    "LM", "JK", "TK", "AH",
    "12/3", "03-24", "Jan15",
    # Incubation / experiment notes
    "37C", "30C", "RT", "ON",
    "Rep1", "Rep2", "n=3",
]


def _handwritten_text(image: np.ndarray, **kwargs) -> np.ndarray:
    """Draw 1–3 handwritten-style annotations at random positions."""
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    h, w = image.shape[:2]

    for _ in range(random.randint(1, 3)):
        text = random.choice(_HANDWRITTEN_TEXTS)
        size = random.randint(max(14, h // 25), max(28, h // 12))
        try:
            font = ImageFont.truetype(str(_FONT_PATH), size)
        except OSError:
            font = ImageFont.load_default()

        # Dark ink (blue-black) most of the time; occasionally light for contrast
        if random.random() < 0.75:
            color = (random.randint(0, 50), random.randint(0, 50), random.randint(40, 100))
        else:
            color = (random.randint(180, 255),) * 3

        # Render text onto a transparent scratch canvas, rotate, then position
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        scratch = Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
        ImageDraw.Draw(scratch).text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=(*color, 255))
        scratch = scratch.rotate(random.uniform(-45, 45), expand=True)

        # Pick position after rotation so the full rotated stamp fits in the image
        px = random.randint(0, max(0, w - scratch.width))
        py = random.randint(0, max(0, h - scratch.height))
        pil.paste(scratch, (px, py), mask=scratch)

    return np.array(pil)

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
    "Defocus":        {"p": 0.05},
    "RandomSunFlare": {"p":0.05},
    "RandomShadow": {"p":0.05},
    "Lambda": {"image": _handwritten_text, "p": 0.2},
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
