"""Custom albumentations transforms for colony counter training."""

import random
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.core.transforms_interface import DualTransform
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# SahiStyleCrop
# ---------------------------------------------------------------------------

class SahiStyleCrop(DualTransform):
    """Crop to simulate SAHI slicing: 4-slice (1124px) or 9-slice (778px) with configurable weights."""

    def __init__(self, sizes=(1124, 778), weights=(0.6, 0.4), p=0.8):
        super().__init__(p=p)
        self.sizes = list(sizes)
        self.weights = list(weights)

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]
        size = random.choices(self.sizes, weights=self.weights, k=1)[0]
        size = min(size, h, w)
        x = random.randint(0, max(0, w - size))
        y = random.randint(0, max(0, h - size))
        return {"x": x, "y": y, "size": size}

    def apply(self, img, x=0, y=0, size=1124, **params):
        return img[y:y + size, x:x + size]

    def apply_to_bboxes(self, bboxes: np.ndarray, x=0, y=0, size=1124, rows=2024, cols=2024, **params) -> np.ndarray:
        # bboxes: (N, 4+) array, columns are x_min, y_min, x_max, y_max normalised to [0, 1]
        if len(bboxes) == 0:
            return bboxes
        result = bboxes.copy().astype(float)
        result[:, 0] = np.clip(bboxes[:, 0] * cols - x, 0, size) / size
        result[:, 1] = np.clip(bboxes[:, 1] * rows - y, 0, size) / size
        result[:, 2] = np.clip(bboxes[:, 2] * cols - x, 0, size) / size
        result[:, 3] = np.clip(bboxes[:, 3] * rows - y, 0, size) / size
        return result

    def get_transform_init_args_names(self):
        return ("sizes", "weights")


# Register so aug_config dicts can reference it by name via getattr(A, ...)
A.SahiStyleCrop = SahiStyleCrop


# ---------------------------------------------------------------------------
# Handwritten text overlay
# ---------------------------------------------------------------------------

_FONT_PATH = Path(__file__).parent / "HomemadeApple-Regular.ttf"

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


def handwritten_text(image: np.ndarray, **kwargs) -> np.ndarray:
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

        px = random.randint(0, max(0, w - scratch.width))
        py = random.randint(0, max(0, h - scratch.height))
        pil.paste(scratch, (px, py), mask=scratch)

    return np.array(pil)
