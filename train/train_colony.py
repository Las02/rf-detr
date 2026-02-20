# ------------------------------------------------------------------------
# Colony Counter — Custom RF-DETR Large Training Script
# Applies online augmentations equivalent to augment_data/augment.py
# ------------------------------------------------------------------------
"""Train an RFDETRLarge model with colony-counting augmentations applied online.

Augmentation pipeline mirrors augment_data/augment.py:
  - HorizontalFlip, VerticalFlip, Rotate(±180°)
  - RandomBrightnessContrast, HueSaturationValue, ISONoise
  - ImageCompression, ChannelDropout
  --extra mode additionally enables:
  - Perspective, Defocus, EdgeFlare, ReflectColonies
"""

import json
import math
import multiprocessing
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import typer
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
import rfdetr.util.misc as utils
import rfdetr.datasets.transforms as T
from rfdetr import RFDETRLarge
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.transforms import AlbumentationsWrapper, ComposeAugmentations
from rfdetr.engine import evaluate, train_one_epoch
from rfdetr.main import populate_args
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.logger import get_logger
from rfdetr.util.utils import ModelEma

logger = get_logger()
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Colony-specific augmentation helpers (inlined from augment_data/augment.py)
# ---------------------------------------------------------------------------


def _reflect_colonies(
    image: np.ndarray,
    bboxes: list[list[float]],
    num_reflections: tuple[int, int] | None = (3, 8),
    opacity: tuple[float, float] = (0.1, 0.35),
    max_offset: int = 80,
    blur_ksize: tuple[int, int] = (3, 7),
    flip_prob: float = 0.5,
    p: float = 0.5,
) -> np.ndarray:
    """Add faint colony reflections as unlabeled distractors.

    Args:
        image: HWC numpy array (BGR or RGB).
        bboxes: List of [x, y, w, h] bounding boxes (absolute pixel coords).
        num_reflections: Range (min, max) of how many reflections to add.
            ``None`` uses all boxes.
        opacity: Alpha blending range for the reflection patches.
        max_offset: Maximum pixel displacement for reflection placement.
        blur_ksize: Kernel size range for Gaussian blur applied to patches.
        flip_prob: Probability of flipping individual reflection patches.
        p: Probability of applying any reflections at all.

    Returns:
        Augmented image array (same dtype and shape as input).
    """
    rng = np.random.default_rng()
    if rng.random() > p or len(bboxes) == 0:
        return image

    image = image.copy()
    h, w = image.shape[:2]
    if num_reflections is None:
        n = len(bboxes)
    else:
        n = rng.integers(num_reflections[0], num_reflections[1] + 1)
        n = min(n, len(bboxes))
    indices = rng.choice(len(bboxes), size=n, replace=False)
    angle = rng.uniform(0, 2 * np.pi)

    for idx in indices:
        x_min, y_min, bw, bh = [int(v) for v in bboxes[idx]]
        if bw < 2 or bh < 2:
            continue

        patch = image[y_min : y_min + bh, x_min : x_min + bw].copy()
        alpha = rng.uniform(opacity[0], opacity[1])
        k = rng.integers(blur_ksize[0] // 2, blur_ksize[1] // 2 + 1) * 2 + 1
        patch = cv2.GaussianBlur(patch, (k, k), 0)
        if rng.random() < flip_prob:
            patch = cv2.flip(patch, 1)
        if rng.random() < flip_prob * 0.3:
            patch = cv2.flip(patch, 0)

        min_dist = max(bw, bh)
        if min_dist >= max_offset:
            continue
        dist = rng.uniform(min_dist, max_offset)
        new_x = int(x_min + dist * np.cos(angle))
        new_y = int(y_min + dist * np.sin(angle))
        new_x = int(np.clip(new_x, 0, w - bw))
        new_y = int(np.clip(new_y, 0, h - bh))

        circle_mask = np.zeros((bh, bw), dtype=np.float32)
        cv2.ellipse(circle_mask, (bw // 2, bh // 2), (bw // 2, bh // 2), 0, 0, 360, alpha, -1)

        roi = image[new_y : new_y + bh, new_x : new_x + bw]
        if roi.shape[:2] != patch.shape[:2]:
            continue
        mask_3ch = circle_mask[:, :, np.newaxis]
        blended = (patch * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
        image[new_y : new_y + bh, new_x : new_x + bw] = blended

    return image


def _build_edge_flare(
    src_radius: int = 250,
    num_flare_circles_range: tuple[int, int] = (3, 6),
    p: float = 0.8,
) -> A.OneOf:
    """Build an Albumentations OneOf transform with edge-ROI sun flares.

    Args:
        src_radius: Radius of the light source in pixels.
        num_flare_circles_range: (min, max) number of flare circles.
        p: Probability of applying the flare.

    Returns:
        ``A.OneOf`` that randomly picks one of the four edge positions.
    """
    return A.OneOf(
        [
            A.RandomSunFlare(
                flare_roi=(0, 0, 0.05, 1),
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
            A.RandomSunFlare(
                flare_roi=(0.95, 0, 1, 1),
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.05),
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0.95, 1, 1),
                angle_range=(0, 1),
                num_flare_circles_range=num_flare_circles_range,
                src_radius=src_radius,
                method="physics_based",
                p=1.0,
            ),
        ],
        p=p,
    )


# ---------------------------------------------------------------------------
# (PIL.Image, dict) → (PIL.Image, dict) transform wrappers
# ---------------------------------------------------------------------------


class ReflectColoniesTransform:
    """Apply _reflect_colonies as an online transform during training.

    Reads existing bounding boxes for placement but does not modify them —
    the reflections are unlabeled distractors that only affect pixel values.

    Args:
        num_reflections: (min, max) number of reflections to add, or ``None``
            to use all boxes.
        opacity: (min, max) alpha blending range for reflection patches.
        max_offset: Maximum pixel displacement for reflection placement.
        blur_ksize: Gaussian blur kernel size range applied to patches.
        flip_prob: Probability of horizontally/vertically flipping each patch.
        p: Probability of applying any reflections to a given image.
    """

    def __init__(
        self,
        num_reflections: tuple[int, int] | None = (3, 8),
        opacity: tuple[float, float] = (0.1, 0.35),
        max_offset: int = 80,
        blur_ksize: tuple[int, int] = (3, 7),
        flip_prob: float = 0.5,
        p: float = 0.5,
    ) -> None:
        self.num_reflections = num_reflections
        self.opacity = opacity
        self.max_offset = max_offset
        self.blur_ksize = blur_ksize
        self.flip_prob = flip_prob
        self.p = p

    def __call__(
        self, image: Image.Image, target: dict
    ) -> tuple[Image.Image, dict]:
        """Apply reflection augmentation.

        Args:
            image: PIL Image in RGB format.
            target: RF-DETR target dict; ``target["boxes"]`` (if present) must be
                a float tensor of shape ``(N, 4)`` in ``[x1, y1, x2, y2]``
                absolute-pixel format.

        Returns:
            Tuple of (augmented PIL Image, unchanged target dict).
        """
        image_np = np.array(image)  # RGB HWC
        bboxes_xywh: list[list[float]] = []
        if "boxes" in target and len(target["boxes"]) > 0:
            for x1, y1, x2, y2 in target["boxes"].tolist():
                bboxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

        aug_np = _reflect_colonies(
            image_np,
            bboxes_xywh,
            num_reflections=self.num_reflections,
            opacity=self.opacity,
            max_offset=self.max_offset,
            blur_ksize=self.blur_ksize,
            flip_prob=self.flip_prob,
            p=self.p,
        )
        return Image.fromarray(aug_np), target

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_reflections={self.num_reflections}, p={self.p})"
        )


class EdgeFlareTransform:
    """Apply edge-positioned sun flares as an online transform during training.

    Wraps an ``A.OneOf`` of four ``A.RandomSunFlare`` transforms placed at
    the left, right, top, and bottom edges of the image.
    Boxes are not modified (pixel-level transform only).

    Args:
        src_radius: Light source radius in pixels.
        num_flare_circles_range: (min, max) number of flare circles.
        p: Probability of applying a flare to a given image.
    """

    def __init__(
        self,
        src_radius: int = 250,
        num_flare_circles_range: tuple[int, int] = (3, 6),
        p: float = 0.1,
    ) -> None:
        self.p = p
        self._flare = _build_edge_flare(
            src_radius=src_radius,
            num_flare_circles_range=num_flare_circles_range,
            p=p,
        )

    def __call__(
        self, image: Image.Image, target: dict
    ) -> tuple[Image.Image, dict]:
        """Apply edge flare augmentation.

        Args:
            image: PIL Image in RGB format.
            target: RF-DETR target dict (boxes unchanged).

        Returns:
            Tuple of (augmented PIL Image, unchanged target dict).
        """
        image_np = np.array(image)
        result = self._flare(image=image_np)
        return Image.fromarray(result["image"]), target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


# ---------------------------------------------------------------------------
# Transform pipeline
# ---------------------------------------------------------------------------

#: Albumentations transforms that mirror augment_data/augment.py build_pipeline()
AUG_COLONY: dict[str, dict] = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": 180, "border_mode": 0, "value": 0, "p": 0.75},
    "RandomBrightnessContrast": {"p": 0.3},
    "HueSaturationValue": {"p": 0.3},
    "ISONoise": {"p": 0.3},
    "ImageCompression": {"p": 0.3},
    "ChannelDropout": {"p": 0.3},
}

#: Extra transforms added when --extra is passed (mirrors augment.py extra=True mode)
AUG_COLONY_EXTRA: dict[str, dict] = {
    "Perspective": {"scale": (0.02, 0.05), "pad_mode": 0, "pad_val": 0, "p": 0.05},
    "Defocus": {"p": 0.1, "alias_blur": (0.02, 0.1), "radius": (1, 2)},
}


def build_colony_transforms(
    image_set: str,
    resolution: int = 576,
    extra: bool = False,
) -> T.Compose:
    """Build the transform pipeline for colony training or validation.

    For training: SquareResize → colony augmentations → optional extra
    augmentations → Normalize.
    For val/test: SquareResize → Normalize.

    Args:
        image_set: One of ``"train"``, ``"val"``, or ``"test"``.
        resolution: Target square resolution in pixels.
        extra: If ``True``, add Perspective, Defocus, EdgeFlare, and
            ReflectColonies on top of the base colony pipeline.

    Returns:
        A ``T.Compose`` transform pipeline.
    """
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    if image_set == "train":
        aug_cfg = {**AUG_COLONY}
        if extra:
            aug_cfg.update(AUG_COLONY_EXTRA)

        aug_list = AlbumentationsWrapper.from_config(aug_cfg)
        if extra:
            img_src_radius = max(30, int(resolution * 0.4))
            aug_list += [
                EdgeFlareTransform(src_radius=img_src_radius, p=0.1),
                ReflectColoniesTransform(p=0.5),
            ]

        return T.Compose(
            [
                T.SquareResize([resolution]),
                ComposeAugmentations(aug_list),
                normalize,
            ]
        )

    # val / test — no augmentation
    return T.Compose([T.SquareResize([resolution]), normalize])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _load_class_names(dataset_dir: Path) -> list[str]:
    """Read category names from the training annotation file.

    Args:
        dataset_dir: Root of the Roboflow COCO dataset (contains train/).

    Returns:
        Sorted list of class names (excludes ``supercategory == "none"``).
    """
    ann_path = dataset_dir / "train" / "_annotations.coco.json"
    with open(ann_path) as f:
        data = json.load(f)
    return [c["name"] for c in data["categories"] if c.get("supercategory") != "none"]


def build_colony_dataset(
    image_set: str,
    dataset_dir: Path,
    resolution: int,
    extra: bool = False,
) -> CocoDetection:
    """Build a CocoDetection dataset with the colony augmentation pipeline.

    Expects the standard Roboflow COCO layout::

        dataset_dir/
            train/_annotations.coco.json  + images
            valid/_annotations.coco.json  + images
            test/_annotations.coco.json   + images

    Args:
        image_set: ``"train"``, ``"val"``, or ``"test"``.
        dataset_dir: Path to the dataset root directory.
        resolution: Square resize resolution passed to the transform pipeline.
        extra: Enable extra augmentations (Perspective, Defocus, EdgeFlare,
            ReflectColonies).

    Returns:
        A ``CocoDetection`` dataset with the colony transforms applied.
    """
    split_map = {"train": "train", "val": "valid", "test": "test"}
    split = split_map[image_set]
    img_folder = dataset_dir / split
    ann_file = img_folder / "_annotations.coco.json"
    transforms = build_colony_transforms(image_set, resolution, extra)
    return CocoDetection(img_folder, ann_file, transforms=transforms)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def train(
    dataset_dir: Path = typer.Option(..., help="Root of the Roboflow COCO dataset"),
    output_dir: Path = typer.Option(Path("output/colony"), help="Where to save checkpoints"),
    epochs: int = typer.Option(100, help="Total training epochs"),
    batch_size: int = typer.Option(4, help="Per-GPU batch size"),
    resolution: int = typer.Option(704, help="Square training resolution (Large default: 704)"),
    lr: float = typer.Option(1e-4, help="Base learning rate"),
    weight_decay: float = typer.Option(1e-4, help="AdamW weight decay"),
    warmup_epochs: float = typer.Option(1.0, help="Linear LR warm-up epochs"),
    num_workers: int = typer.Option(2, help="DataLoader worker count"),
    device: str = typer.Option("auto", help="Device: auto | cuda | mps | cpu"),
    use_ema: bool = typer.Option(True, help="Use Exponential Moving Average model"),
    extra: bool = typer.Option(False, help="Enable extra augmentations: Perspective, Defocus, EdgeFlare, ReflectColonies"),
    resume: str = typer.Option("", help="Path to checkpoint to resume from"),
) -> None:
    """Train RFDETRLarge on a colony-counting COCO dataset with online augmentation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch_device = torch.device(device)
    typer.echo(f"Using device: {device}")

    # --- Class names ---
    class_names = _load_class_names(dataset_dir)
    # Roboflow COCO uses 1-indexed category IDs, so num_classes = max_id + 1
    num_classes = len(class_names) + 1
    typer.echo(f"Classes ({len(class_names)}): {class_names}")

    # --- Model ---
    # RFDETRLarge downloads pretrained weights and initialises the model.
    typer.echo("Building RFDETRLarge …")
    rfdetr = RFDETRLarge(num_classes=num_classes)
    rfdetr.model.reinitialize_detection_head(num_classes)
    raw_model: torch.nn.Module = rfdetr.model.model.to(torch_device)

    # --- Args namespace (needed by criterion, optimizer, scheduler helpers) ---
    # Fields we supply explicitly below — don't double-pass them from model_config
    _skip = {"num_classes", "pretrain_weights", "license", "class_names", "device"}
    mc = {k: v for k, v in rfdetr.model_config.model_dump().items() if k not in _skip}
    mc["resolution"] = resolution  # allow CLI override

    args = populate_args(
        **mc,
        num_classes=num_classes,
        dataset_dir=str(dataset_dir),
        dataset_file="roboflow",
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        lr_encoder=lr * 1.5,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        lr_scheduler="cosine",
        device=device,
        aug_config={},           # transforms are applied by our custom pipeline
        square_resize_div_64=True,
        multi_scale=True,
        num_select=300,
        num_queries=300,
        output_dir=str(output_dir),
        ema_tau=100,
        lr_component_decay=0.7,
        use_ema=use_ema,
        num_workers=num_workers,
        resume=resume,
    )

    # --- Criterion ---
    criterion, postprocess = build_criterion_and_postprocessors(args)
    criterion.to(torch_device)

    # --- Optimizer ---
    param_dicts = get_param_dict(args, raw_model)
    param_dicts = [p for p in param_dicts if p["params"].requires_grad]
    optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)

    # --- LR scheduler ---
    grad_accum_steps = args.grad_accum_steps

    # Datasets first (need len for scheduler)
    typer.echo("Loading datasets …")
    dataset_train = build_colony_dataset("train", dataset_dir, resolution, extra)
    dataset_val = build_colony_dataset("val", dataset_dir, resolution, extra=False)
    dataset_test_path = dataset_dir / "test" / "_annotations.coco.json"
    run_test = dataset_test_path.exists()
    if run_test:
        dataset_test = build_colony_dataset("test", dataset_dir, resolution, extra=False)

    typer.echo(f"  train={len(dataset_train)}, val={len(dataset_val)}")

    effective_batch_size = batch_size * grad_accum_steps
    steps_per_epoch = math.ceil(len(dataset_train) / effective_batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --- DataLoaders ---
    # Handle macOS/Windows spawn multiprocessing restriction
    if num_workers > 0 and multiprocessing.get_start_method(allow_none=True) == "spawn":
        import __main__
        if not hasattr(__main__, "__file__"):
            typer.echo("Warning: setting num_workers=0 (no __main__ guard for spawn)", err=True)
            num_workers = 0

    sampler_train = RandomSampler(dataset_train)
    batch_sampler_train = BatchSampler(sampler_train, effective_batch_size, drop_last=True)
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset_val),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=num_workers,
    )
    coco_api_val = get_coco_api_from_dataset(dataset_val)

    if run_test:
        data_loader_test = DataLoader(
            dataset_test,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset_test),
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=num_workers,
        )
        coco_api_test = get_coco_api_from_dataset(dataset_test)

    # --- EMA ---
    ema_m = ModelEma(raw_model, decay=args.ema_decay, tau=args.ema_tau) if use_ema else None

    # --- Resume ---
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        typer.echo(f"Resumed from {resume} (epoch {start_epoch})")

    # --- Training loop ---
    typer.echo(f"Training for {epochs} epochs …")
    best_map = 0.0

    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            raw_model,
            criterion,
            lr_scheduler,
            data_loader_train,
            optimizer,
            torch_device,
            epoch,
            batch_size,
            args.clip_max_norm,
            ema_m=ema_m,
            schedules={},
            num_training_steps_per_epoch=steps_per_epoch,
            vit_encoder_num_layers=args.vit_encoder_num_layers,
            args=args,
            callbacks=defaultdict(list),
        )

        val_stats, _ = evaluate(
            raw_model, criterion, postprocess, data_loader_val, coco_api_val, torch_device
        )

        map50_95 = val_stats.get("coco_eval_bbox", [0])[0]
        typer.echo(
            f"Epoch {epoch:3d} | loss={train_stats.get('loss', 0):.4f} | "
            f"val mAP={map50_95:.4f}"
        )

        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if use_ema and ema_m is not None:
            checkpoint["ema"] = ema_m.module.state_dict()

        torch.save(checkpoint, output_dir / "checkpoint.pth")
        if map50_95 > best_map:
            best_map = map50_95
            torch.save(checkpoint, output_dir / "best_model.pth")
            typer.echo(f"  → New best mAP: {best_map:.4f}")

    if run_test:
        typer.echo("Running final test evaluation …")
        evaluate(
            raw_model, criterion, postprocess, data_loader_test, coco_api_test, torch_device
        )

    typer.echo(f"Done. Best val mAP: {best_map:.4f}")
    typer.echo(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    app()
