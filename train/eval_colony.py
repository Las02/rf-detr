import cv2
from pathlib import Path
import fiftyone as fo
import supervision as sv
import typer
from tqdm import tqdm

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: Path = typer.Option(..., help="Path to RF-DETR checkpoint (.pth)"),
    dataset: Path = typer.Option(..., help="COCO split dir (contains _annotations.coco.json)"),
    resolution: int = typer.Option(560, help="Inference resolution"),
    confidence: float = typer.Option(0.3, help="Detection confidence threshold"),
    name: str = typer.Option("colony_eval", help="FiftyOne dataset name"),
    view: bool = typer.Option(False, "--view", help="Launch FiftyOne app after evaluation"),
):
    ann_file = dataset / "_annotations.coco.json"
    if not ann_file.exists():
        typer.echo(f"Error: {ann_file} not found", err=True)
        raise typer.Exit(1)

    # Load / reload dataset in FiftyOne
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    typer.echo(f"Loading dataset from {dataset}")
    fo_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(dataset),
        labels_path=str(ann_file),
        name=name,
    )
    fo_dataset.persistent = True
    typer.echo(f"  {len(fo_dataset)} samples loaded")

    # Load model
    typer.echo(f"Loading model: {model}")
    from rfdetr import RFDETRMedium
    rfdetr_model = RFDETRMedium(pretrain_weights=str(model), device="mps")
    rfdetr_model.model.postprocess.num_select = 300
    rfdetr_model.optimize_for_inference()

    # Run predictions
    typer.echo(f"Running predictions (resolution={resolution}, confidence={confidence})...")
    for sample in tqdm(fo_dataset.iter_samples(progress=True, autosave=True)):
        image = cv2.imread(sample.filepath)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = sv.resize_image(image=image_rgb, resolution_wh=(resolution, resolution))

        detections = rfdetr_model.predict(image_resized, threshold=confidence)
        fo_dets = []
        if detections is not None and len(detections) > 0:
            for xyxy, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                x1, y1, x2, y2 = xyxy
                fo_dets.append(fo.Detection(
                    label="colony",
                    bounding_box=[float(x1) / resolution, float(y1) / resolution,
                                  float(x2 - x1) / resolution, float(y2 - y1) / resolution],
                    confidence=float(conf),
                ))
        sample["predictions"] = fo.Detections(detections=fo_dets)

    # Evaluate
    typer.echo("Evaluating...")
    results = fo_dataset.evaluate_detections(
        "predictions",
        gt_field="detections",
        method="coco",
        eval_key="eval",
    )
    results.print_report()

    if view:
        session = fo.launch_app(fo_dataset)
        session.wait()


if __name__ == "__main__":
    app()
