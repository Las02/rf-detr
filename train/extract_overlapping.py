from pathlib import Path
import fiftyone as fo
import typer

app = typer.Typer(add_completion=False)


def _iou(b1, b2):
    x1 = max(b1[0], b2[0]);      y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2]); y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


@app.command()
def main(
    dataset: Path = typer.Option(..., help="COCO split dir"),
    iou: float = typer.Option(0.7, help="IoU threshold"),
    source_name: str = typer.Option("colony_source"),
    output_name: str = typer.Option("overlapping_colonies"),
    view: bool = typer.Option(False, "--view"),
):
    ann_file = dataset / "_annotations.coco.json"
    if not ann_file.exists():
        typer.echo(f"Error: {ann_file} not found", err=True)
        raise typer.Exit(1)

    # Load source dataset
    if fo.dataset_exists(source_name):
        fo.delete_dataset(source_name)
    typer.echo(f"Loading {dataset}")
    src = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(dataset),
        labels_path=str(ann_file),
        name=source_name,
    )
    src.persistent = True
    typer.echo(f"  {len(src)} samples")

    # Tag overlapping detections
    n_overlapping_samples = 0
    for sample in src.iter_samples(progress=True, autosave=True):
        dets = sample.detections.detections if sample.detections else []
        involved = set()
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                if _iou(dets[i].bounding_box, dets[j].bounding_box) > iou:
                    involved.add(i)
                    involved.add(j)
        overlapping = [dets[k] for k in involved]
        sample["overlapping_detections"] = fo.Detections(detections=overlapping)
        if overlapping:
            n_overlapping_samples += 1

    typer.echo(f"Samples with overlapping colonies (IoU>{iou}): {n_overlapping_samples}/{len(src)}")

    # Build filtered view and clone to new dataset
    from fiftyone import ViewField as F
    overlap_view = src.match(F("overlapping_detections.detections").length() > 0)

    if fo.dataset_exists(output_name):
        fo.delete_dataset(output_name)
    out = overlap_view.clone(output_name)
    out.persistent = True
    typer.echo(f"Saved '{output_name}' with {len(out)} samples")

    if view:
        session = fo.launch_app(out)
        session.wait()


if __name__ == "__main__":
    app()
