import typer
from picking_robot.dataset.segmentation import SegmentationDataset

dataset_app = typer.Typer()


@dataset_app.command("segmentation")
def segmentation(
    data_dir: str = typer.Option(..., '--data-dir'),
    val_size: str = typer.Option(..., '--val-size'),
    dataset_dir: str = typer.Option(..., '--dataset-dir'),
):
    """
    Convert data to YOLO dataset
    """
    val_size = float(val_size)
    segmentation_dataset = SegmentationDataset(data_dir=data_dir, dataset_dir=dataset_dir, val_size=val_size)
    segmentation_dataset.save_dataset()
