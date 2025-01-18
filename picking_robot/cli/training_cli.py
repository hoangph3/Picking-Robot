import typer
from picking_robot.model.segmentation import SegmentationModel

training_app = typer.Typer()


@training_app.command("segmentation")
def segmentation(
    model_path: str = typer.Option(..., '--model-path'),
    data: str = typer.Option(..., '--data'),
    epochs: int = typer.Option(500, '--epochs'),
    batch_size: int = typer.Option(4, '--batch_size'),
    imgsz: int = typer.Option(640, '--imgsz'),
    device: str = typer.Option('cuda', '--device'),
):
    """
    Training segmentation model from data directory
    """
    model = SegmentationModel(model_path)
    model.train(
        data=data, epochs=epochs, batch_size=batch_size, imgsz=imgsz, device=device
    )