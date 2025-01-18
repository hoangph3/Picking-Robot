import typer
from picking_robot.cli.dataset_cli import dataset_app
from picking_robot.cli.training_cli import training_app


app = typer.Typer()
app.add_typer(
    dataset_app,
    name="dataset",
    help="Dataset management",
)
app.add_typer(
    training_app,
    name="train",
    help="Training management",
)


def cli():
    app()
