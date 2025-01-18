from pathlib import Path
import sys
import os


def setup():
    file_dir = os.path.dirname(__file__)
    module_dir = str(Path(file_dir).parent.resolve())
    sys.path.append(module_dir)


if __name__ == "__main__":
    setup()
    from picking_robot.cli.toolkit import cli
    cli()
