from ultralytics import YOLO

from picking_robot.model.generics import BaseModel


class SegmentationModel(BaseModel):
    def __init__(
        self,
        model_path: str,
    ):
        self.model = YOLO(model_path)

    def train(self, data, epochs, batch_size, imgsz, device):
        results = self.model.train(
            data=data, # Dataset paths
            epochs=epochs,                  # Number of epochs
            batch=batch_size,                   # Batch size
            imgsz=imgsz,                  # Image size
            project="logs/train",       # Directory to save results
            name="picking_robot",  # Experiment name
            pretrained=True,             # Use pretrained weights
            device=device
        )

    def predict(self):
        return super().predict()