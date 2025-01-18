from pathlib import Path
import shutil
import json
import os
import cv2

from picking_robot.dataset.generics import BaseDataset


class SegmentationDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        dataset_dir: str,
        val_size: 0.1,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_dir = Path(dataset_dir)
        self.val_size = val_size
        self.load_data()
        self.split_data()
        self.get_labels()

    def load_data(self):
        self.image_dir = self.data_dir / "images"
        annotation_file = self.data_dir / "annotation.json"
        camera_info_file = self.data_dir / "camera.json"
        train_ids_file = self.data_dir / "train_ids.txt"
        test_ids_file = self.data_dir / "test_ids.txt"

        self.annotation = json.loads(annotation_file.read_text())
        self.camera_info = json.loads(camera_info_file.read_text())
        self._train_ids = train_ids_file.read_text().splitlines()
        self.test_ids = test_ids_file.read_text().splitlines()

    def split_data(self):
        train_samples = len(self._train_ids)
        self.train_ids = self._train_ids[int(train_samples*self.val_size):]
        self.val_ids = self._train_ids[:int(train_samples*self.val_size)]

    def get_labels(self):
        self.labels = {
            "top": 0,
            "overlap": 1
        }

    def save_dataset(self):
        dataset = {
            "train": self.train_ids, "val": self.val_ids, "test": self.test_ids
        }
        for ds in dataset.keys():
            images_dir = self.dataset_dir / ds / "images"
            images_dir.mkdir(exist_ok=True, parents=True)
            labels_dir = self.dataset_dir / ds / "labels"
            labels_dir.mkdir(exist_ok=True, parents=True)

        for image_file in self.image_dir.iterdir():
            image_id, _ = os.path.splitext(image_file)
            image_id = os.path.basename(image_id)

            for ds, ds_ids in dataset.items():
                images_dir = self.dataset_dir / ds / "images"
                labels_dir = self.dataset_dir / ds / "labels"
                if image_id not in ds_ids:
                    continue
                shutil.copy(src=image_file, dst=images_dir)
                img = cv2.imread(image_file)
                height, width, channels = img.shape

                with open(labels_dir / f"{image_id}.txt", "w") as f:
                    for polygon, label in zip(self.annotation[image_id]['polygons'], self.annotation[image_id]['labels']):
                        label_idx = self.labels[label]
                        xs = polygon['all_points_x']
                        ys = polygon['all_points_y']
                        points = []
                        for x, y in zip(xs, ys):
                            points.append(x / width)
                            points.append(y / height)
                        lines = [label_idx] + points
                        lines = list(map(str, lines))
                        f.write(' '.join(lines)+'\n')

    def get_dataset(self):
        return super().get_dataset()
