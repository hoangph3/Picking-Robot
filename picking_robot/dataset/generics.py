from abc import ABC, abstractmethod


class BaseDataset(ABC):
    @abstractmethod
    def save_dataset(self):
        pass

    @abstractmethod
    def get_dataset(self):
        pass
