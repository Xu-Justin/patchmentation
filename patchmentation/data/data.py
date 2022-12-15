from patchmentation.collections import Dataset
from abc import ABC, abstractmethod

class Data(ABC):
    
    @abstractmethod
    def load(self) -> Dataset:
        pass

    @abstractmethod
    def download(self, overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def extract(self, overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def exists_archive(self) -> bool:
        pass

    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def remove_archive(self) -> None:
        pass

    @abstractmethod
    def remove(self) -> None:
        pass
