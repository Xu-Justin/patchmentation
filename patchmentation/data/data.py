from patchmentation.collections import Dataset
from . import datautils

import os
from abc import ABC, abstractmethod

class Data(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def root(self) -> str:
        return os.path.join(datautils.FOLDER_CACHE, self.name)

    @property
    def folder(self) -> str:
        return os.path.join(self.root, self.name)

    @property
    def archive_name(self) -> str:
        return self.name + '.archive'

    @property
    def file_archive(self) -> str:
        return os.path.join(self.root, self.archive_name)

    @abstractmethod
    def load(self) -> Dataset:
        '''load dataset from self.folder'''
        pass

    @abstractmethod
    def download(self, overwrite: bool = False) -> None:
        '''download dataset archive to self.file_archive'''
        pass

    @abstractmethod
    def extract(self, overwrite: bool = False) -> None:
        '''extract dataset archive from self.file_archive to self.folder'''
        pass

    def exists_archive(self) -> bool:
        return os.path.exists(self.file_archive)

    def exists(self) -> bool:
        return os.path.exists(self.folder)

    def remove_archive(self) -> None:
        datautils.rm(self.file_archive)
        
    def remove(self) -> None:
        datautils.rm(self.folder)

    def initialize(self) -> None:
        if not self.exists():
            if not self.exists_archive():
                self.download(overwrite=True)
            self.extract(overwrite=True)
