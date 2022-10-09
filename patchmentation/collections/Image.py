import numpy as np
from dataclasses import dataclass

@dataclass
class Image:
    path: str

    def summary(self):
        print(
            f'Image path: {self.path}\n'
        )

