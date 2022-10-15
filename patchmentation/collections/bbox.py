from typing import Tuple
from dataclasses import dataclass

@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def __iter__(self) -> Tuple[int, int, int, int]:
        return iter((self.xmin, self.ymin, self.xmax, self.ymax))

    def summary(self) -> None:
        print(
            f'xmin: {self.xmin}\n'
            f'ymin: {self.ymin}\n'
            f'xmax: {self.xmax}\n'
            f'ymax: {self.ymax}\n'
        )

    def width(self) -> int:
        return self.xmax - self.xmin

    def height(self) -> int:
        return self.ymax - self.ymin

    def area(self) -> int:
        return self.width() * self.height()
