from dataclasses import dataclass

@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def __iter__(self):
        return iter((self.xmin, self.ymin, self.xmax, self.ymax))

    def summary(self):
        print(
            f'xmin: {self.xmin}\n'
            f'ymin: {self.ymin}\n'
            f'xmax: {self.xmax}\n'
            f'ymax: {self.ymax}\n'
        )

