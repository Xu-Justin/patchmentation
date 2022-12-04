from typing import Tuple

class BBox:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __iter__(self) -> Tuple[int, int, int, int]:
        return iter((self.xmin, self.ymin, self.xmax, self.ymax))

    def __repr__(self) -> str:
        return f'BBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})'

    def __eq__(self, bbox: 'BBox') -> bool:
        return (self.xmin == bbox.xmin) and (self.ymin == bbox.ymin) and (self.xmax == bbox.xmax) and (self.ymax == bbox.ymax) 

    @property
    def xmin(self) -> int:
        return getattr(self, '_xmin', None)

    @xmin.setter
    def xmin(self, value: int):
        if value is not None:
            if value < 0:
                raise ValueError(f'xmin value cannot smaller than zero, xmin value : {value}')
            if self.xmax is not None and value > self.xmax:
                raise ValueError(f'xmin value cannot greater than xmax, xmin value : {value}, xmax : {self.xmax}')
        self._xmin =  value

    @property
    def ymin(self) -> int:
        return getattr(self, '_ymin', None)

    @ymin.setter
    def ymin(self, value: int):
        if value is not None:
            if value < 0:
                raise ValueError(f'ymin value cannot smaller than zero, ymin value : {value}')
            if self.ymax is not None and value > self.ymax:
                raise ValueError(f'ymin value cannot greater than ymax, ymin value : {value}, ymax : {self.ymax}')
        self._ymin =  value

    @property
    def xmax(self) -> int:
        return getattr(self, '_xmax', None)

    @xmax.setter
    def xmax(self, value: int):
        if value is not None:
            if value < 0:
                raise ValueError(f'xmax value cannot smaller than zero, xmax value : {value}')
            if self.xmin is not None and value < self.xmin:
                raise ValueError(f'xmax value cannot smaller than xmin, xmax value : {value}, xmin : {self.xmin}')
        self._xmax =  value

    @property
    def ymax(self) -> int:
        return getattr(self, '_ymax', None)

    @ymax.setter
    def ymax(self, value: int):
        if value is not None:
            if value < 0:
                raise ValueError(f'ymax value cannot smaller than zero, ymax value : {value}')
            if self.ymin is not None and value < self.ymin:
                raise ValueError(f'ymax value cannot smaller than ymin, ymax value : {value}, ymin : {self.ymin}')
        self._ymax =  value

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height

class OverflowBBox(BBox):
    @BBox.xmin.setter
    def xmin(self, value: int):
        self._xmin =  value

    @BBox.ymin.setter
    def ymin(self, value: int):
        self._ymin =  value

    @BBox.xmax.setter
    def xmax(self, value: int):
        self._xmax =  value

    @BBox.ymax.setter
    def ymax(self, value: int):
        self._ymax =  value
    