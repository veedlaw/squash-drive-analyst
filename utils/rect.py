from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    """Class for representing a rectangle in (x, y, width, height) form."""
    x: float
    y: float
    width: int
    height: int

    def center(self) -> (int, int):
        """
        :return: Center (x: int, y: int) of rectangle.
        """
        return int(self.x + self.width / 2), int(self.y + self.height)

    def center3D(self) -> (int, int, 1):
        """
        :return: Center (x: int, y: int, 1) of rectangle.
        """
        return *self.center(), 1

    def area(self) -> int:
        """
        :return: Area of rectangle.
        """
        return self.width * self.height
