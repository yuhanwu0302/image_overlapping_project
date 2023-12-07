from typing import List
import numpy as np
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    

class Gradient:
    GRAD_MAX = 20  # the maximum gradient value

    def __init__(self, start_point: Point, end_point: Point):
        self.start_point = start_point
        self.end_point = end_point
        self.value = self.calc_gradient()

    def calc_gradient(self) -> float:
        delta = self.end_point - self.start_point
        if delta.x == 0:
            return Gradient.GRAD_MAX
        else:
            return delta.y / delta.x

    def __repr__(self) -> str:
        return f"{self.value}"

