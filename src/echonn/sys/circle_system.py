from .system import DynamicalSystem


class CircleSystem(DynamicalSystem):
    def __init__(self):
        super().__init__(dim=2)

    def fun(self, t, v):
        x, y = v
        xp = -y
        yp = x
        return xp, yp
