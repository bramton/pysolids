import math
from solids.BaseSolid import BaseSolid

class Cylinder(BaseSolid):
    """
    :param ratio: Ratio of the cylinder height to the radius
    """
    def __init__(self, ratio=2):
        self.r = 1
        self.ratio = ratio

    @property
    def vol(self):
        return math.pi*self.r*self.r*self.ratio*self.r

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        self.r = (vol/(math.pi*self.ratio)) ** (1/3)

    @property
    def area(self):
        return 2*math.pi*self.r*self.r*(1 + self.ratio)

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        self.r = math.sqrt(area/(2*math.pi*(1 + self.ratio)))

    def SamplePoint(self, n=1024):
        assert n > 0, f"Number of points should be more than one, got: {n}"


