import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid
from .PyramidWall import PyramidWall

class Pyramid(BaseSolid):
    def __init__(self, a=2, h=2):
        self.a = a
        self.h = h
        self.pw = PyramidWall(a,h)
        self.rng = default_rng()

    @property
    def vol(self):
        return NotImplemented

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        return NotImplemented

    @property
    def area(self):
        return self.pw.area + self.a**2

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        a = np.sqrt(area/self.area)
        self.a = self.a*a
        self.h = self.h*a
        self.pw.a = self.a
        self.pw.h = self.h

    def sample(self, n):
        super(Pyramid, self).sample(n)
        samples = np.zeros((n,3))

        n_base = round(n*(self.a**2)/self.area)
        samples[:n_base,0:2] = self.rng.uniform(-self.a/2, self.a/2, size=(n_base,2))
        samples[n_base:,:] = self.pw.sample(n-n_base)
        samples[:,2] = samples[:,2] - self.h/2

        return samples

