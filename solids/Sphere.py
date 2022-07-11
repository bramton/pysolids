import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid

class Sphere(BaseSolid):
    def __init__(self, radius=1):
        self.r = radius
        self.rng = default_rng()

    @property
    def vol(self):
        return (4/3)*np.pi*self.r ** 3

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        self.r = ((3*vol)/(4*np.pi)) ** (1/3)

    @property
    def area(self):
        return 4*np.pi*self.r ** 2

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        self.r = np.sqrt(area/(4*np.pi))

    def sample(self, n):
        super(Sphere, self).sample(n)
        samples = np.empty((n,3))
        # Loop probably not needed
        for i in np.arange(n):
            xyz = self.rng.normal(size=3)
            xyz = (self.r/np.linalg.norm(xyz))*xyz
            samples[i,:] = xyz

        return samples
