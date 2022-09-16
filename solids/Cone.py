import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid
from .ConeWall import ConeWall

class Cone(BaseSolid):
    def __init__(self, radius=1, height=2):
        self.r = radius
        self.h = height
        self.rng = default_rng()
        self.cw = ConeWall(radius=radius, height=height)

    @property
    def vol(self):
        return NotImplemented

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        return NotImplemented

    @property
    def area(self):
        return self.cw.area + np.pi*self.r**2

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        a = self.h/self.r
        self.r = np.sqrt(area/(np.pi*(1 + np.sqrt(1 + a**2))))
        self.h = self.r*a
        self.cw.r = self.r
        self.cw.h = self.h

    def sample(self, n):
        super(Cone, self).sample(n)
        samples = np.empty((n,3))
        n_base = round(((np.pi*self.r**2)/self.area)*n)

        # Sample base of cone
        for i in np.arange(n_base):
            # Rejection sampling
            while True:
                xy = self.rng.uniform(-self.r, self.r, size=2)
                if np.linalg.norm(xy) <= self.r:
                    break
            samples[i,:] = [xy[0], xy[1], self.h]

        # Sample 'wall' of cone
        samples[n_base:,:] = self.cw.sample(n - n_base)

        samples[:,2] = samples[:,2] - self.h/2
        return samples
