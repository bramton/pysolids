import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid

class Peanut(BaseSolid):
    def __init__(self, r=1, overlap=0.3):
        self.r = r
        self.o = overlap
        assert (overlap > 0 and overlap < 1), f"Overlap should be between 0 and 1"
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
        return 4*np.pi*(2 - self.o)*self.r**2

    @area.setter
    def area(self, area):
        assert (area > 0), f"Area should be positive, got: {area}"
        self.r = np.sqrt(area/(4*np.pi*(2 - self.o)))

    def sample(self, n):
        super(Peanut, self).sample(n)
        samples = np.empty((n,3))
        theta = np.arccos(1 - self.o)

        for i in np.arange(n):
            while True:
                xyz = self.rng.normal(size=3)
                xyz = (self.r/np.linalg.norm(xyz))*xyz
                phi = np.arccos(-1*xyz[2]/(np.linalg.norm(xyz)*1))
                if phi < theta:
                    continue

                if i > n//2:
                    xyz = np.multiply(xyz,[1,-1,-1]) # Rotate around x-axis
                    xyz = np.subtract(xyz,[0,0,(1-self.o)*self.r])

                else:
                    xyz = np.add(xyz, [0,0,(1-self.o)*self.r])
                samples[i,:] = xyz
                break

        return samples

