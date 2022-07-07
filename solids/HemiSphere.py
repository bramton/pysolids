import numpy as np
from numpy.random import default_rng
from solids.BaseSolid import BaseSolid

class HemiSphere(BaseSolid):
    def __init__(self, radius=1):
        self.r = radius
        self.rng = default_rng()

    @property
    def vol(self):
        return (2/3)*np.pi*self.r ** 3

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        self.r = ((3*vol)/(2*np.pi)) ** (1/3)

    @property
    def area(self):
        return 3*np.pi*self.r ** 2

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        self.r = np.sqrt(area/(3*np.pi))

    def sample(self, n):
        super(HemiSphere, self).sample(n)
        samples = np.empty((n,3))
        # Dome of hemisphere
        # Loop probably not needed
        for i in np.arange(round(2*n/3)):
            xyz = self.rng.normal(size=3)
            xyz = (self.r/np.linalg.norm(xyz))*xyz
            xyz[2] = np.abs(xyz[2])
            samples[i,:] = xyz

        # Base of hemisphere
        for i in np.arange(round(2*n/3), n):
            # Rejection sampling
            while True:
                xy = self.rng.uniform(-self.r, self.r, size=2)
                if np.linalg.norm(xy) <= self.r:
                    break
            samples[i,:] = [xy[0], xy[1], 0]

        samples[:,2] = samples[:,2] - 0.5*self.r
        return samples
