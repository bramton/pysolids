import numpy as np
from numpy.random import default_rng
from solids.BaseSolid import BaseSolid

class Cube(BaseSolid):
    def __init__(self,length=1):
        self.L = length
        self.rng = default_rng()

    @property
    def vol(self):
        return self.L**3

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        self.L = vol ** (1/3)

    @property
    def area(self):
        return 6*self.L**2

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        self.L = np.sqrt(area/6)

    def sample(self, n):
        super(Cube, self).sample(n)
        samples = np.empty((n,3))
        for i in np.arange(n):
            plane = self.rng.choice([0,1,2])
            coord = self.rng.uniform(-self.L/2, self.L/2, size=2)
            other = self.rng.choice([-1,1])*0.5*self.L
            # xz-plane
            if plane==0:
                samples[i,:] = [coord[0], other, coord[1]]
            # xy-plane
            elif plane==1:
                samples[i,:] = [coord[0], coord[1], other]
            # yz-plane
            else:
                samples[i,:] = [other, coord[0], coord[1]]

        return samples
