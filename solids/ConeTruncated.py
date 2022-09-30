import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid
from .ConeWall import ConeWall

class ConeTruncated(BaseSolid):
    def __init__(self, radius=1, height=2, cutoff=0.3):
        self.r = radius
        self.h = height
        self.cutoff = cutoff
        self.rng = default_rng()
        self.cw = ConeWall(radius=radius, height=height, cutoff=cutoff)

    @property
    def vol(self):
        return NotImplemented

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        return NotImplemented

    @property
    def area(self):
        return self.cw.area + np.pi*self.r**2 + np.pi*(self.r*self.cutoff)**2

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        a = np.sqrt(area/self.area)
        self.r = a*self.r
        self.h = a*self.h
        self.cw.r = self.r
        self.cw.h = self.h

    def sample(self, n):
        super(ConeTruncated, self).sample(n)
        samples = np.empty((n,3))
        n_base = round(((np.pi*self.r**2)/self.area)*n)
        rc = self.r*self.cutoff
        n_top = round(((np.pi*rc**2)/self.area)*n)

        # Sample base of cone
        for i in np.arange(n_base):
            # Rejection sampling
            while True:
                xy = self.rng.uniform(-self.r, self.r, size=2)
                if np.linalg.norm(xy) <= self.r:
                    break
            samples[i,:] = [xy[0], xy[1], self.h]

        # Sample top of cone
        for i in np.arange(n_base, n_top+n_base):
            # Rejection sampling
            while True:
                xy = self.rng.uniform(-rc, rc, size=2)
                if np.linalg.norm(xy) <= rc:
                    break
            samples[i,:] = [xy[0], xy[1], self.h*self.cutoff]

        # Sample 'wall' of cone
        samples[n_base+n_top:,:] = self.cw.sample(n - n_base - n_top)

        samples[:,2] = samples[:,2] - self.h/2
        return samples
