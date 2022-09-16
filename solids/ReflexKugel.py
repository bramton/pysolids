import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid
from .ConeWall import ConeWall

class ReflexKugel(BaseSolid):
    def __init__(self, r=1, sr=1):
        self.r = r
        self.sr = sr
        assert (sr < np.pi*4), f"Steradian should be smaller than 4*pi"
        self.rng = default_rng()

        cone_h = (self.r - self.h)
        cone_r =  np.sqrt(self.r**2 - cone_h**2)
        self.cw = ConeWall(radius=cone_r, height=cone_h)

    @property
    def vol(self):
        return NotImplemented

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        return NotImplemented

    @property
    def h(self):
        # Returns cap height
        return (self.sr * self.r)/(2*np.pi)

    @property
    def theta(self):
        # Returns half the top angle of cone
        return np.arccos((self.r - self.h)/self.r)

    @property
    def area(self):
        return 4*np.pi*self.r**2 - self.sr*self.r**2 + self.cw.area

    @area.setter
    def area(self, area):
        assert (area > 0), f"Area should be positive, got: {area}"
        a = 1 - self.sr/(2*np.pi)
        self.r = np.sqrt(area/(4*np.pi - self.sr + np.pi*np.sqrt(1 - a**2)))
        self.cw.h = (self.r - self.h)
        self.cw.r = np.sqrt(self.r**2 - self.cw.h**2)

    def sample(self, n):
        super(ReflexKugel, self).sample(n)
        samples = np.empty((n,3))
        n_sphere = round(n*((np.pi*4 - self.sr)*self.r**2)/(self.area))

        for i in np.arange(n_sphere):
            while True:
                xyz = self.rng.normal(size=3)
                xyz = (self.r/np.linalg.norm(xyz))*xyz
                phi = np.arccos(xyz[2]/(np.linalg.norm(xyz)*1))
                if phi > self.theta:
                    samples[i,:] = xyz
                    break

        # Cone wall samples
        samples[n_sphere:,:] = self.cw.sample(n - n_sphere)
        return samples
