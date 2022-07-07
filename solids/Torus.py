import numpy as np
from numpy.random import default_rng
from solids.BaseSolid import BaseSolid

class Torus(BaseSolid):
    def __init__(self, R=2.5, r=1):
        self.r = r
        self.R = R
        self.rng = default_rng()

    @property
    def vol(self):
        return 2*np.pi*np.pi*self.R*self.r**2

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        a = self.R/self.r
        self.r = (vol/(2*a*np.pi**2))**(1/3)
        self.R = a*self.r

    @property
    def area(self):
        return 4*self.R*self.r*np.pi**2

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        a = self.R/self.r
        self.r = np.sqrt(area/(4*a*np.pi**2))
        self.R = a*self.r

    # Credits: https://math.stackexchange.com/questions/2017079/uniform-random-points-on-a-torus
    def sample(self, n):
        super(Torus, self).sample(n)
        samples = np.empty((n,3))
        R = self.R
        r = self.r
        for i in np.arange(n):
            while True:
                u,v,w = self.rng.random(3)
                theta = 2*np.pi*u
                phi = 2*np.pi*v
                if w <= (R + r*np.cos(theta))/(R + r):
                    break
            x = (R + r*np.cos(theta))*np.cos(phi)
            y = (R + r*np.cos(theta))*np.sin(phi)
            z = r*np.sin(theta)
            samples[i,:] = [x,y,z]

        return samples
