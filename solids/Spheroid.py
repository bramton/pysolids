import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid

class Spheroid(BaseSolid):
    def __init__(self, a=1, c=2):
        self.a = a
        self.c = c
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
        # https://en.wikipedia.org/wiki/Spheroid
        a = self.a
        c = self.c
        e = np.sqrt((1+0j) - ((c**2)/(a**2)))
        S = 2*np.pi*a**2 + ((np.pi*c**2)/e)*np.log((1 + e)/(1 - e))
        return np.real(S)

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        a = self.a
        c = self.c
        q = a/c # Ratio
        e = np.sqrt((1+0j) - 1/(q**2))
        self.c = np.real(np.sqrt(area/(2*np.pi*(q**2) + (np.pi/e)*np.log((1+e)/(1-e)))))
        self.a = q*self.c

    # https://math.stackexchange.com/a/982833/1071683
    def sample(self, n):
        super(Spheroid, self).sample(n)
        samples = np.empty((n,3))
        a = self.a
        c = self.c
        mu_max = a*c


        # Loop probably not needed
        for i in np.arange(n):
            while True:
                xyz = self.rng.normal(size=3)
                xyz = xyz/np.linalg.norm(xyz)
                mu_xyz = np.sqrt((a*c*xyz[1])**2 + (a*a*xyz[2])**2 + (a*c*xyz[0]) + (0+0j))
                break
                if self.rng.random() < np.linalg.norm(mu_xyz)/mu_max:
                    break
            samples[i,:] = np.multiply(xyz,[a,a,c])

        return samples
