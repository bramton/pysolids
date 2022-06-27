import numpy as np
from numpy.random import default_rng
from solids.BaseSolid import BaseSolid

class Cylinder(BaseSolid):
    """
    :param ratio: Ratio of the cylinder height to the radius
    """
    def __init__(self, radius=1,  ratio=2):
        self.r = radius
        self.ratio = ratio
        self.rng = default_rng()

    @property
    def vol(self):
        return np.pi*self.r*self.r*self.ratio*self.r

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        self.r = (vol/(np.pi*self.ratio)) ** (1/3)

    @property
    def area(self):
        return 2*np.pi*self.r*self.r*(1 + self.ratio)

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        self.r = np.sqrt(area/(2*np.pi*(1 + self.ratio)))

    def sample(self, n):
        super(Cylinder, self).sample(n)
        weights = np.empty(2)
        weights[0] = (2*np.pi*self.r*self.r)/self.area
        weights[1] = (2*np.pi*self.r*self.r*self.ratio)/self.area
        samples = np.empty((n,3))
        choices = self.rng.choice([0,1], size=n, p=weights)
        for i,choice in enumerate(choices):
            # Top/bottom of cylinder
            if choice==0:
                # Rejection sampling
                while True:
                    xy = self.rng.uniform(-self.r, self.r, size=2)
                    if np.linalg.norm(xy) <= self.r:
                        break
                z = self.rng.choice([-1,1])*0.5*self.r*self.ratio
                samples[i,:] = [xy[0], xy[1], z]
            # Wall of cylinder
            else:
                angle = self.rng.uniform(0, 2*np.pi)
                x = np.cos(angle)*self.r
                y = np.sin(angle)*self.r
                z = self.rng.uniform(-0.5*self.r*self.ratio, 0.5*self.r*self.ratio)
                samples[i,:] = [x, y ,z]

        return samples


