import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid

class ConeWall(BaseSolid):
    def __init__(self, radius=1, height=2):
        self.r = radius
        self.h = height
        self.rng = default_rng()

    @property
    def area(self):
        return np.pi*self.r*np.sqrt(self.r**2 + self.h**2)

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        a = self.h/self.r
        self.r = np.sqrt(area/(np.pi*np.sqrt(1+a**2)))
        self.h = a*self.r

    def sample(self, n=1):
        super(ConeWall, self).sample(n)
        samples = np.empty((n,3))
        h = self.h # Height of cone
        L = np.sqrt(self.r**2 + h**2)

        # Assuming cone has nose at 0,0 and upwarts in the y-direction
        theta = (2*np.pi*self.r)/L # Angle of unfolded cone
        alpha = np.arcsin(self.r/L) # Half angle of cross-cut

        ymax = L
        if theta > np.pi:
            xmax = L
            ymin = -1*np.sin(0.5*(theta - np.pi))*L
        else:
            xmax = np.cos(0.5*(np.pi - theta))*L
            ymin = 0

        # Sample 'wall' of cone
        for i in np.arange(n):
            # Rejection sampling for unfolded cone surface
            while True:
                x = self.rng.uniform(-xmax, xmax)
                y = self.rng.uniform(ymin, ymax)
                l = np.sqrt(x**2 + y**2)
                if (l > L):
                    continue
                phi = np.arccos(np.dot([x,y], [0,1])/(np.linalg.norm([x,y])*1))
                if (phi > 0.5*theta):
                    continue
                break

            angle = np.pi*(phi/(0.5*theta))
            angle = np.sign(x)*angle + np.pi
            r = (theta*l)/(2*np.pi)

            x = np.cos(angle)*r
            y = np.sin(angle)*r
            z = np.cos(alpha)*l
            samples[i,:] = [x,y,z]

        return samples
