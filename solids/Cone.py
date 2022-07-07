import numpy as np
from numpy.random import default_rng
from solids.BaseSolid import BaseSolid

class Cone(BaseSolid):
    def __init__(self, radius=1, height=2):
        self.r = radius
        self.a = height/radius
        self.rng = default_rng()

    #@property
    #def vol(self):
    #    return 

    #@vol.setter
    #def vol(self, vol):
    #    assert vol > 0, f"Volume should be positive, got: {vol}"
    #    self.r = 

    @property
    def area(self):
        return (1 + np.sqrt(1 + self.a**2))*np.pi*self.r**2

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        self.r = np.sqrt(area/(np.pi*(1 + np.sqrt(1 + self.a**2))))

    def sample(self, n):
        super(Cone, self).sample(n)
        samples = np.empty((n,3))
        h = self.r*self.a # Height of cone
        L = np.sqrt(self.r**2 + h**2)
        theta = (2*np.pi*self.r)/L # Angle of unfolded cone
        #print("theta: {:.2f}".format(np.degrees(theta)))
        alpha = np.arcsin(self.r/L) # Half angle of cross-cut
        if theta > 0.5*np.pi:
            xmin = np.cos(theta)*L
            ymax = L
        else:
            xmin = 0
            ymax = np.sin(theta)*L
        n_base = round(((np.pi*self.r**2)/self.area)*n)

        # Sample base of cone
        for i in np.arange(n_base):
            # Rejection sampling
            while True:
                xy = self.rng.uniform(-self.r, self.r, size=2)
                if np.linalg.norm(xy) <= self.r:
                    break
            samples[i,:] = [xy[0], xy[1], h]

        # Sample 'wall' of cone
        for i in np.arange(n_base, n):
            # Rejection sampling for unfolded cone surface
            while True:
                x = self.rng.uniform(xmin, L)
                y = self.rng.uniform(0, ymax)
                l = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                if (phi > theta) or (l > L):
                    continue
                else:
                    break

            angle = (phi/theta)*2*np.pi
            r = (theta*l)/(2*np.pi)

            x = np.cos(angle)*r
            y = np.sin(angle)*r
            z = np.cos(alpha)*l  # Must be a nicer form
            samples[i,:] = [x,y,z]

        samples[:,2] = samples[:,2] - h/2
        return samples
