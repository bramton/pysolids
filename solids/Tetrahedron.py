import numpy as np
from numpy.random import default_rng
from solids.BaseSolid import BaseSolid

class Tetrahedron(BaseSolid):
    def __init__(self, length=1):
        self.L = length
        self.rng = default_rng()

    @property
    def vol(self):
        return (self.L**3)/(6*np.sqrt(2))

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        self.L = (6*np.sqrt(2)*vol) ** (1/3)

    @property
    def area(self):
        return np.sqrt(3)*self.L ** 2

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        self.L = np.sqrt(area/np.sqrt(3))

    def vertices(self):
        # Vertices of tetrahedron are subset of cube corners
        v = np.zeros((4,3))
        v[0,:] = [0, 0, 0]
        v[1,:] = [1, 0, 1]
        v[2,:] = [1, 1, 0]
        v[3,:] = [0, 1, 1]

        v = v*(self.L/np.sqrt(2)) # Set the edge length
        v = v - (self.L ** 2)/2 # Centre tetrahedron

        return v

    def faces(self):
        f = np.zeros((4,3), dtype='int')
        f[0,:] = [0, 3, 1]
        f[1,:] = [0, 1, 2]
        f[2,:] = [0, 3, 2]
        f[3,:] = [1, 3, 2]

        return f

    def sample(self, n):
        super(Tetrahedron, self).sample(n)
        samples = np.empty((n,3))
        v = self.vertices()
        f = self.faces()
        #print(f)
        for i in np.arange(n):
            r1,r2 = self.rng.random(2)
            r1 = np.sqrt(r1)
            #print("{:f} {:f}".format(r1, r2))
            face = self.rng.choice(4)
            xyz = v[f[face]]
            #print(xyz)
            #print(xyz.shape)
            rv = np.array([[1-r1, r1*(1-r2), r2*r1]]).T
            #print(rv)
            #print(rv.shape)
            xyz = np.matmul(xyz.T, rv)
            #print(xyz)
            #print(xyz.shape)
            #print("----------------")
            samples[i,:] = xyz[:,0]

        return samples
