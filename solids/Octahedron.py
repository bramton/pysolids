import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid

class Octahedron(BaseSolid):
    def __init__(self, length=1):
        self.L = length
        self.rng = default_rng()
        self.faces = np.zeros((8,3), dtype='int')
        self.faces[0,:] = [0,1,4]
        self.faces[1,:] = [1,2,4]
        self.faces[2,:] = [2,3,4]
        self.faces[3,:] = [3,0,4]
        self.faces[4:,] = self.faces[0:4,:]
        self.faces[4:,2] = 5

    @property
    def vol(self):
        return (1/3)*np.sqrt(2)*(self.L**3)

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        self.L = ((3*vol)/np.sqrt(2)) ** (1/3)

    @property
    def area(self):
        return 2*np.sqrt(3)*self.L ** 2

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        self.L = np.sqrt(area/(2*np.sqrt(3)))

    def vertices(self):
        v = np.zeros((6,3))
        v[0,:] = [-1, 0, 0]
        v[1,:] = [0, 1, 0]
        v[2,:] = [1, 0, 0]
        v[3,:] = [0, -1, 0]
        v[4,:] = [0, 0, -1]
        v[5,:] = [0, 0, 1]

        v = v*(self.L/np.sqrt(2)) # Set the edge length

        return v


    def sample(self, n):
        super(Octahedron, self).sample(n)
        samples = np.empty((n,3))
        v = self.vertices()
        f = self.faces
        for i in np.arange(n):
            r1,r2 = self.rng.random(2)
            r1 = np.sqrt(r1)
            face = self.rng.choice(len(f))
            xyz = v[f[face]]
            rv = np.array([[1-r1, r1*(1-r2), r2*r1]]).T
            xyz = np.matmul(xyz.T, rv)
            samples[i,:] = xyz[:,0]

        return samples
