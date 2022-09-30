import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid

class PyramidWall(BaseSolid):
    def __init__(self, a=2, h=1):
        self.a = a
        self.h = h
        self.rng = default_rng()
        self.faces = np.zeros((4,3), dtype='int')
        self.faces[0,:] = [0,1,4]
        self.faces[1,:] = [1,2,4]
        self.faces[2,:] = [2,3,4]
        self.faces[3,:] = [3,0,4]

    @property
    def vol(self):
        return NotImplemented

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        return NotImplemented

    @property
    def area(self):
        return (self.a**2)*np.sqrt(2)

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        a = self.h/self.a
        self.a = area/(2*np.sqrt(2))
        self.h = a*self.a

    def vertices(self):
        a = self.a/2
        v = np.zeros((5,3))
        v[0,:] = [-a, -a, 0]
        v[1,:] = [a, -a, 0]
        v[2,:] = [a, a, 0]
        v[3,:] = [-a, a, 0]
        v[4,:] = [0, 0, self.h]

        return v

    def sample(self, n):
        super(PyramidWall, self).sample(n)
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
