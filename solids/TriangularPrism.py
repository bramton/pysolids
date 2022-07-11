import numpy as np
from numpy.random import default_rng
from .BaseSolid import BaseSolid

class TriangularPrism(BaseSolid):
    def __init__(self, edge=1, height=2):
        self.L = edge
        self.r = height/edge # ratio
        self.rng = default_rng()

        # Define faces of TriangularPrism
        self.faces = np.zeros((8,3), dtype='int')
        self.faces[0,:] = [0, 1, 2]
        self.faces[1,:] = [3, 4, 5]

        self.faces[2,:] = [0, 1, 3]
        self.faces[3,:] = [1, 4, 3]

        self.faces[4,:] = [2, 1, 5]
        self.faces[5,:] = [1, 4, 5]

        self.faces[6,:] = [2, 0, 5]
        self.faces[7,:] = [0, 3, 5]

    @property
    def vol(self):
        h = (self.r*self.L)
        return np.sqrt(3)*0.25*h*self.L**2

    @vol.setter
    def vol(self, vol):
        assert vol > 0, f"Volume should be positive, got: {vol}"
        self.L = ((vol*4)/(np.sqrt(3)*self.r)) ** (1/3)

    @property
    def area(self):
        h = (self.r*self.L)
        return np.sqrt(3)*0.5*self.L**2 + 3*self.L*h

    @area.setter
    def area(self, area):
        assert area > 0, f"Area should be positive, got: {area}"
        self.L = np.sqrt((2*area)/(np.sqrt(3) + 6*self.r))

    def vertices(self):
        v = np.zeros((6,3))
        L = self.L
        h = self.L*self.r
        v[0,:] = [0, 0.25*L*np.sqrt(3), h/2]
        v[1,:] = [L/2, -0.25*L*np.sqrt(3), h/2]
        v[2,:] = [-L/2, -0.25*L*np.sqrt(3), h/2]
        v[3:,:] = v[0:3] - [0,0,h]

        return v

    def face_areas(self):
        f = self.faces
        v = self.vertices()
        tmp = v[f.flatten()]
        tmp = tmp.reshape((-1,3,3)) # x,y,z will be in dim 2
        tmp = np.cross(tmp[:,1,:]-tmp[:,0,:],tmp[:,2,:]-tmp[:,0,:])
        tmp = 0.5*np.linalg.norm(tmp, axis=1)
        return tmp

    def sample(self, n):
        super(TriangularPrism, self).sample(n)

        areas = self.face_areas()
        samples = np.empty((n,3))

        rf = self.rng.choice(self.faces, size=n, p=areas/np.sum(areas))
        # Probably can do without loop. Too lazy atm.
        for i,f in enumerate(rf):
            r1,r2 = self.rng.random(2)
            r1 = np.sqrt(r1)
            xyz = self.vertices()[f]
            rv = np.array([[1-r1, r1*(1-r2), r2*r1]]).T
            xyz = np.matmul(xyz.T, rv)
            samples[i,:] = xyz[:,0]

        return samples
