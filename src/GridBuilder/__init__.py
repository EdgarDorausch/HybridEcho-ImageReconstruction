import numpy as np

class GridBuilder:

    def __init__(self, res_x: int, res_y: int, res_z: int=1):
        self.res_x = res_x
        self.res_y = res_y
        self.res_z = res_z


    def build(self, center: bool=True, dtype=np.float32) -> np.ndarray:
        d = np.zeros([3, self.res_x, self.res_y, self.res_z], dtype=dtype)
        d[0,:,:] = np.arange(0., stop=self.res_x, step=1.)[:, np.newaxis, np.newaxis]
        d[1,:,:] = np.arange(0., stop=self.res_y, step=1.)[np.newaxis, :, np.newaxis]
        d[2,:,:] = np.arange(0., stop=self.res_z, step=1.)[np.newaxis, np.newaxis, :]

        if center:
            d-=np.array([self.res_x/2, self.res_y/2, self.res_z/2])[:, np.newaxis, np.newaxis, np.newaxis]

        return d


class ImageGridBuilder:
    """
    Helper class designed to constract a 2D grid in 3D space.
    The element (0,0) of the grid is mapped to an given origin vector.
    The vectors x and y are used to construct the remaining elements of the grid given by:

    grid[i,j] = origin + i*x/(res_x-1) + j*y/(res_y-1)

    where res_x and res_y are the grid sizes in x and y direction.

    ** Note that x is scaled by 1/(res_x) so the res_x-th element of the grid in x direction (indexed by res_x-1) will result full vector x,
    which is added in the formula above. The same arguments hold for y **
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, origin: np.ndarray):
        """
        x, y, origin - Shape: [3]
        """
        assert x.shape == (3,)
        assert y.shape == (3,)
        assert origin.shape == (3,)


        # these vectors have shape [3,1,1]
        self.x = x[:,None, None].astype(float)
        self.y = y[:,None, None].astype(float)
        self.origin = origin[:,None, None].astype(float)

    def build(self, res_x: int, res_y: int) -> np.ndarray:
        """
        return - Shape: [3, res_x, res_y]
        """
        grid = np.full([3, res_x, res_y], self.origin)
        lx = np.linspace(0.0, 1.0, num=res_x) 
        ly = np.linspace(0.0, 1.0, num=res_y) 

        grid += lx[None, :, None] * self.x
        grid += lx[None, None, :] * self.y

        return grid

    def get_x_y_angle(self):
        norm_dp = np.dot(self.x, self.y)/(np.linalg.norm(self.x)*np.linalg.norm(self.y))
        angle = np.arcos(norm_dp)
        return angle