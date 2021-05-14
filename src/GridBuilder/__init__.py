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