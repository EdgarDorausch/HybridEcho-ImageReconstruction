import numpy as np
import math
from typing import *
from tqdm import tqdm


class Scene2d():

    def __init__(self, u: np.ndarray, v: np.ndarray, o: np.ndarray, res_u: int, res_v: int, pos_tx_rx: np.ndarray):
        """Holds information about the image plane and the RX and TX positions

        Args:
            u (np.ndarray): vector defining the horizontal demansion of the image(-plane). shape=(2). unit=[mm]
            v (np.ndarray): vector defining the vertical demansion of the image(-plane). shape=(2). unit=[mm]
            o (np.ndarray): position of pixel (0,0) of the image. shape=(2). unit=[mm]
            res_u (int): resolution of the image in horizontal direction. Must be bigger that one.
            res_v (int): resolution of the image in vertical direction. Must be bigger that one.
            pos_tx_rx (np.ndarray): array holding the positions of the RX and TX elements for each relation.
            Each row is (tx_x, tx_z, rx_x, rx_z). shape=(num_rel, 4). unit=[mm]
        """

        self.o = o
        assert o.shape == (2,)

        self.u = u
        assert u.shape == (2,)

        self.v = v
        assert v.shape == (2,)

        self.res_u = res_u
        assert res_u > 1

        self.res_v = res_v
        assert res_v > 1

        self.pos_tx_rx = pos_tx_rx
        assert self.pos_tx_rx.ndim == 2
        assert self.pos_tx_rx.shape[1] == 4

    @property
    def num_rel(self):
        return self.pos_tx_rx.shape[0]

    def get_separated_rx_tx_pos(self):
        # with duplicates
        pos_tx_wd = self.pos_tx_rx[0:2]
        pos_rx_wd = self.pos_tx_rx[2:4]

        return np.unique(pos_tx_wd, axis=0), np.unique(pos_rx_wd, axis=0)

    def construct_image_grid(self) -> np.ndarray:
        """Generates a grid with the position of every pixel.

        Returns:
            np.ndarray: The grid. shape=(res_u,res_v,2). unit=[mm]
        """
        u_scale = np.linspace(0, 1, self.res_u)[: , None, None]
        v_scale = np.linspace(0, 1, self.res_v)[None, :, None]

        grid = (self.u[None, None, :]*u_scale) + (self.v[None, None, :]*v_scale) + self.o[None, None, :]
        return grid

    def get_imaging_dist_interval(self) -> Tuple[float, float]:
        """Returns the lowest und biggest travel distance, of the points in the image plane.

        Returns:
            Tuple[float, float]: The interval: (low,upper). unit=[mm]
        """
        grid = self.construct_image_grid()

        distances = np.linalg.norm(self.pos_tx_rx[None,None,:,0:2]-grid[:,:,None, :], axis=3) + \
                    np.linalg.norm(self.pos_tx_rx[None,None,:,2:4]-grid[:,:,None, :], axis=3)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        return min_dist, max_dist


class DAS2d():

    def __init__(self, scene: Scene2d, sig: np.ndarray, f_samp: float,  c: float):
        """Generate Delay-And-Sum image

        Args:
            scene (Scene2d): scene object
            sig (np.ndarray): signal. shape=(num_rel, sig_len)
            f_samp (float): sampling frequency. unit=[Hz]
            c (float): wave speed. unit=[mm/s]
        """
        self.scene = scene
        self.sig = sig
        assert sig.ndim == 2
        assert sig.shape[0] == scene.num_rel

        self.f_samp = f_samp
        self.c = c

    @property
    def res_u(self):
        return self.scene.res_u

    @property
    def res_v(self):
        return self.scene.res_v

    @property
    def sig_len(self) -> int:
        return self.sig.shape[1]

    @property
    def dt(self) -> float:
        return 1/self.f_samp

    def get_time_space(self):
        """Physical time value for each signal sample.

        Returns:
            np.ndarray: sequence of time points. shape=(sig_len). unit=[s]
        """
        return np.arange(self.sig_len)/self.f_samp

    def get_imaging_time_interval(self) -> Tuple[float, float]:
        min_dist, max_dist = self.scene.get_imaging_dist_interval()
        return (min_dist / self.c), (max_dist / self.c)

    def get_imaging_sample_interval(self) -> Tuple[int, int]:
        min_t, max_t = self.get_imaging_time_interval()

        lower_sample_idx = math.floor(min_t * self.f_samp) # inclusive
        upper_sample_idx = math.ceil(max_t * self.f_samp)+1 # not inclusive

        return lower_sample_idx, upper_sample_idx

    def run(self) -> np.ndarray:
        min_sample_idx, max_sample_idx = self.get_imaging_sample_interval()

        dr = self.dt * self.c
        grid = self.scene.construct_image_grid()

        def image_tx_rx(tx_pos, rx_pos, rel):
            D_rx = np.linalg.norm(grid - rx_pos, axis= 2)
            D_tx = np.linalg.norm(grid - tx_pos, axis= 2)
            D_total = D_rx+D_tx

            # Transposed matrix
            Mt = np.zeros([self.res_u*self.res_v, max_sample_idx-min_sample_idx])

            for i in range(min_sample_idx, max_sample_idx):
                ff = np.ravel( ((D_total) < (i+1)*dr) * ((D_total) > i*dr))
                rsq = np.ravel(D_total**2)

                # Mt computes the approx. inverse
                Mt[:, i-min_sample_idx] = rsq*ff


            p = np.dot(Mt, self.sig[rel,min_sample_idx:max_sample_idx])
            r = p.reshape([self.scene.res_u, self.scene.res_v])
            return r

        ri = np.zeros([self.res_u, self.res_v])
        for i in tqdm(range(self.scene.num_rel)):
                rii = image_tx_rx(self.scene.pos_tx_rx[None,None,i, 0:2], self.scene.pos_tx_rx[None,None,i, 2:4], i)
                ri += rii
        
        return ri

