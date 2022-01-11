from typing import Optional
import numpy as np
from abc import ABC, abstractmethod
import math

class Simulation2D(ABC):

    def __init__(self, f_samp: float,  c: float, t1: float, t0: float=0.0, mirror_planes: np.ndarray = np.empty([0,3])):
        """Generating simulation data in a 2D coordinate system. (Still 3D physics where used for the simulation itself.)

        Args:
            f_samp (float): sampling frequency. unit=[s]
            c (float): speed of signal. unit=[mm/s]
            t1 (float): simulation stop time. unit=[s]
            t0 (float): simulation start time. unit=[s]
            mirror_planes (np.ndarray): defines reflective planes by the hesse normal form. Each row has the structure: (n_x, n_z, d), where n is the normal vector of the plane and d is origin distance to the plane. shape=(num_planes, 3). unit=[mm]
        """
        if t0 >= t1:
            raise Exception('t0 has to be smaller than t1')

        self.f_samp = f_samp
        self.t0 = t0
        self.t1 = t1
        self.c = c
        self.mirror_planes = mirror_planes
        assert mirror_planes.ndim == 2 and mirror_planes.shape[1] == 3


    @property
    def sig_len(self) -> int:
        """Signal length (i.e. SIG_LEN). unit=none
        """
        return int(self.t_record * self.f_samp)

    @property
    def t_record(self) -> float:
        """duration of recording. unit=[s]
        """
        return self.t1 - self.t0

    @property
    def num_planes(self) -> float:
        """Number of mirror planes. unit=none
        """
        return self.mirror_planes.shape[0]

    @abstractmethod
    def simulate(self, pos_tx_rx: np.ndarray, pos_scatter: np.ndarray) ->  np.ndarray:
        """Runs simulation

        Args:
            pos_tx_rx (np.ndarray): defining the positions of the TX and RX elements. Each row has the form: (tx_x, tx_z, rx_x, rx_z). shape=(NUM_REL, 4) & unit=[mm]
            pos_scatter (np.ndarray): defining the positions of the scatters. shape=(NUM_SCATTER, 2) & unit=[mm]

        Returns:
            np.ndarray: resulting timelines from simulation. shape=(NUM_REL, SIG_LEN) & unit=[s]
        """

        raise NotImplementedError()

    @property
    def time_space(self) -> np.ndarray:
        return np.arange(self.sig_len)/self.f_samp + self.t0

    def get_mirror_plane_y(self, x: np.ndarray):
        y = (self.mirror_planes[:, 2, None] - self.mirror_planes[:, 0, None]*x[None,:])/self.mirror_planes[:, 1, None]
        return y







class NNISimulation(Simulation2D):
    """Uses nearest neighbor interpolition while simulating"""

    def __init__(self, f_samp: float, c: float, t1: float, t0: float = 0):
        super().__init__(f_samp, c, t1, t0=t0)

    def simulate(self, pos_tx_rx: np.ndarray, pos_scatter: np.ndarray) -> np.ndarray:
        time_series = np.zeros([pos_tx_rx.shape[0], self.sig_len])

        for i, row in enumerate(pos_tx_rx):
            for scat in pos_scatter:
                r1 = np.linalg.norm(row[:2]-scat)
                r2 = np.linalg.norm(row[2:]-scat)

                idx =int((r1+r2)/self.c*self.f_samp - self.t0)
                
                # if index is in bounds
                if 0 <= idx < self.sig_len:
                    time_series[i, idx] += 1/(r1*r2)
        
        return time_series





class LinISimulation(Simulation2D):
    """Uses linear interpolition while simulating"""

    def simulate(self, pos_tx_rx: np.ndarray, pos_scatter: np.ndarray) -> np.ndarray:
        time_series = np.zeros([pos_tx_rx.shape[0], self.sig_len])

        for i, row in enumerate(pos_tx_rx):
            for scat in pos_scatter:
                r1 = np.linalg.norm(row[:2]-scat)
                r2 = np.linalg.norm(row[2:]-scat)

                t_shift = (r1+r2)/self.c*self.f_samp - self.t0
                i0 = math.floor(t_shift)
                
                # if index is in bounds
                if 0 <= i0 < self.sig_len:
                    time_series[i, i0] += 1/(r1*r2) * ((i0+1)-t_shift)

                # if index is in bounds
                if 0 <= i0+1 < self.sig_len:
                    time_series[i, i0+1] += 1/(r1*r2) * (t_shift-i0)

        for plane_vec in self.mirror_planes:
            for i, row in enumerate(pos_tx_rx):
                d_tx = np.dot(plane_vec[:2], row[:2]) - plane_vec[2]
                d_rx = np.dot(plane_vec[:2], row[2:])- plane_vec[2]
                
                if np.sign(d_tx) == np.sign(d_rx):
                    tx_new = row[:2] - 2*d_tx*plane_vec[:2]
                    r = np.linalg.norm(tx_new-row[2:])

                    t_shift = r/self.c*self.f_samp - self.t0
                    i0 = math.floor(t_shift)
                    # if index is in bounds
                    if 0 <= i0 < self.sig_len:
                        time_series[i, i0] += 1/(r) * ((i0+1)-t_shift)

                    # if index is in bounds
                    if 0 <= i0+1 < self.sig_len:
                        time_series[i, i0+1] += 1/(r) * (t_shift-i0)


        
        return time_series