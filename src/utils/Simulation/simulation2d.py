import numpy as np
from abc import ABC, abstractmethod
import math

class Simulation2D(ABC):

    def __init__(self, f_samp: float,  c: float, t1: float, t0: float=0.0):
        """Generating simulation data in a 2D coordinate system. (Still 3D physics where used for the simulation itself.)

        Args:
            f_samp (float): sampling frequency. unit=[s]
            c (float): speed of signal. unit=[mm/s]
            t1 (float): simulation stop time. unit=[s]
            t0 (float): simulation start time. unit=[s]
        """
        if t0 >= t1:
            raise Exception('t0 has to be smaller than t1')

        self.f_samp = f_samp
        self.t0 = t0
        self.t1 = t1
        self.c = c


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


    @abstractmethod
    def simulate(self, pos_tx_rx: np.ndarray, pos_scatter: np.ndarray) ->  np.ndarray:
        """Runs simulation

        Args:
            pos_tx_rx (np.ndarray): defining the positions of the TX and RX elements. shape=(NUM_REL, 4) & unit=[mm]
            pos_scatter (np.ndarray): defining the positions of the scatters. shape=(NUM_SCATTER, 2) & unit=[mm]

        Returns:
            np.ndarray: resulting timelines from simulation. shape=(NUM_REL, SIG_LEN) & unit=[s]
        """

        raise NotImplementedError()

    @property
    def time_space(self) -> np.ndarray:
        return np.arange(self.sig_len)/self.f_samp + self.t0






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

    def __init__(self, f_samp: float, c: float, t1: float, t0: float = 0):
        super().__init__(f_samp, c, t1, t0=t0)

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
                if 0 <= i0 < self.sig_len:
                    time_series[i, i0+1] += 1/(r1*r2) * (t_shift-i0)
                    
        
        return time_series