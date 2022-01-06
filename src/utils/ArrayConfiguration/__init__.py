import numpy.typing as npt
import numpy as np

class ArrayConfiguration:

    def __init__(self, rx_pos: npt.ArrayLike, tx_pos: npt.ArrayLike) -> None:
        self.rx_pos = np.array(rx_pos, dtype=np.float64)
        self.tx_pos = np.array(tx_pos, dtype=np.float64)

        rx_shape = self.rx_pos.shape
        tx_shape = self.tx_pos.shape
        assert len(rx_shape) == 2 and rx_shape[1] == 2, 'rx_pos should be of shape (*, 2)'
        assert len(tx_shape) == 2 and tx_shape[1] == 2, 'tx_pos should be of shape (*, 2)'

    def get_combined(self) -> np.ndarray:
        """
        Computes the cartesian product of two position spaces 
        """
        i = 0
        merged = np.empty([self.rx_pos.shape[0]*self.tx_pos.shape[0], 4])
        for prx in self.rx_pos:
            for ptx in self.tx_pos:
                merged[i,:2] = prx
                merged[i,2:] = ptx
                i+=1
        return merged

    def __array__(self):
        return self.get_combined()