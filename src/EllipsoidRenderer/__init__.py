from __future__ import annotations
from typing import *

import numpy as np
if TYPE_CHECKING:
    from Kernel import Kernel
    

class EllipsoidRenderer:
    def __init__(self, a: np.ndarray, b: np.ndarray, kernel: Kernel, h: float = 0.001):
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.kernel = kernel
        self.h = h

        self.d = np.linalg.norm(a-b)
        self.w = kernel.get_width()

    def render(self, pos: np.ndarray, T: np.ndarray, normalize: bool=False) -> np.ndarray:
        """
        pos.shape: [3,X,Y,Z]

        T.shape: [NUM_T]

        res.shape: [NUM_T,X,Y,Z]
        """
        pos_shape = pos.shape[1:]

        t = self._t(pos.reshape([3,-1]), T) # Shape: [1, NUM_T, X*Y*Z]
        f = self.kernel.get_val(t)

        if normalize:
            c = self._c(T)[np.newaxis, :, np.newaxis]
            f /= c

        return f.reshape([len(T), *pos_shape])

    def _t(self, pos, T) -> np.ndarray:
        rep_pos = np.repeat(pos[:, np.newaxis, :], len(T), axis=1) # Shape: [3,NUM_T,X*Y*Z]
        # print(rep_pos.shape)
        return np.linalg.norm(rep_pos - self.a[:, np.newaxis, np.newaxis], axis=0) + \
            np.linalg.norm(rep_pos - self.b[:, np.newaxis, np.newaxis], axis=0) - \
            T[np.newaxis, :,np.newaxis]

    def _c(self, T: np.ndarray) -> np.ndarray:
        c_array = T.copy()

        for i in range(len(T)):
            arr = np.arange(start=max(self.d,T[i]-self.w), stop=T[i]+self.w, step=self.h) # Shape: [N]
            c_array[i] = np.sum(self.kernel.get_val(arr-T[i])*(self._A(arr + self.h) - self._A(arr)), axis=0)

        return c_array

    def _A(self, t: float) -> float:
        return np.pi/6*t*(t**2-self.d**2)

    def _A_2D(self, t: float) -> float:
        return np.pi/4*t*np.sqrt(t**2-self.d**2)