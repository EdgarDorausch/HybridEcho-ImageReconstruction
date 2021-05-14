import numpy as np
from abc import ABC, abstractmethod

class Kernel(ABC):

    @abstractmethod
    def get_val(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_width(self) -> float:
        raise NotImplementedError

class GaussKernel(Kernel):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def get_val(self, t: np.ndarray) -> np.ndarray:
        return np.exp(-(t/self.sigma)**2)

    def get_width(self) -> float:
        prec = 1e-20
        return np.sqrt(-np.log(prec))*self.sigma