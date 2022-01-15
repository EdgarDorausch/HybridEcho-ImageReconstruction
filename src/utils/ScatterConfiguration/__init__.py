from __future__ import annotations
from typing import *
if TYPE_CHECKING:
    pass
import numpy as np
import plotly.graph_objects as go

class ScatterConfiguration:

    def __init__(self) -> None:
        self.pos_scatter: List[List[float]] = []

    def add_circle(self, r_vec: np.ndarray, midpoint: np.ndarray, num: int):
        
        m_cmplx = midpoint[0]+ 1.0j*midpoint[1]
        r_cmplx = r_vec[0] + 1.0j*r_vec[1]

        pos_elements = np.empty([num, 2])
        rotations = np.exp(2j*np.pi*np.arange(num)/num)
        pos_cmplx = rotations*r_cmplx + m_cmplx

        pos_elements[:, 0] = pos_cmplx.real
        pos_elements[:, 1] = pos_cmplx.imag

        self.pos_scatter.extend(pos_elements.tolist())

    def __array__(self):
        return np.array(self.pos_scatter)

    def get_plot_traces(self) -> Any:
        pos_sc = np.array(self)
        return go.Scatter(
            x=pos_sc[:,0], y=pos_sc[:,1], mode='markers', name='scatter'
        )