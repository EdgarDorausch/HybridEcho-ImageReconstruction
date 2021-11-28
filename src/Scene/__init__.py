from __future__ import annotations
from typing import *

if TYPE_CHECKING:
    pass
from GridBuilder import ImageGridBuilder
import numpy as np

class Scene:
    """
    Holds configuration of image plane, piezo / cMut Positions as well as the measured signal
    """

    def __init__(self, u: np.ndarray, v: np.ndarray, o: np.ndarray):

        # Build grid
        self.igb = ImageGridBuilder(u, v, o) #image will be flipped along y axis!
        grid = igb.build(res_u, res_v).reshape([3,-1])
        NUM_PIXEL = res_u*res_v