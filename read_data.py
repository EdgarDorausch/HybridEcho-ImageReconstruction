from __future__ import annotations
from typing import *
if TYPE_CHECKING:
    pass

import numpy as np
from os import path


def load_signal_data(dir_path: str, file_indices: List[int]) -> np.ndarray:
    paths = [path.join(dir_path, f'cir_rx_signal_{idx:05d}.csv') for idx in file_indices]

    first_path = paths[0]
    first_array = np.genfromtxt(first_path, delimiter='\n')

    num_files = len(paths)
    num_samples = first_array.size

    array = np.empty([num_samples, num_files])
    array[:,0] = first_array

    for i, p in enumerate(paths[1:]):
        loaded_array = np.genfromtxt(p, delimiter='\n')

        if loaded_array.size != num_samples:
            print('sample sizes not uniform!')

        array[:,i+1] = loaded_array

        progress = 100.0*(i+2)/num_files
        print(f'Loaded: {progress:3.1f}%', end='\r')

    return array