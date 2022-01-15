from __future__ import annotations
import re
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




def match_position(line: str) -> Tuple[np.ndarray, np.ndarray]:
    FLOAT_GROUP: Callable[[str], str] = lambda name: f'(?P<{name}>-?\d+\.\d+)'

    PATTERN = r'Piezo\[' + \
        FLOAT_GROUP("Px") + ' ' + \
        FLOAT_GROUP("Py") + ' ' + \
        FLOAT_GROUP("Pz") + \
    r'\]\tcMUT\[ ' + \
        FLOAT_GROUP("Cx") + ' ' + \
        FLOAT_GROUP("Cy") + '  ' + \
        FLOAT_GROUP("Cz") + \
    r'\]'

    pattern = re.compile(PATTERN)
    match = pattern.match(line)

    if match is None:
        raise Exception(f'Could not match expression "{line}"')
     
    p_array = np.array([float(match.group('Px')), float(match.group('Py')), float(match.group('Pz'))])
    c_array = np.array([float(match.group('Cx')), float(match.group('Cy')), float(match.group('Cz'))])
    return (p_array, c_array)

def load_position_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(file_path, 'r') as f:
        lines = f.readlines()

        num_lines = len(lines)
        piezo_pos = np.empty([3, num_lines])
        cmut_pos = np.empty([3, num_lines])

        for i, line in enumerate(lines):
            p_pos, c_pos = match_position(line)
            piezo_pos[:, i] = p_pos
            cmut_pos[:, i] = c_pos

        return piezo_pos, cmut_pos


if __name__ == '__main__':
    print(load_position_data('./data/toastgitter/')[:4])