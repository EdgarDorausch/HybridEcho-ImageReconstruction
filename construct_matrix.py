from __future__ import annotations
from typing import *
if TYPE_CHECKING:
    pass
from numpy.lib.function_base import iterable
from src.Kernel import GaussKernel
from src.EllipsoidRenderer import EllipsoidRenderer
from src.GridBuilder import ImageGridBuilder
from read_data import load_position_data
import numpy as np
import math
import matplotlib.pyplot as plt

def true_idx(it: Iterable[bool]) -> Generator[int]:
    for i, x in enumerate(it):
        if x:
            yield i


SPEED_OF_SOUND = 1_484_000.0 # [mm/s]
SAMPLE_RATE = 2_000_000.0 # [1/s]
ECHO_START_TIME = 0.0 # [s]

data = np.load('data/toastgitter/all_data.npy') # Shape [STEPS, NUM_REL]
p_pos, c_pos = load_position_data('data/toastgitter/LOGFILE_unified_coordinates.txt')
print(data.shape)


selected_p = np.apply_along_axis(
    lambda pos: 
        np.allclose(pos, np.array([385.0, -110.0, 198.0])),
    0,
    p_pos)

selected_c = np.apply_along_axis(
    lambda pos: any(
        np.allclose(pos, np.array([1.0*k+311.1, -110.0, 198.0]))
        for k in range(0,61,2)
        ),
    0,
    c_pos)

selected = np.logical_and(selected_p, selected_c)

NUM_RELATION = np.sum(selected)

u = np.array([50,0,0], dtype=float)
v = np.array([0,0,-50], dtype=float)
o = np.array([350, -110, 25], dtype=float)

res_x = 32
res_y = 32
igb = ImageGridBuilder(u, v, o) #image will be flipped along y axis!
grid = igb.build(32,32).reshape([3,-1])
NUM_PIXEL = res_x*res_y


gauss = GaussKernel(1.0)
print(grid.shape)

# compute 
distances = np.linalg.norm(p_pos[:,selected,None]-grid[:,None,:], axis=0) + \
            np.linalg.norm(c_pos[:,selected,None]-grid[:,None,:], axis=0)

min_dist = np.min(distances)
max_dist = np.max(distances)

lower_sample_idx = math.floor((ECHO_START_TIME + (min_dist / SPEED_OF_SOUND)) * SAMPLE_RATE) # inclusive
upper_sample_idx = math.ceil((ECHO_START_TIME + (max_dist / SPEED_OF_SOUND)) * SAMPLE_RATE)+1 # not inclusive

NUM_SAMPLES = upper_sample_idx - lower_sample_idx
print(lower_sample_idx, upper_sample_idx)

#construct matrix
M = np.empty([NUM_SAMPLES*NUM_RELATION, NUM_PIXEL])

for rel_idx, p_c_pos in enumerate(np.concatenate([p_pos[:,selected], c_pos[:,selected]], axis=0).T):

    ts = np.arange(lower_sample_idx, upper_sample_idx)/SAMPLE_RATE - ECHO_START_TIME #TODO: ???
    # print(np.concatenate([p_pos[:,selected], c_pos[:,selected]], axis=0))
    er = EllipsoidRenderer(p_c_pos[0:3], p_c_pos[3:6], gauss)
    foo = er.render(grid[:,: None, None], ts).reshape([NUM_SAMPLES, NUM_PIXEL])
    M[rel_idx*NUM_SAMPLES:(rel_idx+1)*NUM_SAMPLES,:] = foo


print(M.shape)

Minv= np.linalg.pinv(M)

# flatten vector (column major)
selected_data = data[lower_sample_idx:upper_sample_idx, selected].flatten(order='F')

x_hat = Minv @ selected_data

x_hat = x_hat.reshape([res_x, res_y])

plt.imshow(x_hat)
plt.show()