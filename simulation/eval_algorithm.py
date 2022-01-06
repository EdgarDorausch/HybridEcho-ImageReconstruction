#%%
from utils.Simulation.simulation2d import LinISimulation
from utils.ArrayConfiguration import ArrayConfiguration
from utils.ScatterConfiguration import ScatterConfiguration
import numpy as np
from utils.Algorithm.algorithm2d import Scene2d, DAS2d

import plotly.graph_objects as go
from plotly.subplots import make_subplots

f_samp=1e6
c=1_484_000.0

# ------------------------- Generate Simulation Data ------------------------- #

ac = ArrayConfiguration([
    [ 0, 100],
    [10, 100],
    [20, 100]
],
[
    [ -5, 100],
    [ 5, 100],
    [15, 100],
    [25, 100],
])

sc = ScatterConfiguration()
sc.add_circle([10, 0], [10,60], 100)

pos_rx_tx = np.array(ac)
pos_scatter = np.array(sc)

sim = LinISimulation(
    f_samp=f_samp,
    c=c,
    t1=2e-4
)
t = sim.simulate(pos_rx_tx, pos_scatter)

# ----------------------------- Run DAS Algorithm ---------------------------- #

#%%

scene = Scene2d(
    o=np.array([-10.0, 40.0]),
    u=np.array([ 40.0,  0.0]),
    v=np.array([  0.0, 40.0]),
    res_u=150,
    res_v=150,
    pos_tx_rx=pos_rx_tx
)

# %%
alg = DAS2d(scene=scene, sig=t, f_samp=f_samp, c=c)
# %%

im = alg.run()
# %%
import matplotlib.pyplot as plt
plt.imshow(im)
# %%
