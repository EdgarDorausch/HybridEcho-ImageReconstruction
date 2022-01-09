#%%
from utils.Simulation.simulation2d import LinISimulation
from utils.ArrayConfiguration import ArrayConfiguration
from utils.ScatterConfiguration import ScatterConfiguration
import numpy as np
from utils.Algorithm.algorithm2d import Scene2d, DAS2d

import plotly.graph_objects as go
import plotly.express as px

f_samp=2e6
c=1_484_000.0

# ------------------------- Generate Simulation Data ------------------------- #

ac = ArrayConfiguration([
    [ 0, 0],
    [10, 0],
    [20, 0]
],
[
    [ -5, 0],
    [ 5, 0],
    [15, 0],
    [25, 0],
])

sc = ScatterConfiguration()
sc.add_circle([10, 0], [10, 60], 100)

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
    res_u=250,
    res_v=350,
    pos_tx_rx=pos_rx_tx
)

#%%
trs = scene.get_plot_traces() + [sc.get_plot_traces()]
fig = go.Figure(data=trs)
fig.update_yaxes(scaleratio=1, scaleanchor='x')
fig.show()

# %%
alg = DAS2d(scene=scene, sig=t, f_samp=f_samp, c=c)
# %%

im = alg.run()
# %%
import matplotlib.pyplot as plt
plt.imshow(im)
# %%
grid = scene.construct_image_grid()
# %%

x_space, y_space = scene.get_image_xy_space()

fig = px.imshow(
    im.T,
    x=x_space,
    y=y_space,
    aspect=None,
    color_continuous_scale='gray')

fig.add_trace(sc.get_plot_traces())

fig.add_trace(go.Scatter(x=pos_rx_tx[:,0], y=pos_rx_tx[:,1], mode='markers'))
fig.show()

# %%
