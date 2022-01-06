from utils.Simulation.simulation2d import LinISimulation
from utils.ArrayConfiguration import ArrayConfiguration
from utils.ScatterConfiguration import ScatterConfiguration
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    f_samp=100e6,
    c=1_484_000.0,
    t1=2e-4
)
t = sim.simulate(pos_rx_tx, pos_scatter)


fig = make_subplots(rows=1, cols=3,
    specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "scene"}]])

# fig1 = go.Figure(data=go.Scatter(x=sim.time_space, y=t[0]))
# fig1.update_xaxes(title_text='travel time [s]')

fig.add_trace(go.Scatter(x=sim.time_space, y=t[0], name='timeline0'), row=1, col=1)
fig.add_trace(go.Scatter(x=pos_scatter[:,0], y=pos_scatter[:,1], mode='markers', name='scatters'), row=1, col=2)
fig.add_trace(go.Scatter(x=pos_rx_tx[:,0], y=pos_rx_tx[:,1], mode='markers', name='rx'), row=1, col=2)
fig.add_trace(go.Scatter(x=pos_rx_tx[:,2], y=pos_rx_tx[:,3], mode='markers', name='tx'), row=1, col=2)

const = np.full(t.shape[1:2], 0.0)
for i in range(t.shape[0]):
    fig.add_trace(go.Scatter3d(y=sim.time_space, z=t[i], x=const+i, mode='lines'), row=1, col=3)

fig.update_xaxes(
    scaleanchor = "y2",
    scaleratio = 1,
    row=1, col=2)

fig.update_scenes(
    camera_projection_type='orthographic',
    row=1, col=3
)

fig.show()
# fig.write_image("./fig1.pdf", width=1920, height=1080)