#%%
from utils.Simulation.simulation2d import LinISimulation
from utils.ArrayConfiguration import ArrayConfiguration
from utils.ScatterConfiguration import ScatterConfiguration
import numpy as np
from utils.Algorithm.algorithm2d import Scene2d, DAS2d

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

f_samp=4e6
c=1_484_000.0

o, u, v = np.array([
    [0.0,  0.0],
    [40.0, 0.0],
    [0.0, 40.0]
])

# ------------------------- Generate Simulation Data ------------------------- #

ac = ArrayConfiguration([
    s*u for s in np.linspace(0.0, 1.0, num=50)
],
[
    s*u for s in np.linspace(0.0, 1.0, num=50)
])

sc = ScatterConfiguration()
sc.add_circle([10, 0], [20, 20], 50)

pos_tx_rx = np.array(ac)
pos_scatter = np.array(sc)

sim = LinISimulation(
    f_samp=f_samp,
    c=c,
    t1=2e-4
)
t = sim.simulate(pos_tx_rx, pos_scatter)




fig = make_subplots(rows=1, cols=3,
    specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "scene"}]])

# fig1 = go.Figure(data=go.Scatter(x=sim.time_space, y=t[0]))
# fig1.update_xaxes(title_text='travel time [s]')

fig.add_trace(go.Scatter(x=sim.time_space, y=t[0], name='timeline0'), row=1, col=1)
fig.add_trace(go.Scatter(x=pos_scatter[:,0], y=pos_scatter[:,1], mode='markers', name='scatters'), row=1, col=2)
fig.add_trace(go.Scatter(x=pos_tx_rx[:,0], y=pos_tx_rx[:,1], mode='markers', name='tx'), row=1, col=2)
fig.add_trace(go.Scatter(x=pos_tx_rx[:,2], y=pos_tx_rx[:,3], mode='markers', name='rx'), row=1, col=2)

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


# ----------------------------- Run DAS Algorithm ---------------------------- #

#%%



scene = Scene2d(
    o=o,
    u=u,
    v=v,
    res_u=350,
    res_v=350,
    pos_tx_rx=pos_tx_rx
)

#%%
trs = scene.get_plot_traces() + [sc.get_plot_traces()]
fig = go.Figure(data=trs)
fig.update_yaxes(scaleratio=1, scaleanchor='x')

arrow = go.layout.Annotation(dict(
                x=u[0]+o[0],
                y=u[1]+o[1],
                xref="x", yref="y",
                text="u",
                showarrow=True,
                axref = "x", ayref='y',
                ax=o[0],
                ay=o[1],
                arrowhead = 3,
                arrowwidth=1.5))
fig.update_layout(
    annotations=[arrow]
)

fig.show()

# %%
alg = DAS2d(scene=scene, sig=t, f_samp=f_samp, c=c)
# %%

im = alg.run()

# import matplotlib.pyplot as plt
# plt.imshow(im)
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

fig.add_trace(go.Scatter(x=pos_tx_rx[:,0], y=pos_tx_rx[:,1], mode='markers'))
fig.add_trace(go.Scatter(x=pos_tx_rx[:,2], y=pos_tx_rx[:,3], mode='markers'))
fig.update_yaxes(autorange=True)
fig.show()

# %%
