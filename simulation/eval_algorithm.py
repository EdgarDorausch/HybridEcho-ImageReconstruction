#%%--------------------------------------------------------------------------- #
# ------------------------------- Load Modules ------------------------------- #

from utils.Simulation.simulation2d import LinISimulation
from utils.ArrayConfiguration import ArrayConfiguration
from utils.ScatterConfiguration import ScatterConfiguration
import numpy as np
from utils.Algorithm.algorithm2d import Scene2d, DAS2d

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


#%%--------------------------------------------------------------------------- #
# ----------------------------- Define Properties ---------------------------- #

# Physical Properties
f_samp=3e6
c=1_484_000.0

# Image Plane
o, u, v = np.array([
    [0.0,  0.0],
    [40.0, 0.0],
    [0.0, 40.0]
])

res_u, res_v = 350, 350

#%%--------------------------------------------------------------------------- #
# ------------------------- Generate Simulation Data ------------------------- #

ac = ArrayConfiguration([
    s*u-v for s in np.linspace(0.0, 1.0, num=10)
],
[
    s*u-v for s in np.linspace(0.0, 1.0, num=10)
])

sc = ScatterConfiguration()
sc.add_circle([10, 0], [20, 20], 50)

pos_tx_rx = np.array(ac)
pos_scatter = np.array(sc)

sim = LinISimulation(
    f_samp=f_samp,
    c=c,
    t1=20e-4,
    mirror_planes=np.array([[0.05,1.,35.], [0.05,1.,35.5], [0.05,1.,36.]])
)
t = sim.simulate(pos_tx_rx, pos_scatter)


#%%--------------------------------------------------------------------------- #
# -------------------------------- Plot Scene -------------------------------- #

scene = Scene2d(
    o=o,
    u=u,
    v=v,
    res_u=res_u,
    res_v=res_v,
    pos_tx_rx=pos_tx_rx
)

trs = scene.get_plot_traces() + [sc.get_plot_traces()]
fig = go.Figure(data=trs)
fig.update_yaxes(scaleratio=1, scaleanchor='x')

fig.update_layout(
    title="Generated Image",
    xaxis_title="X Axis [mm]",
    yaxis_title="Z Axis [mm]",
    legend_title="Legend",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ),
    annotations=scene.get_image_plane_annotations()
)

# Generate Plane Traces
full_fig = fig.full_figure_for_development()
plane_x = np.array(full_fig.layout.xaxis.range)

plane_tr = [
    go.Scatter(
        x=plane_x,
        y=sim.get_mirror_plane_y(plane_x)[i],
        mode='lines',
        name=f'plane-{i}'
    ) for i in range(sim.num_planes)
]
fig.add_traces(plane_tr)

fig.show()

#%%--------------------------------------------------------------------------- #
# ----------------------------- Run DAS Algorithm ---------------------------- #

alg = DAS2d(scene=scene, sig=t, f_samp=f_samp, c=c)
im = alg.run(with_fspl=False)


#%%--------------------------------------------------------------------------- #
# -------------------------------- Plot Image -------------------------------- #

x_space, y_space = scene.get_image_xy_space()

fig = px.imshow(
    im.T,
    x=x_space,
    y=y_space,
    aspect=None,
    color_continuous_scale='gray')

# fig.add_trace(sc.get_plot_traces())

fig.add_traces(scene.get_plot_traces()[:2])
fig.update_yaxes(autorange=True)
fig.update_layout(
    coloraxis_colorbar_x=-0.15,
    title="Generated Image",
    xaxis_title="X Axis [mm]",
    yaxis_title="Z Axis [mm]",
    legend_title="Legend",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

fig.show()



fig = px.imshow(
    im.T,
    x=x_space,
    y=y_space,
    aspect=None
)
fig.update_yaxes(autorange=True)
fig.show()


# %%
