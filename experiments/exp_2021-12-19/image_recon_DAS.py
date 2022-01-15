#%%--------------------------------------------------------------------------- #
# ------------------------------ Import Modules ------------------------------ #

from typing import *
import numpy as np
from matplotlib import pyplot as plt
from time import time
from numpy.core.function_base import linspace
from tqdm import tqdm
from numba import jit
import math
from utils.Scene import Scene
import plotly.graph_objects as go
import plotly.express as px
import os

#%matplotlib inline
plt.rcParams['figure.dpi'] = 140


#%%--------------------------------------------------------------------------- #
# --------------------------------- Load Data -------------------------------- #

# Load TX Signal
sig_tx_raw = np.fromfile(os.path.join(
    os.path.dirname(__file__),
    'data/ch_000/tx_signal_ch_000_00000.float32'), dtype='>f4')
len_sig_raw, = sig_tx_raw.shape

# Load RX Signals
NUM_SIGNALS = 312

# (It is assumed that the rx signals have the same length as the tx signals)
sig_rx_raw = np.empty([len_sig_raw, NUM_SIGNALS])

print('Load RX Data:')
for i in tqdm(range(NUM_SIGNALS)):
    sig_rx_raw[:, i] = np.fromfile(os.path.join(
    os.path.dirname(__file__),
        f'data/ch_000/rx_signal_ch_000_{i+1:05}.float32'), dtype='>f4')


#%%--------------------------------------------------------------------------- #
# ----------------------------- Experiment Specs ----------------------------- #

# Sampling freq.
f_samp = 100e6 # [Hz]

f_min, f_max = 2e6, 10e6 # [Hz]

# Sampling period
dt_samp = 1/f_samp # [s]

# Mean speed of sound (water)
c_sound = 1_484_000.0 # [mm/s]

f_s = f_samp

x_l = len_sig_raw

f0 = int(f_min/f_samp*len_sig_raw) #min of bandpass signal
f1 = int(f_max/f_samp*len_sig_raw) #max freq of bandpass signal

print({'f0': f0, 'f1': f1})

#%%--------------------------------------------------------------------------- #
# ----------------------------- Data Preparation ----------------------------- #

sig_rx_raw_f = np.fft.fft(sig_rx_raw, axis=0)
sig_tx_raw_f = np.fft.fft(sig_tx_raw)

sig_rx_np_f = sig_rx_raw_f[f0:f1, :] # bandpass signal only (tx)
sig_tx_bp_f = sig_tx_raw_f[f0:f1] # bandpass signal only (rx)
sig_rx_ir_f = sig_rx_np_f * sig_tx_bp_f[:, None].conj()  # "rotate back phases" of chirp
sig_rx_cir = np.abs(np.fft.ifft(sig_rx_ir_f, axis=0)) # reconstruct whole signal
sig_rx_cir.shape

f_res = f_samp/len_sig_raw * (sig_rx_cir.shape[0])
dt = 1/f_res


# Plot band selection
fig = go.Figure()
fig.add_trace(go.Scatter(y=sig_tx_raw_f.real, x=np.linspace(0.0, f_samp, len_sig_raw)))
fig.add_vrect(
    x0=f_min, x1=f_max,
    fillcolor="LightSalmon", opacity=0.5,
    layer="above", line_width=0,
)
fig.show()


#%%-------------------------------------------------------------------------- #
# --------------------------- Load TX/RX Positions --------------------------- #

# Construct mask for selected relations
def construct_selected(offsets: List[int]) -> np.ndarray:
    selected = np.empty([sum(offsets)], dtype=bool)
    curr_offset = 0
    mode = True
    for n in offsets:
        next_offset = curr_offset+n
        selected[curr_offset:next_offset] = mode
        curr_offset = next_offset
        mode = not mode
    return selected

selected = construct_selected([50, 0, 50, 4, 50, 3, 50, 4, 100, 1])

pos_raw = np.genfromtxt(os.path.join(
    os.path.dirname(__file__),'data/positions.csv'), delimiter=',', dtype=np.float64)

sig_rx_sel_cir = sig_rx_cir[:,selected]
pos_raw_sel = pos_raw[selected]

pitch = 260e-3 # [mm]
size_piezo =  pos_raw_sel[50, 1] - pos_raw_sel[49, 1] - 2*pitch # [mm]

diff = (pos_raw_sel[49,:2]+pos_raw_sel[50, :2])/2 - pos_raw_sel[100, 4:6]
pos = pos_raw_sel.copy()
pos[:, 4:6] += diff[None,:] 


#%%--------------------------------------------------------------------------- #
# ----------------------------- Image Properties ----------------------------- #
# Define image plane
u, v, o = np.array([[0., 100., 0.], [0., 0., -180.], [256, -254, 100.]])

# Define image plane resolution
res_u, res_v = 3*100, 3*180


#%%--------------------------------------------------------------------------- #
# ------------------------------- Define Scene ------------------------------- #
c_pos = pos[:, :3].T
p_pos = pos[:, 4:7].T

data = sig_rx_cir.T

from utils.Algorithm.algorithm2d import Scene2d, DAS2d
scene = Scene2d(u[1:], v[1:], o[1:], res_u, res_v, pos[:, [5,6,1,2]])

fig = go.Figure(data=scene.get_plot_traces())
fig.update_yaxes(scaleratio=1, scaleanchor='x')
fig.show()

#%%--------------------------------------------------------------------------- #
# --------------------------- Image Reconstruction --------------------------- #

alg = DAS2d(scene, data[selected], f_res, c_sound)
im = alg.run()

#%%--------------------------------------------------------------------------- #
# ---------------------------- Plot Data Selection --------------------------- #
t_min, t_max = alg.get_imaging_time_interval()

fig = go.Figure(data=[go.Scatter(
    y=np.log(data.mean(axis=0)),
    x=np.linspace(0.0, dt*data.shape[1], data.shape[1]),
    name='mean signal'
)])
fig.add_vrect(
    x0=t_min, x1=t_max,
    fillcolor="LightSalmon", opacity=0.5,
    layer="below", line_width=0,
)
fig.update_layout(
    coloraxis_colorbar_x=-0.15,
    title="Data Selection",
    xaxis_title="time [s]",
    yaxis_title="logarithmic amplitude",
    legend_title="Legend",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
fig.update_layout(showlegend=True)
fig.show()

#%%-------------------------------------------------------------------------- #
# -------------------------------- Plot Image -------------------------------- #

x_space, y_space = scene.get_image_xy_space()
fig = px.imshow(
    im.T,
    x=x_space,
    y=y_space,
    aspect=None,
    color_continuous_scale='gray')
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
    np.log(im.T),
    x=x_space,
    y=y_space,
    aspect=None,
    # color_continuous_scale='gray'
)
fig.update_yaxes(autorange=True)
fig.show()

#%%-------------------------------------------------------------------------- #