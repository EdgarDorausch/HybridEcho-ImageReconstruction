from __future__ import annotations
from typing import *

if TYPE_CHECKING:
    pass
import numpy as np
import plotly.graph_objects as go


def construct_plotly_image_plane(u,v,o, name='image plane'):
    vert = np.empty([4,3])
    vert[0] = o
    vert[1] = o+u
    vert[2] = o+v
    vert[3] = o+u+v

    x = vert[:, 0]
    y = vert[:, 1]
    z = vert[:, 2]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=[0,1],
        j=[1,3],
        k=[2,2],
        name=name,
        opacity=0.2
    )

def create_cone(pos, dir1, dir2, name):
    dir1/=np.linalg.norm(dir1)
    dir2/=np.linalg.norm(dir2)
    return go.Cone(x=[pos[0],pos[0]], y=[pos[1],pos[1]], z=[pos[2],pos[2]], u=[dir1[0], dir2[0]], v=[dir1[1], dir2[1]], w=[dir1[2], dir2[2]],
    sizemode="absolute",
    sizeref=2,
    anchor="tip",
    name=name)


class Scene:
    """
    Holds configuration of image plane, piezo / cMut Positions as well as the measured signal
    """

    def __init__(self, u: np.ndarray, v: np.ndarray, o: np.ndarray, res_u, res_v, pos_tx, pos_rx, selected):

        self.o = o
        self.u = u
        self.v = v
        self.res_u = res_u
        self.res_v = res_v
        self.pos_tx = pos_tx 
        self.pos_rx = pos_rx 
        self.selected = selected

    def plot_scene(self):
        fig = self.__create_figure()
        fig.show()

    def save_figure(self, path: str):
        fig = self.__create_figure()
        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.write_html(path)

    def __create_figure(self):

        # Piezo data
        piezo_data = go.Scatter3d(
            x=self.pos_tx[0,:],
            y=self.pos_tx[1,:],
            z=self.pos_tx[2,:],
            mode='markers',
            marker=dict(
                size=[12 if s else 8 for s in self.selected],
                color=['red' if s else 'green' for s in self.selected],                # set color to an array/list of desired values
                # colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            ),
            name='TX'
        )

        # cMut data
        cmut_data = go.Scatter3d(
            x=self.pos_rx[0,:],
            y=self.pos_rx[1,:],
            z=self.pos_rx[2,:],
            mode='markers',
            marker=dict(
                size=[12 if s else 8 for s in self.selected],
                color=['red' if s else 'blue' for s in self.selected],                # set color to an array/list of desired values
                opacity=0.8
            ),
            name='RX'
        )

        u = self.u
        v = self.v
        o = self.o
        image_plane = construct_plotly_image_plane(u,v,o)

        xL = []
        yL = []
        zL = []
        res_u = self.res_u
        res_v = self.res_v
        for du in range(res_u+1):
            h1 = o+du*u/res_u
            h2 = o+du*u/res_u+v
            xL.extend([h1[0], h2[0], None])
            yL.extend([h1[1], h2[1], None])
            zL.extend([h1[2], h2[2], None])
        for dv in range(res_v+1):
            h1 = o+dv*v/res_v
            h2 = o+dv*v/res_v+u
            xL.extend([h1[0], h2[0], None])
            yL.extend([h1[1], h2[1], None])
            zL.extend([h1[2], h2[2], None])

        wireframe = go.Scatter3d(
                x=xL,
                y=yL,
                z=zL,
                mode='lines',
                line=dict(color= 'rgb(70,70,70)', width=1),
                name='image grid')  

        # arrow_u = create_cone(o, u, v, 'u')
        # arrow_v = create_cone(o, v, 'v')

        layout = go.Layout(
            scene=dict(
                aspectmode='data'
        ))

        fig = go.Figure(data=[
            piezo_data,
            cmut_data,
            image_plane,
            wireframe
            # arrow_u,
            # arrow_v,
        ], layout=layout)

        return fig
        