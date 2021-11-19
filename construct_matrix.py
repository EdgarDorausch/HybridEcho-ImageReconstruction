from __future__ import annotations
from re import VERBOSE
from typing import *
if TYPE_CHECKING:
    pass
from numpy.lib.function_base import iterable
from src.Kernel import GaussKernel, Kernel
from src.EllipsoidRenderer import EllipsoidRenderer
from src.GridBuilder import ImageGridBuilder
from read_data import load_position_data
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from time import sleep




class MatrixBuilder:

    def __init__(
        self,
        data_path: str,
        positions_path: str,
        piezo_filter: Callable[[np.ndarray], bool],
        cMut_filter: Callable[[np.ndarray], bool],
        verbose: bool = True,
        SPEED_OF_SOUND = 1_484_000.0, # [mm/s]
        SAMPLE_RATE = 100_000_000.0, # [1/s]
        ECHO_START_TIME = 0.0 # [s]
    ):
        self.SPEED_OF_SOUND = SPEED_OF_SOUND
        self.SAMPLE_RATE = SAMPLE_RATE
        self.ECHO_START_TIME = ECHO_START_TIME

        self.verbose = verbose
        self.data = np.load(data_path) # Shape [STEPS, NUM_REL]
        self.p_pos, self.c_pos = load_position_data(positions_path)

        selected_p = np.apply_along_axis(piezo_filter, 0, self.p_pos)
        selected_c = np.apply_along_axis(cMut_filter, 0, self.c_pos)
        self.selected = np.logical_and(selected_p, selected_c)

        self.NUM_RELATION = np.sum(self.selected)

    def _log(self, *text: str):
        if self.verbose:
            print(*text)

    def build_matrix(self, u: np.ndarray, v: np.ndarray, o: np.ndarray, res_u: int, res_v: int, kernel: Kernel) -> Tuple[np.ndarray, int, int]:
        
        # Build grid
        igb = ImageGridBuilder(u, v, o) #image will be flipped along y axis!
        grid = igb.build(res_u, res_v).reshape([3,-1])
        NUM_PIXEL = res_u*res_v

        # Only use samples where ellipsoide goes through the image plane 
        # therefore compute minimal and maximal time define interval of interest on samples
        distances = np.linalg.norm(self.p_pos[:,self.selected,None]-grid[:,None,:], axis=0) + \
                    np.linalg.norm(self.c_pos[:,self.selected,None]-grid[:,None,:], axis=0)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        lower_sample_idx = math.floor((self.ECHO_START_TIME + (min_dist / self.SPEED_OF_SOUND)) * self.SAMPLE_RATE) # inclusive
        upper_sample_idx = math.ceil((self.ECHO_START_TIME + (max_dist / self.SPEED_OF_SOUND)) * self.SAMPLE_RATE)+1 # not inclusive

        NUM_SAMPLES = upper_sample_idx - lower_sample_idx

        # Finally construct the matrix
        M = np.empty([NUM_SAMPLES*self.NUM_RELATION, NUM_PIXEL])
        for rel_idx, p_c_pos in enumerate(np.concatenate([self.p_pos[:,self.selected], self.c_pos[:,self.selected]], axis=0).T):
            ts = (np.arange(lower_sample_idx, upper_sample_idx)/self.SAMPLE_RATE - self.ECHO_START_TIME)*self.SPEED_OF_SOUND
            # print(ts)
            # sleep(3)
            er = EllipsoidRenderer(p_c_pos[0:3], p_c_pos[3:6], kernel=kernel)
            foo = er.render(grid[:,: None, None], ts).reshape([NUM_SAMPLES, NUM_PIXEL])
            M[rel_idx*NUM_SAMPLES:(rel_idx+1)*NUM_SAMPLES,:] = foo

        return M, lower_sample_idx, upper_sample_idx

    def plot_positions(self, u: np.ndarray, v: np.ndarray, o: np.ndarray, res_u: int, res_v: int, kernel: Kernel):
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


        # selected_p = np.apply_along_axis(
        #     lambda pos: 
        #         np.allclose(pos, np.array([385.0, -110.0, 198.0])),
        #     0,
        #     self.p_pos)

        # selected_c = np.apply_along_axis(
        #     lambda pos: any(
        #         np.allclose(pos, np.array([1.0*k+311.1, -110.0, 198.0]))
        #         for k in range(0,61,2)
        #         ),
        #     0,
        #     self.c_pos)


        # Piezo data
        piezo_data = go.Scatter3d(
            x=self.p_pos[0,:],
            y=self.p_pos[1,:],
            z=self.p_pos[2,:],
            mode='markers',
            marker=dict(
                size=[12 if s else 8 for s in self.selected],
                color=['red' if s else 'green' for s in self.selected],                # set color to an array/list of desired values
                # colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            ),
            name='Piezo'
        )

        # cMut data
        cmut_data = go.Scatter3d(
            x=self.c_pos[0,:],
            y=self.c_pos[1,:],
            z=self.c_pos[2,:],
            mode='markers',
            marker=dict(
                size=[12 if s else 8 for s in self.selected],
                color=['red' if s else 'blue' for s in self.selected],                # set color to an array/list of desired values
                opacity=0.8
            ),
            name='cMut'
        )

        # u = np.array([50,0,0], dtype=float)
        # v = np.array([0,0,-50], dtype=float)
        # o = np.array([350, -110, 25], dtype=float)
        image_plane = construct_plotly_image_plane(u,v,o)

        xL = []
        yL = []
        zL = []
        # res_u = 32
        # res_v = 32
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

        # tight layout
        # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()
        fig.write_html("plots/setup.html")



if __name__ == "__main__":
    mb = MatrixBuilder(
        data_path      = 'data/toastgitter/all_data.npy',
        positions_path = 'data/toastgitter/LOGFILE_unified_coordinates.txt',
        piezo_filter   = lambda pos: np.allclose(pos, np.array([385.0, -110.0, 198.0])),
        cMut_filter    = lambda pos: any(
            np.allclose(pos, np.array([1.0*k+311.1, -110.0, 198.0]))
            for k in range(0,61,2)
        ),
    )

    u = np.array([50,0,0], dtype=float)
    v = np.array([0,0,-50], dtype=float)
    o = np.array([350, -110, 25], dtype=float)

    res_u = 32
    res_v = 32

    kernel = GaussKernel(1.0)
    mb.plot_positions(u=u, v=v, o=o, res_u=res_u, res_v=res_v, kernel=kernel)
    M, lower_sample_idx, upper_sample_idx = mb.build_matrix(u=u, v=v, o=o, res_u=res_u, res_v=res_v, kernel=kernel)


    print(M.shape)

    Minv= np.linalg.pinv(M)

    # flatten vector (column major)
    selected_data = mb.data[lower_sample_idx:upper_sample_idx, mb.selected].flatten(order='F')

    x_hat = Minv @ selected_data

    x_hat = x_hat.reshape([res_u, res_v])

    plt.imshow(x_hat)
    plt.show()