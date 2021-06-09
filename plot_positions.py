import plotly.graph_objects as go
import numpy as np
from read_data import load_position_data

p_pos, c_pos = load_position_data('data/toastgitter/LOGFILE_unified_coordinates.txt')

# p_pos = np.array([[675.0, -200.0, 0.0]]).T - p_pos 

print(p_pos)

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


# Piezo data
piezo_data = go.Scatter3d(
    x=p_pos[0,:],
    y=p_pos[1,:],
    z=p_pos[2,:],
    mode='markers',
    marker=dict(
        size=[12 if s else 8 for s in selected],
        color=['red' if s else 'green' for s in selected],                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    ),
    name='Piezo'
)

# cMut data
cmut_data = go.Scatter3d(
    x=c_pos[0,:],
    y=c_pos[1,:],
    z=c_pos[2,:],
    mode='markers',
    marker=dict(
        size=[12 if s else 8 for s in selected],
        color=['red' if s else 'blue' for s in selected],                # set color to an array/list of desired values
        opacity=0.8
    ),
    name='cMut'
)

u = np.array([50,0,0], dtype=float)
v = np.array([0,0,-50], dtype=float)
o = np.array([350, -110, 25], dtype=float)
image_plane = construct_plotly_image_plane(u,v,o)

xL = []
yL = []
zL = []
res_u = 32
res_v = 32
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
fig.write_html("plots/positions.html")



