"""
Testing split beads
"""
# %%
# clear workspace
from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic('reset', '-sf')

# %% Imports
import numpy as np
import scipy as sp
from modules import mesh, weld, fem, plot
from shape import shapeOffset, splitBeads
from shape.shapeOffset import connection_table_n, vertical_connections_n

# %%
''''''
# Integration
elemOrder = 1
quadOrder = elemOrder + 1

# Material behavior
E = 3e3  # Young modulus /!\ in MPa !n
nu = 0.3  # poisson ratio
alpha = 1.13e-5  # thermal expansion coefficient
optimizedBehavior = True  # use the optimized behavior

# Geometry
L = 5  # length in mm
Hn = 0.1  # width of the section in mm
Hb = 0.1  # height of the section in mm
beadType = "linear"  # linear, circular, square, quarter_circular, quarter_square, sinus
layerType = "normal"  # default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
meshType = True  # meshRefined
zigzag = False

# Mesh Parameters
nLayers_h = 1  # number of horizontal layers (beads)
nLayers_v = 3  # number of vertical layers
nLayers = nLayers_h * nLayers_v  # number of layers

nNodes = 2  # number of nodes per layers
nNodes_h = nNodes * nLayers_h
nNodesTot = nNodes * nLayers
nElems = nNodes - 1
lc = L / nElems  # length of elements np.sqrt(12)*lt

# thermal data
dT = -60
loadType = "uniform"  # "uniform", "linear", "quad" or "random"
# path = r".\thermal_data\wall"     #path = r".\thermal_data\fake_data"   #
# generate_thermal.generate_fake_data(nElems*nLayers, path)

# Plot
toPlot = False
clrmap = 'stt'
scfplot = 1

# %% Data recap as tuple
meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
loading = dT, loadType
discretisation = elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
plotting = toPlot, clrmap, scfplot
offset = shapeOffset.stacking_offset(meshing, "linear", 0, 0.4)  # offset between successive layers along t and n (2, nLayers_v*nNodes_h)
# offset = np.array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,    0. , 0. ,  0. ,  0. ,   0. ,  0. ],
#        [ 0. ,  0. ,  0. ,  0. ,   -0.2, -0.2, -0.2,  -0.2,  0.2,  0.2,  0.2,    0.2]])

# %% geometry without splitbeads
### Discretization and meshing
X, Elems, U0 = mesh.mesh_first_bead(L, nNodes, beadType, meshType)
X, Elems, U0 = mesh.mesh_first_layer(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, layerType, zigzag)
X, Elems, U0 = mesh.mesh_structure(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, zigzag)
weldDof = weld.welding_conditions(X, meshing)

meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
X = shapeOffset.mesh_offset(meshing, offset, X, Elems)
# plot.plotpoints(X, True)

# %% splitbeads
### Extra split beads to model offset
split_bool = True
split_bool *= (nLayers_v > 1)  # no split beads if single layer

if bool(split_bool):
    ## generate trajectory with extra split beads (as if split beads were regular beads)
    meshing_split = L, Hn, Hb, nLayers_h + 1, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
    X_split, Elems_split, U0_split = mesh.generate_trajectory(meshing_split)
    # plot.plotpoints(X_split, True)

    ## apply offset (considering respective bead width for split beads)
    offset_new, offset_split, overhang_n = splitBeads.stacking_offset_split_beads(meshing, offset)
    X_split = splitBeads.mesh_offset_split(X_split, Elems_split, meshing_split, offset_split, zigzag)
    # plot.plotpoints(X_split, True)

    # making split weldDof relies on weldDof weights computed without splitbeads
    extended_weldDof = shapeOffset.make_extended_weldDof(meshing, offset, weldDof)

    extended_weldDof_split = splitBeads.extended_split_weldDof(meshing_split, extended_weldDof, overhang_n)

    X, Elems, U0 = X_split, Elems_split, U0_split
    meshing = meshing_split
    # offset = offset_split
    offset = offset_new
    weldDof = extended_weldDof_split[:, :4].astype(int)
    nodes_connections = connection_table_n(meshing, weldDof)
    print('follower_nodes\n', nodes_connections[0], '\nleader_nodes\n', nodes_connections[1])
    node, particle = 0, 1
    print(f'({node},{particle}) is follower to\n {extended_weldDof_split[(extended_weldDof_split[:, 2] == node) * (extended_weldDof_split[:, 3] == particle)]}\n'
          f'and leader of\n {extended_weldDof_split[(extended_weldDof_split[:, 0] == node) * (extended_weldDof_split[:, 1] == particle)]}')

# %% FEM
Xunc, uncElems = mesh.uncouple_nodes(X, Elems)

### Prevision taille
nElemsTOT = Elems.shape[0]
nUncNodes = Xunc.shape[0]
nNodesTot = X.shape[0]
nParticules = 4
nCoord = 3
nNodeDOF = nCoord * nParticules

### Useful matrices
## Integration matrices
xiQ, wQ = fem.element_quadrature(quadOrder)
XIQ, WQ = fem.fullmesh_quadrature(quadOrder, nElemsTOT)
nQP = WQ.shape[0]
N, Dxi, Ds, J, W, O, qp2elem, elemQ = fem.integration_matrices(X, Elems, elemOrder, quadOrder)
## Identity matrix (nQP, nQP)
I_nqp = sp.sparse.diags(np.ones(nQP))
## Projection matrices
T, Tv, Tc, Ta = fem.alpha2beta_matrices(nNodesTot, nUncNodes)
t, n, b, P = fem.local2global_matrices(Xunc, uncElems, elemOrder)

# %% Welding matrices
Sni, Sn = weld.weldnode_matrices(Elems, uncElems)
Sw0 = weld.weldwire_matrix(nNodesTot, weldDof)
if split_bool:
    Sw = splitBeads.offset_split_weldwire_matrix(meshing, extended_weldDof_split, Sw0)
else:
    Sw = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw0)
plot.plotmatrix_sparse_t(Sw, nNodesTot, f'{nLayers_v} couches de {nLayers_h} cordons \n offset {int(offset[-1][-1] * 100)}% (Noeuds pairs, axe t)')
