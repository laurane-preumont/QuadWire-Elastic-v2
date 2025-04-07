"""
Testing welding wires vertically with offset between successive beads
"""
# %%
# clear workspace
from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic('reset', '-sf')

# %% Imports
import numpy as np

import qw_structure as qws
from modules import mesh, weld, plot, forces
from shape import shapeOffset, shapeOptim, derivateOffset

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
L = 10  # length in mm
Hn = 0.1  # width of the section in mm
Hb = 0.2  # height of the section in mm
beadType = "circular"  # linear, circular, square, quarter_circular, quarter_square, sinus
layerType = "normal"  # default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
meshType = False  # meshRefined
zigzag = False

# Mesh Parameters
nLayers_h = 1  # number of horizontal layers (beads)
nLayers_v = 20  # number of vertical layers
nLayers = nLayers_h * nLayers_v  # number of layers

nNodes = 50  # number of nodes per layers
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
discretization = elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
plotting = toPlot, clrmap, scfplot
offset = shapeOffset.stacking_offset(meshing, "gauss", 0, 0.4)  # offset between successive layers along t and n (2, nLayers_v*nNodes_h)


# %% Mesh geometry without offset
X, Elems, U0 = mesh.mesh_first_bead(L, nNodes, beadType, meshType)
X, Elems, U0 = mesh.mesh_first_layer(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, layerType, zigzag)
X, Elems, U0 = mesh.mesh_structure(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, zigzag)
weldDof = weld.welding_conditions(X, meshing)
plot.plotpoints(X, False)

X = shapeOffset.mesh_offset(meshing, offset, X, Elems)
plot.plotpoints(X, False)




# %% welding matrix without offset
Sw = weld.weldwire_matrix(nNodesTot, weldDof)  # welding matrix without offset is used to find primary leaders
#plot.plotmatrix_sparse_t(Sw, nNodesTot)

# %% Generate offset geometry data (considering null offset)

v_connections = shapeOffset.vertical_connections_n(meshing, weldDof)
v_mask = shapeOffset.connectionType(weldDof)
offset_weights0 = shapeOffset.welding_weights(meshing, 0 * offset)
extended_weldDof0 = shapeOffset.make_welding_connections_n(meshing, offset_weights0, weldDof, v_connections, v_mask)
offset_weights = shapeOffset.welding_weights(meshing, offset)
extended_weldDof = shapeOffset.make_welding_connections_n(meshing, offset_weights, weldDof, v_connections, v_mask)

extended_v_mask = shapeOffset.welding_type(extended_weldDof0)[:, -1].astype(bool)  # adds column with 0 for horizontal connection and 1 for vertical connection

# %% Set offset
X = shapeOffset.mesh_offset(meshing, offset, X, Elems)
#plot.plotpoints(X, True)

# %% Compute welding matrix with new offset weights
Sw_offset = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw)
#plot.plotmatrix_sparse_t(Sw_offset, nNodesTot, f'{nLayers_v} couches de {nLayers_h} cordons \n offset {int(offset[-1][-1] * 100)}% (Noeuds pairs, axe t)')

# %% Set offset + epsilon delta_i
epsilon = 10**-3
i=1
offset_i = 0 * offset
offset_i[:, i * nNodes_h:(i + 1) * nNodes_h] = 1
offset_epsilon_i = offset_i * epsilon
X = shapeOffset.mesh_offset(meshing, offset_epsilon_i, X, Elems)
offset_weights = shapeOffset.welding_weights(meshing, offset_epsilon_i)
extended_weldDof_weights = shapeOffset.make_welding_connections_n(meshing, offset_weights, weldDof, v_connections, v_mask)  # weights due to offset   /!\/!\/!\
primary_weldDof_weights = shapeOffset.make_primary_leaders(extended_weldDof_weights, Sw)  # shift all follower particles to their primary leader
extended_v_mask = shapeOffset.welding_type(extended_weldDof_weights)[:, -1].astype(bool)

Sw_epsilon = shapeOffset.update_weldwire_matrix_weights(nNodesTot, primary_weldDof_weights[extended_v_mask], Sw)  # only vertical connections need to be updated
# plot.plotSw_t(Sw_update, nNodesTot)
plot.plotmatrix_sparse_t_evenNodes(Sw_epsilon, nNodesTot, f'{nLayers_v} couches de {nLayers_h} cordons \n offset {int(offset[-1][-1] * 100)}% (Noeuds pairs, axe t)')

Sw_derivative = derivateOffset.derive_Sw(meshing, offset, weldDof, Sw, i)
plot.plotmatrix_sparse_t_evenNodes(Sw_derivative, nNodesTot, f'{nLayers_v} couches de {nLayers_h} cordons \n offset {int(offset[-1][-1] * 100)}% (Noeuds pairs, axe t)')


Y_epsilon = derivateOffset.finDiff_Y(U0, Sw_offset, Sw_epsilon, epsilon)
Y_derivative = derivateOffset.derive_Y(U0, Sw, Sw_derivative)
# plot.plotpoints(X, True)


# %% Matrice cible pour 2x2 cordons, décalage +0.75  mais followers de leaders
#
# Sw_cible = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
#                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], [0.25, 0.75, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0.25, 0., 0., 0., 0.75, 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
#                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 0.25, 0.75, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
#
# # plot.plotmatrix(Sw_cible, 'Sw_cible')
# # plot.plotmatrix(Sw_cible @ Sw_cible, 'Sw_cible²')
# %%
Sw_d0 = derivateOffset.derive_Sw(meshing, offset, weldDof, Sw, 0)
plot.plotmatrix_sparse_t_evenNodes(Sw_d0, nNodesTot, f'dérivée de Sw par rapport à delta_0')

Sw_d1 = derivateOffset.derive_Sw(meshing, offset, weldDof, Sw, 1)
plot.plotmatrix_sparse_t_evenNodes(Sw_d1, nNodesTot, f'dérivée de Sw par rapport à delta_1')

Sw_d2 = derivateOffset.derive_Sw(meshing, offset, weldDof, Sw, 2)
plot.plotmatrix_sparse_t_evenNodes(Sw_d2, nNodesTot, f'dérivée de Sw par rapport à delta_2')

# # %% Solution analytique pour 3 couches de 1 éléments, axe t éléments pairs (plotmatrix_sparse_t_evenNodes : Sw.todense()[:4 * nNodesTot, :4 * nNodesTot][::2, ::2]) et offset = np.array([[-0., -0., -0., -0., -0., -0.], [0., 0., 0.75, 0.75, 1.25, 1.25]])
# SwCheck = np.matrix([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], [0.25, 0., 0., 0.75, 0., 0., 0., 0., 0., 0., 0., 0.],
#                 [0., 0.5, 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
# Sw_d0 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# Sw_d1 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [-1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# Sw_d2 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [-0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                   [0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])


# %% Computation

# data with 0% offset
offset = shapeOffset.stacking_offset(meshing, "linear", 0, 0)  # layerwise offset from initial X
U0, Eps0, Sigma0, elementEnergyDensity0, qp2elem, nQP, x0, Elems = qws.structure(L, Hn, beadType, layerType, offset, zigzag, nNodes, nLayers_h, nLayers_v)

sigmatt_elem0 = qp2elem @ Sigma0[:nQP]  # tensile stress at quadrature points
Elem_Sigma_tt0 = sigmatt_elem0.reshape(nLayers_h, nLayers_v, nNodes - 1)
Avg_Sigma_tt0 = np.average(Elem_Sigma_tt0[-1, :, :], axis=1)

# data with 50% offset
offset = shapeOffset.stacking_offset(meshing, "linear", 0, 0.5)  # layerwise offset from initial X
U1, Eps1, Sigma1, elementEnergyDensity1, qp2elem, nQP, x1, Elems = qws.structure(L, Hn, beadType, layerType, offset, zigzag, nNodes, nLayers_h, nLayers_v)

sigmatt_elem1 = qp2elem @ Sigma1[:nQP]  # tensile stress at quadrature points
Elem_Sigma_tt1 = sigmatt_elem1.reshape(nLayers_h, nLayers_v, nNodes - 1)
Avg_Sigma_tt1 = np.average(Elem_Sigma_tt1[-1, :, :], axis=1)

# data  with 75% offset
offset = shapeOffset.stacking_offset(meshing, "linear", 0, 0.75)  # layerwise offset from initial X
U2, Eps2, Sigma2, elementEnergyDensity2, qp2elem, nQP, x2, Elems = qws.structure(L, Hn, beadType, layerType, offset, zigzag, nNodes, nLayers_h, nLayers_v)

sigmatt_elem2 = qp2elem @ Sigma2[:nQP]  # tensile stress at quadrature points
Elem_Sigma_tt2 = sigmatt_elem2.reshape(nLayers_h, nLayers_v, nNodes - 1)
Avg_Sigma_tt2 = np.average(Elem_Sigma_tt2[-1, :, :], axis=1)

# Define constants
scfplot = 1
clr0 = sigmatt_elem0.flatten()
clr1 = sigmatt_elem1.flatten()
clr2 = sigmatt_elem2.flatten()
clrdiff1 = (clr0 - clr1) / clr0
clrdiff2 = (clr0 - clr2) / clr0

delamination0 = forces.delamination(Sigma0, Hn, Hb, nLayers_v, nLayers_h, nNodes, axis=2)
dela0 = delamination0.flatten()
delamination1 = forces.delamination(Sigma1, Hn, Hb, nLayers_v, nLayers_h, nNodes, axis=2)
dela1 = delamination0.flatten()
delamination2 = forces.delamination(Sigma2, Hn, Hb, nLayers_v, nLayers_h, nNodes, axis=2)
dela2 = delamination0.flatten()
deladiff1 = (dela0 - dela1) / dela0
deladiff2 = (dela0 - dela1) / dela0

# %% Plot tensile stress data
plot.plot_data(x0, U0, clr0, Hn, Hb, L, Elems, scfplot, 'Data0 : offset 0%')
plot.plot_data(x1, U1, clr1, Hn, Hb, L, Elems, scfplot, 'Data1 : offset 50%')
plot.plot_data(x2, U2, clr2, Hn, Hb, L, Elems, scfplot, 'Data2 : offset 75%')

plot.plot_data(x1, U1, clrdiff1, Hn, Hb, L, Elems, scfplot, '(Data0 - Data1)/Data0')
plot.plot_data(x2, U2, clrdiff2, Hn, Hb, L, Elems, scfplot, '(Data0 - Data2)/Data0')

# %%  Plot delamination stress data

plot.plot_data(x0, U0, dela0, Hn, Hb, L, Elems, scfplot, 'dela0 : offset 0%')
plot.plot_data(x1, U1, dela1, Hn, Hb, L, Elems, scfplot, 'dela1 : offset 50%')
plot.plot_data(x2, U2, dela2, Hn, Hb, L, Elems, scfplot, 'dela2 : offset 75%')

plot.plot_data(x1, U1, deladiff1, Hn, Hb, L, Elems, scfplot, '(dela0 - dela1)/dela0')
plot.plot_data(x2, U2, deladiff2, Hn, Hb, L, Elems, scfplot, '(dela0 - dela2)/dela0')

# %% Vérifier la soudure sans offset
# # Egalité des déplacements au centre pour 2 couches de 2 cordons (particule 1 de 0 leader de 2_0 4_3 6_2)
# #   4 __ 6
# #   |    |
# #   0 __ 2
# U[:,0,2]-U[:,1,0]
# U[:,3,4]-U[:,1,0]
# U[:,2,6]-U[:,1,0]
#
# U[:,2,4]-U[:,0,0]
#
# U[:,0,3]-U[:,1,1]
# U[:,3,5]-U[:,1,1]
# U[:,2,7]-U[:,1,1]
#
# # et deux couches de trois cordons
# #   6 __ 8 __ 10
# #   |    |    |
# #   0 __ 2 __ 4
# U[:,0,2]-U[:,1,0]
# U[:,3,6]-U[:,1,0]
# U[:,2,8]-U[:,1,0]
#
# U[:,2,6]-U[:,0,0]
#
# U[:,0,5]-U[:,1,1]
# U[:,3,7]-U[:,1,1]
# U[:,2,9]-U[:,1,1]
