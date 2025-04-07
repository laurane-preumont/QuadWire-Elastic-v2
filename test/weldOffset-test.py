"""
Testing behavior due to welding offset
"""
# %%
# clear workspace

from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic('reset', '-sf')

# %% Imports
import numpy as np
import scipy as sp
import qw_structure as qws
from modules import mesh, weld, plot, forces, fem
from shape import shapeOffset, splitBeads
from shape import derivateOffset  # import shape_structure

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
Hb = 0.1  # height of the section in mm
beadType = "linear"  # linear, circular, square, quarter_circular, quarter_square, sinus
layerType = "normal"  # default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
meshType = False  # meshRefined
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
# path = r".\thermal_data\wall"     #path = r".\thermal_data\fake_data"   #
# generate_thermal.generate_fake_data(nElems*nLayers, path)
dT = -60
loadType = "uniform"  # "uniform", "linear", "quad" or "random"

# Plot
toPlot = True
clrmap = 'stt'
scfplot = 10

# Data recap as tuple
meshing = L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes
loading = dT, loadType
discretisation = elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
plotting = toPlot, clrmap, scfplot

# offset
offsetType = 'linear'
tcoef, ncoef = 0, 0.4
offset = shapeOffset.stacking_offset(meshing, offsetType, tcoef, ncoef)  # offset between successive layers along t and n (nLayers_v)
#offset_n = [0, -0.2, 0.2] # [0, -0.2, 0.2, 0.6]  #
#offset = np.vstack((np.zeros((1, nLayers_v * nNodes_h)), np.repeat(offset_n, nNodes_h)))



# %%

U, us, yKy, Assemble, Y, yfbc, X, Elems = derivateOffset.shape_structure(meshing, 0 * offset, loading, discretisation, material)
U, Eps, Sigma, nrg, qp2elem, nQP, x, Elems = qws.structure(meshing, 0 * offset, loading, discretisation, material, toPlot, clrmap="stt", scfplot=10)

U_offset, us_offset, yKy_offset, Assemble_offset, Y_offset, yfbc_offset, X_offset, Elems_offset = derivateOffset.shape_structure(meshing, offset, loading, discretisation, material)
U_offset, Eps_offset, Sigma_offset, nrg_offset, qp2elem_offset, nQP_offset, x_offset, Elems_offset = qws.structure(meshing, offset, loading, discretisation, material, toPlot, clrmap="stt", scfplot=10)
