"""
Testing ZigZag printing strategy with the QuadWire model.
Thermal data is fake
zigzag parameter is taken into account for functions mesh_first_layer and mesh_structure (3 options to alternate layers, defined inside mesh_structure as zigzagType 1 2 3)
"""
# %%
# clear workspace
from IPython import get_ipython

get_ipython().magic('reset -sf')

# %% Imports
import numpy as np
import qw_additive as qwa
import qw_structure as qws
from modules import mesh, fem, weld, behavior, plot, thermaldata, forces
from shape import shapeOffset
import generate_thermal


#%% Integration
elemOrder = 1
quadOrder = elemOrder + 1

#%% Material behavior
E = 3e3  # Young modulus /!\ in MPa !n
nu = 0.3  # poisson ratio
alpha = 1.13e-5  # thermal expansion coefficient
optimizedBehavior = True  # use the optimized behavior

#%% Geometry
L = 10 # length in mm
Hn = 0.1  # width of the section in mm
Hb = 0.1  # height of the section in mm
beadType = "linear"  # linear, circular, square, quarter_circular, quarter_square, sinus
layerType = "normal" # default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
meshType = True #meshRefined
zigzag = False
split_bool = False


#%% Mesh Parameters
nLayers_h = 10  # number of horizontal layers (beads)
nLayers_v = 15  # number of vertical layers
nLayers = nLayers_h * nLayers_v  # number of layers
nNodes = 31  # number of nodes per layers
nElems = nNodes-1 #nNodes for closed geometry (beadType "circular" or "square")
nElemsTot = nElems * nLayers

# thermal data
dT = -60
loadType = "poc"  # "uniform", "linear", "quad" or "random"

# Plot
toPlot = False
clrmap = 'stt'
scfplot = 1

# Data recap as tuple
meshing = L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes
loading = dT, loadType
discretisation = elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
plotting = toPlot, clrmap, scfplot

offset = 0*shapeOffset.stacking_offset(meshing, "linear", 0, 0.5*(nLayers_v-1))  # offset between successive layers along t and n (2, nLayers_v*nNodes_h)
#offset *= 0 #no offset for zigzag examples


#%% Thermal loading
path = r".\thermal_data\wall"

path = r".\thermal_data\fake_data"
tau_final = nLayers_h*(nNodes-1)#int(np.ceil((nNodes-1)*3/4))
nPas = (nNodes-1)*nLayers + tau_final
Ncooling = nNodes
generate_thermal.generate_fake_data(nPas, nElemsTot, path, loadType)

scalefig = 10

# %% Reference without zigzag
zigzag = False
U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems, Un_max, U_data, Eps_data, Sigma_data = qwa.additive(path, meshing, offset, loading, discretisation, material, toPlot, clrmap, scfplot, split_bool)

plot.plot_sigma(Sigma, qp2elem, nQP, x, Elems, L, Hn, Hb, 'stt', scalefig)
plot.plot_sigma(Sigma, qp2elem, nQP, x, Elems, L, Hn, Hb, 'stn', scalefig)
plot.plot_sigma(Sigma, qp2elem, nQP, x, Elems, L, Hn, Hb, 'stb', scalefig)

# %% ZigZag test
zigzag = True
U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems, Un_max, U_data, Eps_data, Sigma_data = qwa.additive(path, meshing, offset, loading, discretisation, material, toPlot, clrmap, scfplot, split_bool)

plot.plot_sigma(Sigma, qp2elem, nQP, x, Elems, L, Hn, Hb, 'stt', scalefig)
plot.plot_sigma(Sigma, qp2elem, nQP, x, Elems, L, Hn, Hb, 'stn', scalefig)
plot.plot_sigma(Sigma, qp2elem, nQP, x, Elems, L, Hn, Hb, 'stb', scalefig)
