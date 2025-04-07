"""
Testing shape optimization algorithms
"""
# %%
# clear workspace
from IPython import get_ipython

import qw_additive

ipython = get_ipython()
ipython.run_line_magic('reset', '-sf')

# %% Imports
import time
import numpy as np
import matplotlib.pyplot as plt

import qw_additive as qwa
import qw_additive_mixte as qwam
from modules import mesh, weld, plot, forces
from shape import shapeOffset, derivateOffset
from shape.shapeOptim_module import projected_gradient_algorithm, offset2param, param2offset, us2param, ScaledCounterDeformationCost, DisplacementMinimizationCost, CounterDeformationCost, analyze_optimization_results, \
    save_optimization_report, param2overhang, max_decalage

''''''
# Integration
elemOrder = 1
quadOrder = elemOrder + 1

# Material behavior
E = 3e3  # Young modulus /!\ in MPa
nu = 0.3  # poisson ratio
alpha = 1.13e-5  # thermal expansion coefficient
optimizedBehavior = False  # use the optimized behavior

# Geometry
L = 50 #0  # length in mm                        #TODO: modif
Hn = 50 # width of the section in mm            #TODO: modif
Hb = 25 # height of the section in mm          #TODO: modif
beadType = "linear"  # linear, circular, square, quarter_circular, quarter_square, sinus
layerType = "normal"  # default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
meshType = True  # meshRefined
zigzag = False
split_bool = False

# Mesh Parameters
nLayers_h = 1  # number of horizontal layers (beads)
nLayers_v = 20 # number of vertical layers             #TODO: modif
nLayers = nLayers_h * nLayers_v  # number of layers

nNodes = 50 # number of nodes per layers                 #TODO: modif
nNodes_h = nNodes * (nLayers_h+split_bool)
nNodesTot = nNodes * nLayers
nElems = nNodes - 1
nElemsTot = nElems * nLayers

# thermal data
dT = -60
loadType = "uniform"  # "uniform", "linear", "quad" or "random", "poc" ie top layer cools down other layers don't

#%%% Modif param
nNodes = 25
meshType = False
L = 1500
Hn = 150
Hb = 100
nLayers_v = 20
E = 3 #MPa
dT = -600


#%%%
path = None
# Plot
toPlot = False
clrmap = 'stt'
scfplot = 100

# % Data recap as tuple
meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
loading = dT, loadType
loading0 = 0*dT, loadType
discretization = elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
plotting = toPlot, clrmap, scfplot

offset = shapeOffset.stacking_offset(meshing, "linear", 0, -0.8*(nLayers_v-1))  # offset between successive layers along t and n (2, nLayers_v*nNodes_h)

# %%%

start_time=time.time()
U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems, Un_max, U_data, Eps_data, Sigma_data, clr = qwa.additive(path, meshing, offset, loading, discretization, material, toPlot)
end_time=time.time()

Un=U[1]
Un_moy = np.average(U[1], axis=0)
Un_beads = Un_moy.reshape(nLayers_v, nNodes)
Un_beads_delta = np.max(Un_beads[:-1]-Un_beads[1:])

print('nNodes =', nNodes)
print('meshType =', meshType)
print('L =', L)
print('Hn =', Hn)
print('Hb =', Hb)
print('E =', E)
print('nLayers_v =', nLayers_v)
print('Time (min)=', (end_time-start_time)/60)
print('max Un', np.max(Un_moy))
print('max offset', Un_beads_delta)