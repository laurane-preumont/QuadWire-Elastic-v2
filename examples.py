"""
Script used to compute wall and carpet structures with the QuadWire model. 
Reproduces the figures in the article
"""
#%%
# clear workspace

from IPython import get_ipython
get_ipython().magic('reset -sf')

import time

#%% Imports
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import qw_additive as qwa
from modules import mesh, fem, weld, behavior, plot, thermaldata, forces

#%% Carpet Parameters 
### Geometry
L = 10  # length in mm
Hn = 0.1  # width of the section in mm
Hb = 0.1  # height of the section in mm
beadType = "linear"  # linear, circular, square, quarter_circular, quarter_square, sinus
layerType = "normal"  # default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
meshType = False #meshRefined
zigzag = False
### Mesh
nLayers = 15 # number of layers
nNodes = 51 # number of nodes per layers

### Integration
elemOrder = 1
quadOrder = elemOrder + 1 

### Material behavior
E = 3e3  # Young modulus /!\ in MPa !n
nu = 0.3  # poisson ratio
alpha = 1.13e-5  # thermal expansion coefficient
optimizedBehavior = False  # use the optimized behavior

### Plot
toPlot = False # True if you want to see the evolution of the field during printing
clrmap = "stt" # Field to observe on the graph

scalefig = 10

color1 = "#00688B" 
color2 = "#B22222"
color3 = "#556B2F"


#%% Carpet 

### Parameters
nLayers_v = 1
nLayers_h = nLayers
offset = np.zeros((2, nLayers*nNodes))  # offset between successive layers along t and n (nLayers_v, 2)

# % Data recap as tuple
meshing = L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes
loading = None, None #dT, loadType
discretisation = elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
plotting = toPlot, clrmap, scalefig

path = r".\thermal_data\carpet"

### Computation
tic = time.time()
U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems = qwa.additive(path, meshing, offset, loading, discretisation, material, toPlot, clrmap, scalefig)

tac = time.time()
print('carpet:', tac-tic, 's, ', (tac-tic)/60, 'min')

#%% Plot
### Plot stt
projection_index = 0
sigmaplot = qp2elem @ Sigma[projection_index*nQP:(projection_index+1)*nQP]
clr = sigmaplot[:,0]
clr = clr / (Hn * Hb)


fig = plt.figure()
ax = plt.axes(projection='3d', proj_type='ortho')
ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scalefig), np.ptp(x[:, 2]*scalefig)))

srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='none', clrfun=clr  , outer=False)
colorbar = plt.colorbar(srf, pad=0.15)
colorbar.set_label('$\sigma_{tt}$ [MPa]')


### Plot stn
projection_index = 3
sigmaplot = qp2elem @ Sigma[projection_index*nQP:(projection_index+1)*nQP]
clr = sigmaplot[:,0]            
clr = clr / (Hn * Hb)


fig = plt.figure()
ax = plt.axes(projection='3d', proj_type='ortho')
ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scalefig), np.ptp(x[:, 2]*scalefig)))

srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='none', clrfun=clr, outer=False)
colorbar = plt.colorbar(srf, pad=0.15)
colorbar.set_label('$\sigma_{tn}$ [MPa]')


#%% Wall
### Parameters
nLayers_h = 1
nLayers_v = nLayers
stacking_offset = np.zeros((2, nLayers*nNodes))  # offset between successive layers along t and n (nLayers_v, 2)

# % Data recap as tuple
meshing = L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes
loading = None, None #dT, loadType
discretisation = elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
plotting = toPlot, clrmap, scalefig
path = r".\thermal_data\wall"

### Computation
tic = time.time()
U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems = qwa.additive(path, meshing, offset, loading, discretisation, material, toPlot, clrmap, scalefig)
tac = time.time()
print('wall :', tac-tic, 's, ', (tac-tic)/60, 'min')

#%% Plot
### Plot stt
projection_index = 0
sigmaplot = qp2elem @ Sigma[projection_index*nQP:(projection_index+1)*nQP]
clr = sigmaplot[:,0]            
clr = clr / (Hn * Hb)


fig = plt.figure()
ax = plt.axes(projection='3d', proj_type='ortho')
ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scalefig), np.ptp(x[:, 2]*scalefig)))

srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='none', clrfun=clr, outer=False)
colorbar = plt.colorbar(srf, pad=0.15)
colorbar.set_label('$\sigma_{tt}$ [MPa]')


### Plot stb
projection_index = 4
sigmaplot = qp2elem @ Sigma[projection_index*nQP:(projection_index+1)*nQP]
clr = sigmaplot[:,0]            
clr = clr / (Hn * Hb)


fig = plt.figure()
ax = plt.axes(projection='3d', proj_type='ortho')
ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scalefig), np.ptp(x[:, 2]*scalefig)))

srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='none', clrfun=clr  , outer=False)
colorbar = plt.colorbar(srf, pad=0.15)
colorbar.set_label('$\sigma_{tb}$ [MPa]')

#%%
sigtb = Sigma[projection_index*nQP:(projection_index+1)*nQP]/(Hn*Hb)

x_list = np.linspace(-L/2, L/2, int(nQP/nLayers))
plt.figure()
plt.ylabel("$\sigma_{tb}$ [MPa]")
plt.xlabel("s [mm]")
plt.plot(x_list, sigtb[:int(nQP/nLayers)], color=color1, label = "First layer")
plt.plot(x_list, sigtb[int(nQP/nLayers)*(nLayers-8):int(nQP/nLayers)*(nLayers-7)], color=color2, label = "Middle layer")
plt.plot(x_list, sigtb[int(nQP/nLayers)*(nLayers-1):], color=color3, label = "Last layer")
plt.xticks(np.linspace(-L/2, L/2, L//10))
plt.legend()
plt.grid()


#%%
f1, f2, f3, f4, F1, F2, F3, F4 = forces.internal_forces(Sigma, Hn, Hb)

crossSection = Hn*Hb
sig3D_tt_1 = qp2elem @ F1[0]/crossSection
sig3D_tn_1 = qp2elem @ F1[1]/crossSection
sig3D_tb_1 = qp2elem @ F1[2]/crossSection

x_list = np.linspace(-L/2, L/2, nNodes-1)

nomfig = "sig3D_particule1_v1"
plt.figure()
plt.ylabel("$\sigma$ [MPa]")
plt.xlabel("s [mm]")
plt.plot(x_list, sig3D_tt_1[:nNodes-1], label="$\sigma_{tt}$")
plt.plot(x_list, sig3D_tn_1[:nNodes-1], label="$\sigma_{tn}$")
plt.plot(x_list, sig3D_tb_1[:nNodes-1], label="$\sigma_{tb}$")
plt.xticks(np.linspace(-L/2, L/2, L//10))
plt.legend()
plt.grid()

sig3D_tt_1 = F1[0]/crossSection
sig3D_tn_1 = F1[1]/crossSection
sig3D_tb_1 = F1[2]/crossSection

etiquettes = ["B " + str(k) for k in np.arange(1, nLayers +1 )]  # ["Segment 1", "Segment 2", "Segment 3", "Segment 4", "Segment 5"]
etiquettes.append('')
major_ticks = np.arange(0, nQP, nQP/nLayers -1)

nomfig = "sig3D_particule1_v2"
plt.figure()
plt.ylabel("$\sigma$ [MPa]")
plt.xlabel("s [mm]")
plt.plot(sig3D_tt_1, label="$\sigma_{tt}$")
plt.plot( sig3D_tn_1, label="$\sigma_{tn}$")
plt.plot(sig3D_tb_1, label="$\sigma_{tb}$")
plt.xticks(major_ticks, labels=etiquettes, rotation=45, ha='left', fontsize=12)
plt.legend()
plt.grid()