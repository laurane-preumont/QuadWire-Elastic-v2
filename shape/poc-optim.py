"""
Providing proof of concept argumentation
"""
# %%
# clear workspace
from IPython import get_ipython


ipython = get_ipython()
ipython.run_line_magic('reset', '-sf')

# %%
import time
import csv
import os

import numpy as np
import scipy as sp
from operator import itemgetter

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d


from PIL import Image
from matplotlib.patches import Polygon

from scipy.optimize import curve_fit
import os, shutil
import pandas as pd
import seaborn as sns


import qw_structure as qws
import qw_additive as qwa
import qw_additive_mixte as qwam
import generate_thermal
from modules import mesh, fem, weld, behavior, plot, thermaldata, forces, plot
from shape import shapeOffset, splitBeads, shapeOptim, derivateOffset

import generate_thermal
from modules.plot import cross_section, plot_cross_section

# %%
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
L = 10 #0  # length in mm                        #TODO: modif
Hn = 0.1 # width of the section in mm            #TODO: modif
Hb = 0.1 # height of the section in mm          #TODO: modif
beadType = "linear"  # linear, circular, square, quarter_circular, quarter_square, sinus
layerType = "normal"  # default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
meshType = True  # meshRefined
zigzag = False
split_bool = False

# Mesh Parameters
nLayers_h = 1  # number of horizontal layers (beads)
nLayers_v = 2  # number of vertical layers             #TODO: modif
nLayers = nLayers_h * nLayers_v  # number of layers

nNodes = 5 # number of nodes per layers                 #TODO: modif
nNodes_h = nNodes * (nLayers_h+split_bool)
nNodesTot = nNodes * nLayers
nElems = nNodes - 1
nElemsTot = nElems * nLayers

# thermal data
dT = -600 #TODO: remettre le chargement thermique    -60
loadType = "uniform"  # "uniform", "linear", "quad" or "random", "poc" ie top layer cools down other layers don't

path = None          # make dTelem data from 'structure' dTfcn #TODO: check Lth in additive
#path = r".\thermal_data\wall"

# tau_final = int(np.ceil((nNodes-1)*3/4)) #(nNodes-1) # 0 #10 #
# nPas = (nNodes-1) * nLayers_v * (nLayers_h - split_bool) + tau_final    # must be greater or equal to nPas in additive function
# Ncooling = nElems//2
# path = r".\thermal_data\fake_data"      # add .zfill(4) in data_path in qw_additive and check that tau_final and nPas are the same
# generate_thermal.generate_fake_data(nPas, nElemsTot, path, loadType, Ncooling, 293.15-dT)

# Plot
toPlot = True
clrmap = 'stt'
scfplot = 100

# % Data recap as tuple
meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
loading = dT, loadType
loading0 = 0*dT, loadType
discretisation = elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
plotting = toPlot, clrmap, scfplot

offset = shapeOffset.stacking_offset(meshing, "linear", 0, -0.5*(nLayers_v-1))  # offset between successive layers along t and n (2, nLayers_v*nNodes_h)
offset = np.zeros(offset.shape)
layer = 1
actualTime = np.arange((layer-1)*(nNodes-1), layer*(nNodes-1))[-1]+1


# %%

U_mixte, Eps_mixte, Sigma_mixte, elementEnergyDensity_mixte, qp2elem_mixte, nQP_mixte, x_mixte, Elems_mixte, clr_mixte = qwam.additive_mixte(path, meshing, offset, loading, discretisation, material, toPlot, clrmap, scfplot=10,
                                                                                                                                             split_bool=False)
U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems, Un_max, U_data, Eps_data, Sigma_data, clr = qwa.additive(path, meshing, offset, loading, discretisation, material, toPlot, clrmap, scfplot=10, split_bool=False)
U_classic, Eps_classic, Sigma_classic, elementEnergyDensity, qp2elem, nQP, x, Elems, clr = qwam.additive_classic(path, meshing, offset, loading, discretisation, material, toPlot, clrmap, scfplot=10, split_bool=False)

print(np.sum(np.abs(U-U_classic)), np.sum(np.abs(Sigma-Sigma_classic)))
print(np.sum(np.abs(U-U_mixte)), np.sum(np.abs(Sigma-Sigma_mixte)))

# %%
directory = "figures\poc\_brouillon"
# Emplacement où sauvegarder la figure
figure_path = r"C:\Users\preumont\OneDrive\Documents\GitHub\WorkingBook\FIGURES\python"



U_offset0_loading0, Eps, Sigma, nrg, qp2elem, nQP, x_offset0_loading0, Elems = qws.structure(meshing, 0 * offset, loading0, discretisation, material, True, 'temp', 10, False)  # offset=0 impose split_bool=False
U_offset0, Eps, Sigma, nrg, qp2elem, nQP, x_offset0, Elems, Un_max, U_data, Eps_data, Sigma_data, clr = qwa.additive(path, meshing, 0 * offset, loading, discretisation, material, toPlot, clrmap, scfplot,
                                                                                                                     False)  # offset=0 impose split_bool=False
#plt.savefig(figure_path+'\sigma_tt.png', dpi=300, bbox_inches='tight')

U_offset_loading0, Eps, Sigma, nrg, qp2elem, nQP, x_offset_loading0, Elems = qws.structure(meshing, offset, loading0, discretisation, material, True, 'temp', 10, False)  # offset=0 impose split_bool=False
U_offset, Eps, Sigma, nrg, qp2elem, nQP, x_offset, Elems, Un_max, U_data, Eps_data, Sigma_data, clr_offset = qwa.additive(path, meshing, offset, loading, discretisation, material, toPlot, clrmap, scfplot, False)
#plt.savefig(figure_path+'\sigma_tt_offset.png', dpi=300, bbox_inches='tight')

if split_bool :
    U_offset0_split, Eps, Sigma, nrg, qp2elem, nQP, x_offset0_split, Elems = qws.structure(meshing, offset, loading0, discretisation, material, True, 'temp', 10, True)  # offset=0 impose split_bool=False
    U_offset_split, Eps, Sigma, nrg, qp2elem, nQP, x_offset_split, Elems, Un_max, U_data, Eps_data, Sigma_data, clr_offset = qwa.additive(path, meshing, offset, loading, discretisation, material, toPlot, clrmap, scfplot, True)


# %% Define layer parameters and color maps
nLayers_v_min, nLayers_v_max = 1, nLayers_v  # studied layers
bead_range = [nLayers_v_min - 1, nLayers_v_max - 1] * (nLayers_v_min != nLayers_v_max) + [(nLayers_v_min - 1)] * (nLayers_v_min == nLayers_v_max)
green_cmap = cm.get_cmap('Greens')
blue_cmap = cm.get_cmap('Blues')
purple_cmap = cm.get_cmap('Purples')
orange_cmap = cm.get_cmap('Oranges')
num_layers = nLayers_v_max - nLayers_v_min + 1



# %% Positions
# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), gridspec_kw={'height_ratios': [2, 1]})

# Container for legend handles and labels
handles1 = []
labels1 = []
handles2 = []
labels2 = []

# Define positions
position_offset0_loading0 = np.moveaxis(x_offset0_loading0, (0, 1, 2), (2, 0, 1)) + U_offset0_loading0
position_offset0 = np.moveaxis(x_offset0, (0, 1, 2), (2, 0, 1)) + U_offset0
#offset0 to erase offset on scale
position_offset_loading0 = np.moveaxis(x_offset0_loading0, (0, 1, 2), (2, 0, 1)) + U_offset_loading0
position_offset = np.moveaxis(x_offset0, (0, 1, 2), (2, 0, 1)) + U_offset
position_offset_split = np.moveaxis(x_offset0_split, (0, 1, 2), (2, 0, 1)) + U_offset_split

# Loop through each layer and plot data with evolving colors
for i, n in enumerate(range(nLayers_v_min, nLayers_v_max + 1)):
    # Calculate colors for each layer
    green_color = green_cmap(i / num_layers)
    blue_color = blue_cmap(i / num_layers)
    purple_color = purple_cmap(i / num_layers)
    orange_color = orange_cmap(i / num_layers)
    if num_layers == 1:
        green_color, blue_color, purple_color, orange_color = 'green', 'blue', 'purple', 'orange'

    plot_offset0_loading0 = position_offset0_loading0[:2, 0, (n - 1) * nNodes : n * nNodes]
    plot_offset0 = position_offset0[:2, 0, (n - 1) * nNodes : n * nNodes]
    plot_offset_loading0 = position_offset_loading0[:2, 0, (n - 1) * nNodes_h : (n - 1) * nNodes_h + nNodes]
    plot_offset = position_offset[:2, 0, (n - 1) * nNodes_h : (n - 1) * nNodes_h + nNodes]
    plot_offset_split = position_offset_split[:2, 1, (n - 1) * nNodes_h : (n - 1) * nNodes_h + nNodes]

    # Plot each line with distinct labels and colors on ax1
    line_offset0_loading0, = ax1.plot(plot_offset0_loading0[0], plot_offset0_loading0[1], label='$x_{0}$' + f'(Bead {n-1})', color='black')
    line_offset0, = ax1.plot(plot_offset0[0], plot_offset0[1], label='$x$' + f'(Bead {n-1})', color=green_color)
    line_offset_loading0, = ax1.plot(plot_offset_loading0[0], plot_offset_loading0[1], label='$x_{offset_0}$' + f'(Bead {n-1})', color='black')
    line_offset, = ax1.plot(plot_offset[0], plot_offset[1], label='$x_{offset}$' + f'(Bead {n-1})', color=blue_color)
    if split_bool:
        line_offset_split, = ax1.plot(plot_offset_split[0], plot_offset_split[1], label='$x_{offset}$' + f'Split Bead {n-1}', color=purple_color)

    # Plot delta lines on ax2 with their own scale
    U_plot_offset0 = U_offset0[:2, 0, (n - 1) * nNodes : n * nNodes]
    U_plot_offset = U_offset[:2, 0, (n - 1) * nNodes : n * nNodes]

    line_delta, = ax2.plot(U_plot_offset[0] - U_plot_offset0[0], U_plot_offset[1] - U_plot_offset0[1], label='$U_{offset}-U$' + f'(Bead {n-1})', color=orange_color, linestyle=':')
    line_delta_avg, = ax2.plot([x_offset0_loading0[0,0,0] + L / 2], [np.average(U_plot_offset[1]) - np.average(U_plot_offset0[1])], label='$\overline{U_{offset}}-\overline{U}$' + f'(Bead {n-1})', color=orange_color, marker='x')

# Add to ax1 handles and labels
handles1.extend([line_offset0_loading0, line_offset0, line_offset_loading0, line_offset])
labels1.extend(['$x_{0}$ (no thermal loading)', '$x$ (no offset)', '$x_{offset_0}$ (offset without thermal loading)', '$x_{offset}$ (offset)'])
if split_bool:
    handles1.append(line_offset_split)
    labels1.append('$x_{offset}$ Split Bead (offset+split)')

# Add to ax2 handles and labels
handles2.extend([line_delta, line_delta_avg])
labels2.extend(['$U_{offset}-U$', '$\overline{U_{offset}}-\overline{U}$'])

# Customize ax1 labels and title
ax1.set_xlabel('$x_t (mm)$')
ax1.set_ylabel('$x_n (mm)$')
ax1.set_title('Position Data')

# Customize ax2 labels and title
ax2.set_xlabel('$x_t (mm)$')
ax2.set_ylabel('Delta Values')
ax2.set_title('Delta Comparison')

# Set the overall figure title
fig.suptitle('Additive' + ' Split' * split_bool + f' (offset -0.5)\n Particle 0 of bead {bead_range} in thinwall of {nLayers_v} printed layers', fontsize=16)

# Add legends for each subplot
ax1.legend(handles1, labels1, title='Datasets', loc='upper right')
ax2.legend(handles2, labels2, title='Deltas', loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Display the combined plot with subplots
plt.show()


# %% Déplacements
# Create figure and subplots
fig, (ax_t, ax_n) = plt.subplots(2, 1, figsize=(14, 10))  # Two rows, one column

# Container for legend handles and labels
handles = []
labels = []

U0_t_data0 = U0[0, 0, : nNodes]   # 0 values # split_bool=False pour U sans offset
U0_n_data0 = U0[1, 0, : nNodes]   # 0 values # split_bool=False pour U sans offset

# Loop through each layer and plot data with evolving colors
for i, n in enumerate(range(nLayers_v_min, nLayers_v_max + 1)):
    print(n)
    # Calculate colors for each layer in blue and green
    green_color = green_cmap(i / num_layers)
    blue_color = blue_cmap(i / num_layers)
    purple_color = purple_cmap(i / num_layers)
    orange_color = orange_cmap(i / num_layers)
    if num_layers == 1 :
        green_color, blue_color, purple_color, orange_color= 'green', 'blue', 'purple', 'orange'

    # Extract data for each dataset for U_t and U_n
    U_data0 = U[:, 0, (n-1) * nNodes : n * nNodes]  # split_bool=False pour U sans offset
    Uoffset_data0 = Uoffset[:, 0, (n-1) * nNodes_h : (n-1) * nNodes_h + nNodes]
    Uoffset_split_data0 = Uoffset_split[:, 1, (n-1) * nNodes_h : (n-1) * nNodes_h + nNodes]

    # Plot U, U0, and Uoffset with evolving colors on the first subplot (ax_t)
    line_U0, = ax_t.plot(U0_t_data0, label='$U_{0}$'+f'(Bead {n-1})', color='black')
    line_U, = ax_t.plot(U_data0[0,:], label='$U$'+f'(Bead {n-1})', color=green_color)
    line_Uoffset, = ax_t.plot(Uoffset_data0[0,:], label='$U_{offset}$'+f'(Bead {n-1})', color=blue_color)
    if split_bool :
        line_Uoffset_split, = ax_t.plot(Uoffset_split_data0[0,:], label='$U_{offset}$'+f'(Split Bead {n-1})', color=purple_color)
    line_delta, = ax_t.plot(Uoffset_data0[0,:]-U_data0[0,:], label='$U_{offset}-U$'+f'(Bead {n-1})', color=orange_color, linestyle=':')
    line_delta_avg, = ax_t.plot(L/2, np.average(Uoffset_data0[0,:])-np.average(U_data0[0,:]), label='$\overline{U_{offset}}-\overline{U}$'+f'(Bead {n-1})', color=orange_color, marker='x')

    # Plot U, U0, and Uoffset with evolving colors on the second subplot (ax_n)
    ax_n.plot(U0_n_data0, color='black')
    ax_n.plot(U_data0[1,:], color=green_color)
    ax_n.plot(Uoffset_data0[1,:], color=blue_color)
    if split_bool :
        ax_n.plot(Uoffset_split_data0[1,:], color=purple_color)
    ax_n.plot(Uoffset_data0[1,:]-U_data0[1,:], color=orange_color, linestyle=':')
    ax_n.plot(L/2, np.average(Uoffset_data0[1,:])-np.average(U_data0[1,:]), color=orange_color, marker='x')

# Append handles and labels for the legend
if split_bool :
    handles.extend([line_U0, line_U, line_Uoffset, line_Uoffset_split, line_delta, line_delta_avg])
    labels.extend(['$U_{0}$'+f' (no thermal loading)', '$U$'+f' (no offset)', '$U_{offset}$'+f' (offset)', '$U_{offset}$'+f'Split Bead (offset+split)', '$U_{offset}-U$', '$\overline{U_{offset}}-\overline{U}$'])
else :
    handles.extend([line_U0, line_U, line_Uoffset, line_delta, line_delta_avg])
    labels.extend(['$U_{0}$'+f' (no thermal loading)', '$U$'+f' (no offset)', '$U_{offset}$'+f' (offset)', '$U_{offset}-U$', '$\overline{U_{offset}}-\overline{U}$'])

# Customize subplots
ax_t.set_xlabel('t')
ax_t.set_ylabel('$U_t (mm)$')
ax_n.set_xlabel('t')
ax_n.set_ylabel('$U_n (mm)$')

# Set the title for the figure
fig.suptitle('additive'+' split'*split_bool+f' (offset -0.5) \n Particle 0 of bead {bead_range} in thinwall of {nLayers_v} printed layers', fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 0.85, 0.95])

# Create a shared legend outside the figure
fig.legend(handles, labels, title='Datasets', bbox_to_anchor=(0.88, 0.5), loc='center')

# Display the combined plot
plt.show()


# Sauvegarde de la figure
#plt.savefig(figure_path+'\comparaison_U_offset_.png', dpi=300, bbox_inches='tight')
#print(f'saved figure at {figure_path}')

# %%

# # Display sigma difference due to offset
# ### Discretization and meshing
# X, Elems, U0 = mesh.mesh_first_bead(L, nNodes, beadType, meshType)
# X, Elems, U0 = mesh.mesh_first_layer(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, layerType, zigzag)
# X, Elems, U0 = mesh.mesh_structure(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, zigzag)
# X = shapeOffset.mesh_offset(meshing, offset, X, Elems, zigzag)
#
# Xunc, uncElems = mesh.uncouple_nodes(X, Elems)
# t, n, b, P = fem.local2global_matrices(Xunc, uncElems, elemOrder)
# ### Plot initialization
#
# # mean basis vectors on nodes
# Me = sp.sparse.csr_matrix((np.ones(Elems.size), (Elems.flatten(), np.arange(Elems.size))))
# nm = Me @ n;
# nm = nm / np.linalg.norm(nm, axis=1)[:, np.newaxis]
# bm = Me @ b;
# bm = bm / np.linalg.norm(bm, axis=1)[:, np.newaxis]
#
# # undeformed shape
# x = X[:, :, np.newaxis] + 0.5 * (Hn * nm[:, :, np.newaxis] * np.array([[[-1, 1, -1, 1]]]) + Hb * bm[:, :, np.newaxis] * np.array(
#     [[[1, 1, -1, -1]]]))
# fig = plt.figure()
# ax = plt.axes(projection='3d', proj_type='ortho')
# ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scfplot), np.ptp(x[:, 2]*scfplot)))
# y = x
#
# srf, tri = plot.plotMesh(ax, L, y, Elems, color='none', edgecolor='black', outer=False)
# plt.show()
# # data with offset
# scale= 0.5 * np.linalg.norm([Hn, Hb]) / np.max(abs(Uoffset), (0, 1, 2)) #50   #
# uplot = np.moveaxis(offset, (0, 1, 2), (1, 2, 0))
# y = x + scale * uplot
# y = np.moveaxis(y, (0, 1, 2), (0, 2, 1)).reshape(4 * y.shape[0], 3)
# srf.set_verts(y[tri])
# clr -= clr_offset
# srf.set_array(np.mean(clr.flatten()[tri], axis=1))
# srf.set_clim(np.nanmin(clr), np.nanmax(clr))
#
# ### Plot parameters
# ax.set_xlabel('Axe t')
# ax.set_ylabel('Axe n')
# ax.set_zlabel('Axe b')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# colorbar.set_label('$\sigma_{tt}$ [MPa]')
#
# colorbar = plt.colorbar(srf, pad=0.15)

# %%  Construction approximative d'un U additif

def poc_additive(meshing, offset, zigzag, loading, discretisation, material, toPlot, clrmap="stt", scfplot=10, split_bool=False):
    L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes, meshType = meshing

    U_additive = np.zeros((3, 4, nLayers_v*(nLayers_h+split_bool)*nNodes))
    #U, Eps, Sigma, nrg, qp2elem, nQP, x, Elems = qws.structure(meshing, offset, zigzag, loading, discretisation, material, toPlot, clrmap, scfplot, split_bool, meshType)

    for n in range(2,nLayers_v+1) :       #1+split_bool (split_bool only works for min 2 layers)
        print(n)
        meshing = L, Hn, Hb, beadType, layerType, nLayers_h, n, nNodes, meshType
        offset_n = offset[:,:n*nNodes] # offset between successive layers along t and n (2, nLayers_v*nNodes_h)
        U, Eps, Sigma, nrg, qp2elem, nQP, x, Elems = qws.structure(meshing, offset_n, loading, discretisation, material, toPlot, clrmap, scfplot, split_bool)
        U_additive[:,:,:n*(nLayers_h+split_bool)*nNodes] += U
    
    for n in range(2,nLayers_v+1) :
        for i in range(4) :
            U_additive[2,i,n+1:] += np.average(U_additive[2,:,:], axis=0)

    return U_additive#, Eps, Sigma, nrg, qp2elem, nQP, x, Elems

# %% study split_bead effect : single element, two layers, poc thermal loading

#split_bool=False
U_unsplit, Eps, Sigma, nrg, qp2elem, nQP, x0_unsplit, Elems = qws.structure(meshing, offset, loading, discretisation, material, toPlot, 'temp', scfplot, False)
#U_unsplit = poc_additive(meshing, offset, zigzag, loading, discretisation, material, toPlot, clrmap, scfplot, False)
#split_bool=True
U_split, Eps, Sigma, nrg, qp2elem, nQP, x0_split, Elems = qws.structure(meshing, offset, loading, discretisation, material, toPlot, 'temp', scfplot, True)
#U_split = poc_additive(meshing, offset, zigzag, loading, discretisation, material, toPlot, clrmap, scfplot, True)



node_mid = nNodes//2

#cross_section(U_unsplit, x0_unsplit, node_mid, nNodes)
#cross_section(U_split, x0_split, node_mid, nNodes)

fig, ax = plt.subplots()
cross_section_U_unsplit = U_unsplit[:,:,node_mid::nNodes]
cross_section_U_split = U_split[:,:,node_mid::nNodes]
cross_section_x0_unsplit = x0_unsplit[node_mid::nNodes,:,:]
cross_section_x0_split = x0_split[node_mid::nNodes,:,:]

scale = 100
cross_section_x_unsplit = cross_section_x0_unsplit + scale * np.moveaxis(cross_section_U_unsplit, (0,1,2), (1, 2, 0))   #from (3, 4, nLayers) to (nLayers, 3, 4)
cross_section_x_split = cross_section_x0_split + scale * np.moveaxis(cross_section_U_split, (0,1,2), (1, 2, 0))   #from (3, 4, nLayers) to (nLayers, 3, 4)

plot_cross_section(ax, cross_section_x_unsplit, plt.cm.spring, True)
plot_cross_section(ax, cross_section_x0_unsplit, plt.cm.Greys, True)
plot_cross_section(ax, cross_section_x_unsplit, plt.cm.spring, False)
plot_cross_section(ax, cross_section_x_split, plt.cm.winter, False)

# Add labels, title, and grid
plt.xlabel('n (mm)')
plt.ylabel('b (mm)')
plt.title('Beads defined by particle position')
plt.grid(True)
plt.axis('equal')  # Ensure equal scaling for x and y axes

plt.show()

# %% Create additive data for growing nLayers_v
list_layers = list(range(1,nLayers_v+1))

# Mise en donnée température

meshing_data, offset_data = [], []
x0_data = []
U_data, Eps_data, Sigma_data, nrg_data, qp2elem_data, nQP_data, x_data, Elems_data = [], [], [], [], [], [], [], []
for n in list_layers :
    print(n)
    meshing_n = L, Hn, Hb, beadType, layerType, nLayers_h, n, nNodes
    offset_n = offset[:,:n*nNodes_h]
    x0 = qws.structure(meshing_n, offset_n, loading0, discretisation, material, toPlot, clrmap, scfplot, split_bool)[6]

    U, Eps, Sigma, nrg, qp2elem, nQP, x, Elems, Un_max, U_d, Eps_d, Sigma_d, clr = qwa.additive(path, meshing_n, offset_n, loading, discretisation, material, toPlot, clrmap, scfplot, split_bool)
    #U, Eps, Sigma, nrg, qp2elem, nQP, x, Elems = poc_additive(meshing, offset_n, zigzag, loading, discretisation, material, toPlot, clrmap, scfplot, split_bool, meshType)

    meshing_data.append(meshing_n), offset_data.append(offset_n)
    x0_data.append(x0)
    U_data.append(U), Eps_data.append(Eps), Sigma_data.append(Sigma), nrg_data.append(nrg), qp2elem_data.append(qp2elem), nQP_data.append(nQP), x_data.append(x), Elems_data.append(Elems)


# %% plot sigma maps with growing printed layers  (poc additive only returns additive U, Sigma is not cumulative)
data_i = [i for i in range(len(list_layers))]

def save_sigma(data_i, Sigma_data, qp2elem_data, nQP_data, x_data, Elems_data, output_dir, L, Hn, Hb, field, scfplot=10, start=True) :

    # Check if the directory exists
    if os.path.exists(output_dir):
        # Clear the directory by deleting its contents
        shutil.rmtree(output_dir)
    # Create the directory to save images if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List to store filenames of saved images
    image_filenames = []

    for i in data_i: #  range(len(list_layers))
        # Create the plot
        plot.plot_sigma(Sigma_data[i], qp2elem_data[i], nQP_data[i], x_data[i], Elems_data[i], L, Hn, Hb, field, scfplot)
        # Save the plot as an image
        filename = os.path.join(output_dir, f'plot_{i}.png')
        plt.savefig(filename)
        image_filenames.append(filename)

        # Close the plot window to avoid display
        plt.close()
    if start :
        # Open the folder with the saved images
        os.startfile(output_dir)
    return image_filenames

if 'poc' in directory :
    print('Sigma is not cumulative with poc additive, only U is')
else :
    for field in ['stt', 'stn', 'stb'] :
        output_dir = directory+"\plot_sigma_maps_"+field
        image_filenames = save_sigma(data_i, Sigma_data, qp2elem_data, nQP_data, x_data, Elems_data, output_dir, L, Hn, Hb, field, scfplot=10, start=True)
        plot.save_gif(output_dir, image_filenames, gif_filename="plots_animation.gif", duration=len(data_i)*30, loop=2, start=True)

# %% plot U0 displacements for beads in [[nLayers_min, nLayers_max]] range with growing printed layers

nLayers_v_min, nLayers_v_max = 1, list_layers[-1]
data_i =  [i for i in range(len(list_layers))]
output_dir = directory+"\plot_U0"+f'_[{nLayers_v_min}-{nLayers_v_max}]'

def save_U0_beads(title, data_i, meshing_data, U_data, nLayers_v_min, nLayers_v_max, split_bool, output_dir, start=True) :

    # Check if the directory exists
    if os.path.exists(output_dir):
        # Clear the directory by deleting its contents
        shutil.rmtree(output_dir)
    # Create the directory to save images if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List to store filenames of saved images
    image_filenames = []

    # get the y-limits from the last dataset
    ylim = plot.plot_U0_layers('', meshing_data[data_i[-1]], U_data[data_i[-1]], nLayers_v_min, nLayers_v_max, split_bool)

    for i in data_i: #  range(len(list_layers))
        # Create the plot
        plot.plot_U0_layers(title, meshing_data[i], U_data[i], nLayers_v_min, nLayers_v_max, split_bool, ylim)

        # Save the plot as an image
        filename = os.path.join(output_dir, f'plot_{i}.png')
        plt.savefig(filename)
        image_filenames.append(filename)

        # Close the plot window to avoid display
        plt.close()

    if start :
        # Open the folder with the saved images
        os.startfile(output_dir)

    return image_filenames


                    #'\n Fake Additive '  'offset 50% \n Fake Additive ' 'split_bool offset 50% \n POC Additive '
image_filenames = save_U0_beads('offset 50% \n fake Additive ', data_i, meshing_data, U_data, nLayers_v_min, nLayers_v_max, split_bool, output_dir)
plot.save_gif(output_dir, image_filenames, gif_filename="plots_animation.gif", duration=len(data_i)*30, loop=2, start=True)


# %% plot cross-section middle of thinwall
node_mid = nNodes//10
output_dir = directory+"\plot_cross_section_comparative"+f'node{node_mid}sur{nNodes}'

for i in data_i[:10] :
    cross_section(U_data[i], x0_data[i], node_mid, nNodes)

# Check if the directory exists
if os.path.exists(output_dir):
    # Clear the directory by deleting its contents
    shutil.rmtree(output_dir)
# Create the directory to save images if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List to store filenames of saved images
image_filenames = []


def plot_cross_section_comparative(cross_section_U, cross_section_x0, ylim=None) :
    fig, ax = plt.subplots()
    nLayers_v = cross_section_U.shape[-1]
    denominator = np.max(abs(cross_section_U), (0, 1, 2))
    if denominator != 0 :
        scale = .25 * np.linalg.norm([Hn, Hb]) / denominator
    else :
        scale = 0
    cross_section_x = cross_section_x0 + scale*np.moveaxis(cross_section_U, (0,1,2), (1, 2, 0))   #from (3, 4, nLayers) to (nLayers, 3, 4)

    if ylim is None :
        print('ylim is auto')
        ylim = plot.plot_cross_section(ax, cross_section_x, plt.cm.plasma, True)
    else :
        print('ylim is given')

    plot.plot_cross_section(ax, cross_section_x0, plt.cm.Greys, False, ylim)
    plot.plot_cross_section(ax, cross_section_x, plt.cm.plasma, False, ylim)

    # annotate last bead displacements
    average_displacement = np.average(cross_section_U[:, :, -1], axis=1) # top bead displacement
    annotation = f'$u_n = ($ {average_displacement[1]:.1e}$)$\n $u_b = (${average_displacement[2]:.1e}$)$'
    ax.text(np.mean(cross_section_x[-1, 1, :]), np.mean(cross_section_x[-1, 2, :]),
            annotation,
            fontsize=10, ha='left', va='top', color='black')

    # Add labels, title, and grid
    plt.xlabel('n (mm)')
    plt.ylabel('b (mm)')
    plt.title('Beads defined by particle position')
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling for x and y axes

    plt.show()

    return ylim

# get the y-limits from the last dataset
ylim = plot_cross_section_comparative(U_data[-1][:,:,node_mid::nNodes], x0_data[-1][node_mid::nNodes,:,:])

for i in data_i[:10]: #  range(len(list_layers))
    cross_section(U_data[i], x0_data[i], node_mid, nNodes)
    # Save the plot as an image
    filename = os.path.join(output_dir, f'plot_{i}.png')
    plt.savefig(filename)
    image_filenames.append(filename)

    # Close the plot window to avoid display
    plt.close()

plot.save_gif(output_dir, image_filenames, gif_filename="plots_animation.gif", duration=len(data_i)*30, loop=2, start=True)



output_dir = directory+"\plot_cross_section_comparative"

def save_cross_section_comparative(U_data, x0_data, output_dir, start=True):
    # Check if the directory exists
    if os.path.exists(output_dir):
        # Clear the directory by deleting its contents
        shutil.rmtree(output_dir)
    # Create the directory to save images if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List to store filenames of saved images
    image_filenames = []

    # get the y-limits from the last dataset
    ylim = plot_cross_section_comparative(U_data[-1][:,:,node_mid::nNodes], x0_data[-1][node_mid::nNodes,:,:])

    for i in data_i: #  range(len(list_layers))
        cross_section_U = U_data[i][:,:,node_mid::nNodes]
        cross_section_x0 = x0_data[i][node_mid::nNodes,:,:]   #x0_data[-1]
        # Create the plot
        plot_cross_section_comparative(cross_section_U, cross_section_x0, ylim)
        # Save the plot as an image
        filename = os.path.join(output_dir, f'plot_{i}.png')
        plt.savefig(filename)
        image_filenames.append(filename)

        # Close the plot window to avoid display
        plt.close()

    if start :
        # Open the folder with the saved images
        os.startfile(output_dir)

    return image_filenames

image_filenames = save_cross_section_comparative(U_data, x0_data, output_dir, start=True)
plot.save_gif(output_dir, image_filenames, gif_filename="plots_animation.gif", duration=len(data_i)*30, loop=2, start=True)

# %% plot trajectory of bead n compared to initial trajectory

n = 1  # what bead are we observing  /!\ n < list_layers[i] otherwise bead has not yet been printed

output_dir = directory+f"\plot_trajectory_comparative_bead-{n}"

def plot_trajectory(n, i, list_layers, U_data, x0_data, ylim=None, padding=0.1):
    """
    Plots the trajectory of a specific bead through layers.

    Parameters:
    -----------
    n : int
        Bead index.
    i : int
        Layer index.
    list_layers : list
        List of layer numbers.
    U_data : array-like
        Displacement data.
    x0_data : array-like
        Initial position data.
    ylim : tuple, optional
        Y-axis limits.
    padding : float, optional
        Padding fraction for y-limits.

    Returns:
    --------
    ylim : tuple
        The y-limits after plotting.
    """
    nLayers_v = list_layers[-1]

    U = U_data[i]
    #x0 = x0_data[i]

    x = x0 + np.moveaxis(U, (0, 1, 2), (1, 2, 0))  # from (3, 4, nLayers) to (nLayers, 3, 4)

    trajectory_U = np.average(U[:2, :, (n-1) * nNodes_h:n * nNodes_h], axis=1)
    trajectory_x = np.average(x[(n-1) * nNodes_h:n * nNodes_h, :2, :], axis=2)
    trajectory_x0 = np.average(x0[(n-1) * nNodes_h:n * nNodes_h, :2, :], axis=2)

    fig, ax = plt.subplots()

    # Choose a colormap
    cmap = plt.cm.plasma
    colors = cmap(np.arange(1, nLayers_v+1)/nLayers_v)

    # Plot the trajectories
    plt.plot(trajectory_x0[:nNodes, 0], trajectory_x0[:nNodes, 1], label='Target', color='black')
    plt.plot(trajectory_x[:nNodes, 0], trajectory_x[:nNodes, 1], label='As-built', color=colors[n])

    # Calculate y-limits
    ymin = np.min((trajectory_x[:nNodes, 1], trajectory_x0[:nNodes, 1]))
    ymax = np.max((trajectory_x[:nNodes, 1],trajectory_x0[:nNodes, 1]))

    # Add padding to the calculated y-limits
    if ylim is None:
        range_y = ymax - ymin
        ymin -= padding * range_y #np.max((padding * range_y, 1e-12))
        ymax += padding * range_y #np.max((padding * range_y, 1e-12))
        ylim = (ymin, ymax)
    else:
        (ymin, ymax) = ylim

    # Set the y-limits with the added padding
    plt.ylim(ymin, ymax)

    # Add labels, title, and grid
    plt.xlabel('t (mm)')
    plt.ylabel('n (mm)')
    plt.title(f'Trajectory of layer {n} after printing of {list_layers[i]} layers')

    # Add grid and set equal axis scaling
    plt.grid(True)
    plt.subplots_adjust(right=0.8)  # Leave 20% space for the legend
    fig.legend(title='Trajectories', bbox_to_anchor=(0.90, 0.5), loc='center')

    # Show the plot
    plt.show()

    return ylim

def save_trajectory_comparative(n, list_layers, data_i, U_data, x0_data, output_dir, start=True):
    """
    Saves comparative trajectory plots for a given bead across multiple layers.

    Parameters:
    -----------
    n : int
        Bead index.
    U_data : array-like
        Displacement data.
    x0_data : array-like
        Initial position data.
    output_dir : str
        Output directory to save images.
    list_layers : list
        List of layers for trajectory comparison.
    data_i : list
        List of data indices for plotting.
    start : bool, optional
        Whether to open the directory after saving files.

    Returns:
    --------
    image_filenames : list
        List of saved image filenames.
    """
    # Clear and create the directory to save images
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # List to store filenames of saved images
    image_filenames = []

    # Get the y-limits from the last dataset
    ylim = plot_trajectory(n, -1, list_layers, U_data, x0_data)

    # Iterate over the data indices starting from the observed bead
    for i in data_i[n-1:]:
        # Create the plot
        plot_trajectory(n, i, list_layers, U_data, x0_data, ylim)

        # Save the plot as an image
        filename = os.path.join(output_dir, f'bead-{n}_layer-{i}.png')
        plt.savefig(filename)
        image_filenames.append(filename)

        # Clear the figure to free memory
        plt.close()

    # Optionally open the output directory
    if start:
        os.startfile(output_dir)

    return image_filenames

image_filenames = save_trajectory_comparative(n, list_layers, data_i, U_data, x0_data, output_dir, start=True)
plot.save_gif(output_dir, image_filenames, gif_filename="plots_animation.gif", duration=len(data_i)*30, loop=2, start=True)

# %%
#
# # %%
# print("sigma_tt", 'sigma_tn', "sigma_tb")
# sigma_tt_avg = []
# sigma_tn_avg = []
# sigma_tb_avg = []
# width = [Hn/100, Hn/10, Hn/3, Hn/2, Hn, Hn*2, Hn*3, Hn*10, Hn*100]
# for _Hn in width :
#     meshing = L, _Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes, meshType
#
#     U, Eps, Sigma, nrg, qp2elem, nQP, x, Elems = qws.structure_bifil_overhang(meshing, offset, zigzag, loading, discretisation, material, False, clrmap, scfplot)
#
#     sigma_ti = []
#     #### divide sigma by cross-section : /(_Hn*Hb)
#     for i in range(3) :
#         projection_index = [0, 3, 4]
#         sigma_ti.append(qp2elem @ Sigma[projection_index[i] * nQP:(projection_index[i] + 1) * nQP])
#     sigma_ti_avg = np.mean(np.abs(np.array(sigma_ti)), axis=1)
#     sigma_tt_avg.append(sigma_ti_avg[0][0]/(_Hn*Hb))
#     sigma_tn_avg.append(sigma_ti_avg[1][0]/(_Hn*Hb))
#     sigma_tb_avg.append(sigma_ti_avg[2][0]/(_Hn*Hb))
#     print(_Hn/Hn, '\n', sigma_tt_avg[-1], '\n',  sigma_tn_avg[-1], '\n',  sigma_tb_avg[-1], '\n')
#
# if False :
#     # Create a figure with three subplots (1 row, 3 columns)
#     fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
#
#     # Plot the first dataset in the first subplot
#     axs[0].plot(width, sigma_tt_avg, color='b')
#     axs[0].set_title('Sigma TT Avg / (Hn*Hb)')
#     axs[0].set_xlabel('Width')
#     axs[0].set_ylabel('Sigma TT Avg / (Hn*Hb)')
#
#     # Plot the second dataset in the second subplot
#     axs[1].plot(width, sigma_tn_avg, color='r')
#     axs[1].set_title('Sigma TN Avg / (Hn*Hb)')
#     axs[1].set_xlabel('Width')
#     axs[1].set_ylabel('Sigma TN Avg / (Hn*Hb)')
#
#     # Plot the third dataset in the third subplot
#     axs[2].plot(width, sigma_tb_avg, color='g')
#     axs[2].set_title('Sigma TB Avg / (Hn*Hb)')
#     axs[2].set_xlabel('Width')
#     axs[2].set_ylabel('Sigma TB Avg / (Hn*Hb)')
#
#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#
#     # Display the plot
#     plt.suptitle('Three Datasets in Separate Subfigures', y=1.05)
#     plt.show()
#
# sigma_tt_avg = []
# sigma_tn_avg = []
# sigma_tb_avg = []
# for _Hn in width :
#     meshing = L, _Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes, meshType
#
#     U, Eps, Sigma, nrg, qp2elem, nQP, x, Elems = qws.structure_bifil_overhang(meshing, offset, zigzag, loading, discretisation, material, False, clrmap, scfplot)
#     sigma_ti = []
#     #### sigma not divided by cross-section
#     for i in range(3) :
#         projection_index = [0, 3, 4]
#         sigma_ti.append(qp2elem @ Sigma[projection_index[i] * nQP:(projection_index[i] + 1) * nQP])
#     sigma_ti_avg = np.mean(np.abs(np.array(sigma_ti)), axis=1)
#     sigma_tt_avg.append(sigma_ti_avg[0][0])
#     sigma_tn_avg.append(sigma_ti_avg[1][0])
#     sigma_tb_avg.append(sigma_ti_avg[2][0])
#     print(_Hn/Hn, '\n', sigma_tt_avg[-1], '\n',  sigma_tn_avg[-1], '\n',  sigma_tb_avg[-1], '\n')
#
# if False :
#     # Create a figure with three subplots (1 row, 3 columns)
#     fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
#
#     # Plot the first dataset in the first subplot
#     axs[0].plot(width, sigma_tt_avg, color='b')
#     axs[0].set_title('Sigma TT Avg')
#     axs[0].set_xlabel('Width')
#     axs[0].set_ylabel('Sigma TT Avg')
#
#     # Plot the second dataset in the second subplot
#     axs[1].plot(width, sigma_tn_avg, color='r')
#     axs[1].set_title('Sigma TN Avg')
#     axs[1].set_xlabel('Width')
#     axs[1].set_ylabel('Sigma TN Avg')
#
#     # Plot the third dataset in the third subplot
#     axs[2].plot(width, sigma_tb_avg, color='g')
#     axs[2].set_title('Sigma TB Avg')
#     axs[2].set_xlabel('Width')
#     axs[2].set_ylabel('Sigma TB Avg')
#
#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#
#     # Display the plot
#     plt.suptitle('Three Datasets in Separate Subfigures', y=1.05)
#     plt.show()
#
#
#
# # %%  Affichage du mesh déformé vs sans chargement
#
# loading0 = 0, 'uniform'
# x0 = qws.structure(meshing, offset, zigzag, loading0, discretisation, material, toPlot, clrmap, scfplot, split_bool, meshType)[6]
# #U_0, Eps_0, Sigma_0, elementEnergyDensity_0, qp2elem_0, nQP_0, x_0, Elems_0 = qwa.additive(r".\thermal_data\fake_data", meshing, offset, zigzag, loading0, discretisation, material, toPlot, clrmap, scfplot, split_bool, meshType)[:8]
#
# U, Eps, Sigma, nrg, qp2elem, nQP, x, Elems = qws.structure(meshing, offset, zigzag, loading, discretisation, material, toPlot, clrmap, scfplot, split_bool)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d', proj_type='ortho')
#
# ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scfplot), np.ptp(x[:, 2])*scfplot))
# clr=[]
# srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='black', clrfun=clr, outer=False)
# srf, tri = plot.plotMesh(ax, L, x0, Elems, color='none', edgecolor='blue', clrfun=clr, outer=False)
#
# plt.show()
#
