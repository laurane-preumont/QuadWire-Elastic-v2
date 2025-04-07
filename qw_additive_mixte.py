"""
QuadWire function for calculation during printing with activation of elements during layer and birth of new layers

qwa.additive fonctionne avec une boucle sur tous les éléments activés un par un.
au contraire, additive_mixte repose d'abord sur une boucle sur la naissance de couches,
qui doit gérer les augmentations de taille de matrices,
en plus de la boucle sur l'activation d'éléments.

Tout le fonctionnement de qwa.additive a été recopié dans des fonctions locales pour plus de lisibilité,
et surtout être plus fexible pour intégrer la boucle sur la naissance de couches.

Ces fonctions recopiées doivent donner exactement la meme chose que qwa.additive si elles sont utilisées de la meme facon,
c'est ce qui est fait dans 'additive_classic'.  additive_classic et additive sont quasiment identiques (erreurs < 0.4% d'après la comparaison compare_dicts_with_sparse(results_v1, results_v2) dans debug_mixte.py)

"""
#%% Imports
import time
import csv
import os

import numpy as np
import scipy as sp
from operator import itemgetter

import matplotlib.pyplot as plt

import qw_additive as qwa
from modules import mesh, fem, weld, behavior, plot, thermaldata, forces
from shape import shapeOffset, splitBeads


#%% Function

#############
# affichage #
#############
def initialize_plot(X, L, scfplot, n, b, Elems, Hn, Hb, split_bool):
    ### Plot initialization
    Hn /= (1+split_bool)      # offset = 50% # Hn/2 si split_bool, Hn/1 sinon
    # mean basis vectors on nodes
    Me = sp.sparse.csr_matrix((np.ones(Elems.size), (Elems.flatten(), np.arange(Elems.size))))
    nm = Me @ n;
    nm = nm / np.linalg.norm(nm, axis=1)[:, np.newaxis]
    bm = Me @ b;
    bm = bm / np.linalg.norm(bm, axis=1)[:, np.newaxis]

    # undeformed shape
    x = X[:, :, np.newaxis] + 0.5 * (Hn * nm[:, :, np.newaxis] * np.array([[[-1, 1, -1, 1]]]) + Hb * bm[:, :, np.newaxis] * np.array(
        [[[1, 1, -1, -1]]]))
    fig = plt.figure()
    ax = plt.axes(projection='3d', proj_type='ortho')
    ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scfplot), np.ptp(x[:, 2]*scfplot)))
    y = x

    srf, tri = plot.plotMesh(ax, L, y, Elems, color='none', edgecolor='black', outer=False)
    plt.show()

    return x, srf, tri, ax, fig

def update_plot(U, x, srf, tri, Sigma, Sni, Hn, Hb, clrmap='stt'):
    ## Update deformed shape
    scale = 1 #/(1+split_bool)   #0.5 * np.linalg.norm([Hn, Hb]) / np.max(abs(U), (0, 1, 2)) #0.5 * np.linalg.norm([Hn, Hb]) / np.max(abs(U), (0, 1, 2)) #
    uplot = np.moveaxis(U, (0, 1, 2), (1, 2, 0))
    y = x + scale * uplot

    y = np.moveaxis(y, (0, 1, 2), (0, 2, 1)).reshape(4 * y.shape[0], 3)
    srf.set_verts(y[tri])

    # QuadWire_Elastic version
    nQP = Sni.shape[0]
    if clrmap == "stt" :
        data_plot = Sni.T @ Sigma[0:nQP]/(Hn*Hb)  #sp.sparse.linalg.lsqr(N @ Sni, Sigma[0:nQP])[0][:, None]
    elif clrmap == "stn" :
        data_plot = Sni.T @ Sigma[nQP*3:nQP*4]/(Hn*Hb)  #sp.sparse.linalg.lsqr(N @ Sni, Sigma[nQP*3:nQP*4])[0][:, None]
    elif clrmap == "stb" :
        data_plot = Sni.T @ Sigma[nQP*4:nQP*5]/(Hn*Hb)  #sp.sparse.linalg.lsqr(N @ Sni, Sigma[nQP*4:nQP*5])[0][:, None]

    clr = np.expand_dims(data_plot[:, None] * [1, 1, 1, 1], axis=1) #data_plot[:, None] * [[1, 1, 1, 1]]

    ## Plot stress
    srf.set_array(np.mean(clr.flatten()[tri], axis=1))
    srf.set_clim(np.nanmin(clr), np.nanmax(clr)) #srf.set_clim(clr.min(), clr.max())  # Masked min/max will ignore NaN values
    srf.set_cmap("viridis")

    return clr

def plot_colorbar(clr, srf, tri, ax, fig, clrmap):

    ### Plot parameters
    srf.set_array(np.mean(clr.flatten()[tri], axis=1))
    srf.set_clim(np.nanmin(clr), np.nanmax(clr)) #srf.set_clim(clr.min(), clr.max())  # Masked min/max will ignore NaN values

    ax.set_xlabel('Axe t')
    ax.set_ylabel('Axe n')
    ax.set_zlabel('Axe b')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    colorbar = fig.colorbar(srf, pad=0.15)
    ticks = list(colorbar.get_ticks())+[clr.max()]
    colorbar.set_ticks(ticks)

    if clrmap == "stt" :
        colorbar.set_label('$\sigma_{tt}$ [MPa]')
    elif clrmap == "stn" :
        colorbar.set_label('$\sigma_{tn}$ [MPa]')
    elif clrmap == "stb" :
        colorbar.set_label('$\sigma_{tb}$ [MPa]')
    elif clrmap == "temp" :
        colorbar.set_label('$temperature$ [K]')
    plt.pause(0.1)  # Forces a refresh
    plt.show()

#######
# FEM #
#######
def generate_mesh_data(meshing, offset, discretization, split_bool):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    elemOrder, quadOrder = discretization

    ### Discretization and meshing
    X, Elems, U0 = mesh.mesh_first_bead(L, nNodes, beadType, meshType)
    X, Elems, U0 = mesh.mesh_first_layer(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, layerType, zigzag)
    X, Elems, U0 = mesh.mesh_structure(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, zigzag)

    #meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag # Tupple
    X = shapeOffset.mesh_offset(meshing, offset, X, Elems)

    # save unsplit data for thermal loading
    Xunc_unsplit, uncElems_unsplit = mesh.uncouple_nodes(X, Elems)     # save unsplit data for fake temperature loading
    X_unsplit, Elems_unsplit = X, Elems

    ### Extra split beads to model offset
    if nLayers_v > 1 : # no split beads if single layer
        offset_new, offset_split, overhang_n = splitBeads.stacking_offset_split_beads(meshing, offset)

        if bool(split_bool) :
            ## generate trajectory with extra split beads (as if split beads were regular beads)
            meshing_split = L, Hn, Hb, nLayers_h + 1, nLayers_v, nNodes, beadType, layerType, meshType, zigzag

            X_split, Elems_split, U0_split = mesh.generate_trajectory(meshing_split, zigzag)
            #plot.plotpoints(X_split, True)

            ## apply offset (considering respective bead width for split beads)
            X_split = splitBeads.mesh_offset_split(X_split, Elems_split, meshing_split, offset_split, zigzag)
            #X_split = splitBeads.mesh_offset_split(meshing_split, offset_new, zigzag)  # don't compensate for smaller split beads
            #plot.plotpoints(X_split, True)

            X, Elems, U0 = X_split, Elems_split, U0_split
            meshing = meshing_split
            #offset = offset_split
            offset = offset_new

    else :
        overhang_n = offset.T
    Xunc, uncElems = mesh.uncouple_nodes(X, Elems)

    ### Prevision taille
    nElemsTot = Elems.shape[0]
    nUncNodes = Xunc.shape[0]
    nNodesTot = X.shape[0]
    nParticules = 4
    nCoord = 3
    nNodeDOF = nCoord * nParticules
    nDOF = nNodeDOF * nUncNodes

    return meshing, X, Elems, U0, offset, overhang_n, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit, X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF

def useful_matrices(meshing, X, Elems, nElemsTot, nNodesTot, nUncNodes, Xunc, uncElems, discretization):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    elemOrder, quadOrder = discretization

    ### Useful matrices
    ## Integration matrices
    xiQ, wQ = fem.element_quadrature(quadOrder)
    XIQ, WQ = fem.fullmesh_quadrature(quadOrder, nElemsTot)
    nQP = WQ.shape[0]
    N, Dxi, Ds, J, W, O, qp2elem, elemQ = fem.integration_matrices(X, Elems, elemOrder, quadOrder)
    elem2node = fem.elem2node(nNodes, nLayers_h, nLayers_v)

    ## Projection matrices
    T, Tv, Tc, Ta = fem.alpha2beta_matrices(nNodesTot, nUncNodes)
    t, n, b, P = fem.local2global_matrices(Xunc, uncElems, elemOrder)

    return xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P

def assembly_matrix(Ta, P, Elems, uncElems):
    ## Welding matrices
    Sni, Sn = weld.weldnode_matrices(Elems, uncElems)

    ## Assembly matrix node_alpha to qp_beta
    Assemble = Ta @ P.T @ Sn

    return Sni, Sn, Assemble

###############################
# synthèse problème mécanique #
###############################

def generate_structural_data(meshing, offset, discretization, material, split_bool):
    meshing, X, Elems, U0, offset, overhang_n, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit, X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF = generate_mesh_data(meshing, offset, discretization, split_bool)
    xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P = useful_matrices(meshing, X, Elems, nElemsTot, nNodesTot, nUncNodes, Xunc, uncElems, discretization)
    Sni, Sn, Assemble = assembly_matrix(Ta, P, Elems, uncElems)
    Btot, Ctot, Rxi, Rchi = generate_behavior(meshing, material, split_bool, N, Ds)

    return meshing, X, Elems, U0, offset, overhang_n, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit, X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF, xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P, Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi

#################################
# synthese chargement thermique #
#################################

def generate_thermal_data(path, meshing, discretization, loading, X, X_unsplit, Xunc_unsplit, Elems_unsplit, split_bool):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    dT, loadType = loading
    elemOrder, quadOrder = discretization
    nLayers = nLayers_v * nLayers_h    # temperature is given for unsplit setting
    nPasPerLayer = (nNodes-1) * nLayers
    ### Thermal eigenstrain    #TODO: Tg should be included in material
    Tbuild = 293.15 - dT
    Tsub = 323.15
    Tamb= 293.15
    Tg = 328 #Tbuild*0.9 #Tbuild means material is active as soon as it starts cooling down ie right away #328 #TODO: remove Tbuild

    ## Donnees thermique         # temperature is always given first for unsplit setting, then eventually split
    N_unsplit, qp2elem_unsplit = itemgetter(0, 6)(fem.integration_matrices(X_unsplit, Elems_unsplit, elemOrder, quadOrder))
    nNodesTot_unsplit = X_unsplit.shape[0]
    nUncNodes_unsplit = Xunc_unsplit.shape[0]
    nElemsTot_unsplit = Elems_unsplit.shape[0]
    if path is not None :
        ## Time
        tau_final = 0 #int(np.ceil((nNodes-1)*3/4)) # 0 #nNodes-1 # 10 #
        nPas = nPasPerLayer + tau_final
        ## Data
        dataFiles = [path + '/temp_' + str(k) + ".txt" for k in range(1, nPas+1)]  # remove .zfill(4) if needed or apply generate_thermal.rename_files for compatible naming
        ## Appeler la fonction pour lire les fichiers
        Telem = thermaldata.get_data2tamb(dataFiles, nNodes-1, nLayers_h, nLayers_v, Tbuild) # size (nTimeStep, nElemsTot)
        # Compute the dT_n = T_n - T_n-1
        dTelem = thermaldata.delta_elem_transition_vitreuse(Telem, Tbuild, Tg)
    else :                        #unsplit data would not be needed if this was define before the split_bool modelling (almost right at the beginning)
        ## Data : thermal load
        T_unsplit = fem.alpha2beta_matrices(nNodesTot_unsplit, nUncNodes_unsplit)[0]
        dTalpha = behavior.dTfcn(N_unsplit @ Xunc_unsplit, dT, 'uniform', nLayers_h, nLayers_v)            #-split_bool
        dTbeta = dTalpha @ T_unsplit
        dTmoy = dTbeta[:, 0].copy()[:, np.newaxis]
        # refroidissement linéaire sur longueur Lth donc sur Nth éléments
        Lth = L/10 # longueur du refroidissement en mm         #TODO: choisir la longueur du refroidissement
        Nth = np.sum(X[:nNodes,0]-X[0,0] < Lth)   # nombre d'éléments du refroidissement (longueur variable pour un maillage raffiné...)
        dTmoy[:-Nth] *= 0
        ## Time
        tau_final = 0 #Nth
        nPas = nPasPerLayer + tau_final
        ## Data : broadcast on iterative loading
        dTelem_tk = qp2elem_unsplit @ (dTmoy / np.sum(dTmoy) * quadOrder)     # thermal load profile from 'structure', ie at final time step tk=nTimeStep, size (nElemsTot,1)
        dTelem_tk *= dT # normalized with dTmoy / np.sum(dTmoy) so that all cooling add up to dT parameter
        dTelem = sp.sparse.diags(dTelem_tk[::-1], -np.arange(nElemsTot_unsplit)-1, shape=(nPas + 1, nElemsTot_unsplit), format='csr')  # dTelem_tk broadcast and stored on diagonals to account for successive time steps
        #print(dTelem[:,2].todense())     # thermalData[:, i] = temperature of the ith element throughout the simulation
        Telem = sp.sparse.tril(np.ones(((nPas+1, nPas+1)))) @ dTelem#.todense()   # sp.sparse.tril but +Tbuild requires dense format
        dTelem = dTelem.todense() # to use np.repeat in loop on dTelem[actualTime]
        Telem = Tbuild+Telem.todense() # to use np.repeat in loop on np.repeat(Telem[actualTime], 2)
    if bool(split_bool) : # make split beads experience exactly same temperature as primary beads, assuming thinwall
       kron_duplicate = sp.sparse.kron(sp.sparse.eye(nLayers_v),sp.sparse.kron(np.ones((1,2)),sp.sparse.eye(nNodes-1))) # duplicate groups of nElems=nNodes-1 columns to account for split_beads undergoing the same cooling on both halves
       dTelem = dTelem @ kron_duplicate
       Telem = Telem @ kron_duplicate

    return nPasPerLayer, nPas, dTelem, Telem, Tg, nNodesTot_unsplit

###########################################################
# donnees de soudage de cordons (conditions cinématiques) #
###########################################################

def generate_weldDof_data(meshing, X, offset, overhang_n, split_bool):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    weldDof = weld.welding_conditions(X, meshing)

    # making split weldDof relies on weldDof weights computed without splitbeads
    extended_weldDof = shapeOffset.make_extended_weldDof(meshing, offset, weldDof)

    ### Extra split beads to model offset
    split_bool *= (nLayers_v > 1) # no split beads if single layer
    weldDof_unsplit = weldDof.copy()

    if bool(split_bool) :
        ## generate trajectory with extra split beads (as if split beads were regular beads)
        meshing_split = L, Hn, Hb, beadType, layerType, nLayers_h + 1, nLayers_v, nNodes

        extended_weldDof = splitBeads.extended_split_weldDof(meshing_split, extended_weldDof, overhang_n)  # ie extended_weldDof_split

        weldDof = extended_weldDof[:,:4].astype(int)    # ie weldDof_split

    return weldDof, extended_weldDof, weldDof_unsplit

def generate_welding_matrices(meshing, offset, nNodesTot, weldDof, extended_weldDof, U0, split_bool):

    Sw0 = weld.weldwire_matrix(nNodesTot, weldDof)

    if split_bool:
        # Sw0 = weld.weldwire_matrix(nNodesTot, weldDof_unsplit)
        Sw = splitBeads.offset_split_weldwire_matrix(meshing, extended_weldDof, Sw0)

    else :
        # Sw0 = weld.weldwire_matrix(nNodesTot, weldDof)
        Sw = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw0)

    #Y0 = weld.bcs_matrix(U0, Sw0)
    Y = weld.bcs_matrix(U0, Sw)

    return Sw0, Sw, Y

################
# Comportement #
################
def generate_behavior(meshing, material, split_bool, N, Ds):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    E, nu, alpha, optimizedBehavior = material

    ### Behavior
    ## material parameters
    k0 = E / (3 * (1 - 2 * nu))  # bulk modulus (K)
    mu = E / (2 * (1 + nu))  # shear modulus (G)
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # k0-2/3*mu

    ## Local behavior
    Btot = behavior.derivation_xi(Hn/(1+split_bool), Hb, N, Ds)
    Ctot = behavior.derivation_chi(Hn/(1+split_bool), Hb, N, Ds)

    if optimizedBehavior :
        Rxi, Rchi = behavior.optimization_Rxi_Rchi()
    else :
        Rxi = behavior.homogeneization_Rxi(Hn/(1+split_bool), Hb, lmbda, mu)
        Rchi = behavior.homogenization_Rchi(L, Hn/(1+split_bool), Hb, lmbda, mu, nNodes)

    return Btot, Ctot, Rxi, Rchi

def generate_assembled_behavior(meshing, offset, material, split_bool, Elems, uncElems, nNodesTot, weldDof, extended_weldDof, U0, Ta, P, N, Ds):

    Sni, Sn, Assemble = assembly_matrix(Ta, P, Elems, uncElems)
    Btot, Ctot, Rxi, Rchi = generate_behavior(meshing, material, split_bool, N, Ds)
    Sw0, Sw, Y = generate_welding_matrices(meshing, offset, nNodesTot, weldDof, extended_weldDof, U0, split_bool)

    return Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi, Sw0, Sw, Y

#########################
# Activation d'éléments #
#########################

def activate_element(actualTime, Telem, dTelem, N, Tg, WQ, J):
    ## Stiffness zero
    tol = 1e-12

    Telem_instant = (N @ np.repeat(Telem[actualTime], 2).T).flatten() #  Telem_instant = (N @ Telem[actualTime][:,np.newaxis][:,[0,0]].flatten().T).T
    activeFunction = tol + (1 - tol) * 1 / (1 + np.exp(1 * (Telem_instant - Tg)))  # size (nQP,)
    # matrice W pondérée par activeFunction (Integral weight matrix including local jacobian)
    Wa = sp.sparse.diags((np.array(activeFunction)*(WQ * J[:, np.newaxis]).T).flatten())

    # dTelemmoy doit etre vertical
    dTelemmoy = (N @ np.repeat(dTelem[actualTime], 2).T) # N @ dTelem[actualTime][:,np.newaxis][:,[0,0]].flatten()[:,np.newaxis]
    if len(dTelemmoy.shape) < 2 :  # for some reason shape differs for dataFiles or dTelem_tk
        dTelemmoy = dTelemmoy[:,np.newaxis]

    return Wa, dTelemmoy

#############################################
# résolution méca : initialisation et solve #
#############################################

def initialize_solution(meshing):  # initialize zero values (first layer for displacements, all layers for stress)
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nQP = 2 * (nNodes-1) * nLayers_v * nLayers_h # all layers

    ## Displacement (first layer only)
    # u00 = U0.flatten()[:, np.newaxis].copy()
    # freeDOF = np.isnan(U0.flatten())  # True/False table storing remaining DOF
    # u00[freeDOF] = 0  # cull NaNs

    V = np.zeros((3*4*nNodes,1)) #u00
    U = V.reshape((3, 4, nNodes)) #u00.reshape((3, 4, nNodesTot))

    ### Initialization of the QW variables
    ## Generalised strains
    Eps = sp.sparse.csr_matrix(((6 + 9) * nQP, 1))

    ## Generalised stresses
    Sigma = sp.sparse.csr_matrix(((6 + 9) * nQP, 1))  # 6 strain components and 9 curvature components

    return V, U, Eps, Sigma

def solve_mechanical_problem(Wa, Rxi, Rchi, Btot, Ctot, Assemble, Y, alpha, dTelemmoy):
    ## Assembling behaviour
    Kxi = behavior.assembly_behavior(Rxi, Btot, Wa)
    Kchi = behavior.assembly_behavior(Rchi, Ctot, Wa)
    K = Kxi + Kchi
    K = Assemble.T @ K @ Assemble
    yKy = Y.T @ K @ Y  # Deleting non-useful dof

    ## Force vector
    Eps_thermal = behavior.thermal_eigenstrain(alpha, dTelemmoy)
    f_thermal = Btot.T @ sp.sparse.kron(Rxi, Wa) @ Eps_thermal

    f = f_thermal
    f = Assemble.T @ f
    fbc = f #- K @ u00 #u00 is 0 anyways, unless imposed outside load
    yfbc = Y.T @ fbc  # Deleting non-useful dof

    ## Solve
    vs = sp.sparse.linalg.spsolve(yKy, yfbc)
    v = Y @ vs[:, np.newaxis]
    vUncouple = Assemble @ v  # node beta configuration
    return v, vUncouple, Eps_thermal

################################################
### nouvelles fonctions pour additive_mixte  ###
################################################

def collect_thermal_data(layer, nNodes, dTelem, Telem) :
    ''' collect thermal data for the current mesh
    objectif : tronquer les colonnes aux couches qui sont déjà nées
    layer : nombre de couches déjà nées
    dTelem : matrice des variations de température (lignes=temps, colonnes=éléments)
    Telem : matrice des températures (lignes=temps, colonnes=éléments)
    '''
    nElems = nNodes-1
    dTelem_layer = dTelem[:, :layer*nElems]  # shape (nPas, nElemsTot) for current structure including current layer but excluding future layers
    Telem_layer = Telem[:, :layer*nElems]  # shape (nPas, nElemsTot) for current structure including current layer but excluding future layers

    return dTelem_layer, Telem_layer

def update_offset(offset, U, layer, nNodes) :
    ''' Avant de déposer une nouvelle couche, mise à jour de son offset à partir des déplacements de la couche précédente '''
    offset[1][nNodes*(layer-1):nNodes*(layer-1)+nNodes] += np.average(U[1], axis=0)[-nNodes:]
    return offset


######################################################
### Mise à jour des grandeurs de tailles différentes #
######################################################

def build_index_matrix(meshing, nNodesTot, nNodesTot_new):
    """
    Construit la matrice d'indexation pour mapper l'incrément vers la structure complète.

    Arguments:
        meshing: Paramètres de la géométrie
        nNodesTot: Nombre de noeuds actuels
        nNodesTot_new: Nombre de noeuds de l'incrément
    """
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, *_ = meshing
    nNodesTot_full = nNodes * nLayers_h * nLayers_v

    # Construction des indices
    rows = []
    data = []
    cols = []

    # Pour chaque degré de liberté
    for k in range(3):  # axes t,n,b
        for i in range(4):  # particules
            # Base pour cette composante/particule
            base_new = k*4*nNodesTot_new + i*nNodesTot_new
            base_full = k*4*nNodesTot_full + i*nNodesTot_full

            # Ajout des correspondances
            for j in range(nNodesTot_new):
                rows.append(base_full + j)
                cols.append(base_new + j)
                data.append(1.0)

    # Création de la matrice sparse
    return sp.sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(3*4*nNodesTot_full, 3*4*nNodesTot_new)
    ).tocsr()


def construct_extended_solution(V, v, nNodesTot, nNodesTot_new, nNodes):
    """
    Construit la nouvelle solution en préservant les déplacements existants
    et en ajoutant la contribution de la nouvelle couche.

    Arguments:
        V: Solution actuelle, shape (3*4*nNodesTot, 1)
        v: Incrément de déplacement, shape (3*4*nNodesTot_new, 1)
        nNodesTot: Nombre de noeuds avant extension
        nNodesTot_new: Nombre de noeuds après extension
        nNodes: Nombre de noeuds par couche
    """
    # Initialisation avec les anciennes dimensions
    V_new = np.zeros((3*4*nNodesTot_new, 1))

    # Pour chaque composante (t,n,b) et chaque particule
    for k in range(3):  # axes t,n,b
        for i in range(4):  # particules TL,TR,BL,BR
            # Indices pour accéder aux blocs
            old_start = k*4*nNodesTot + i*nNodesTot
            old_end = old_start + nNodesTot
            new_start = k*4*nNodesTot_new + i*nNodesTot_new
            new_end = new_start + nNodesTot_new

            # Copie des anciens déplacements
            V_new[new_start:new_start+nNodesTot] = V[old_start:old_end]

            # Ajout de l'incrément pour toute la structure
            V_new[new_start:new_end] += v[new_start:new_end]

            # Propagation des déplacements verticaux uniquement
            if k == 2:  # axe b
                last_layer = V[old_end-nNodes:old_end]
                V_new[new_end-nNodes:new_end] += last_layer

    return V_new

def update_strains(Eps, nQP, nQP_new, BtotCtotAssemble, v):
    """
    Met à jour les déformations généralisées pour la structure étendue.

    Arguments:
        Eps: Déformations actuelles ((6+9)*nQP, 1)
        nQP, nQP_new: Nombres de points de quadrature avant/après
        BtotCtotAssemble: Matrices de gradient assemblées
        v: Déplacements assemplé

    Returns:
          Eps_new aux points de Gauss (assemblés)
    """
    # Conversion en array dense si Eps est sparse
    if sp.sparse.issparse(Eps):
        Eps = Eps.toarray()

    # Création du nouveau vecteur
    Eps_new = sp.sparse.csr_matrix((((6+9)*nQP_new, 1)), dtype=np.float64)

    # Conservation des anciennes déformations
    if nQP > 0:
        Eps_new[:(6+9)*nQP] = Eps[:(6+9)*nQP]

    # Calcul de l'incrément sur toute la structure
    dEps = BtotCtotAssemble @ v

    # Ajout de l'incrément
    Eps_new = Eps_new + dEps

    return Eps_new

def update_stresses(Sigma, Eps, nQP, nQP_new, Wa, Rxi, Rchi, alpha, dTelemmoy):
    """
    Met à jour les contraintes généralisées pour la structure étendue.
    Sigma_new aux points de Gauss (assemblés)
    """
    # Conversion en array dense si Sigma est sparse
    if sp.sparse.issparse(Sigma):
        Sigma = Sigma.toarray()

    # Création du nouveau vecteur
    Sigma_new = sp.sparse.csr_matrix((((6+9)*nQP_new, 1)), dtype=np.float64)

    # Conservation des anciennes contraintes
    if nQP > 0:
        Sigma_new[:(6+9)*nQP] = Sigma[:(6+9)*nQP]

    # Construction de la matrice de comportement
    Rtot = sp.sparse.block_diag((
        sp.sparse.kron(Rxi, Wa),
        sp.sparse.kron(Rchi, Wa)
    ))

    # Calcul de la contribution thermique
    Eps_thermal = behavior.thermal_eigenstrain(alpha, dTelemmoy)
    Eps_th_full = sp.sparse.vstack((
        sp.sparse.kron(Rxi, Wa) @ Eps_thermal,
        sp.sparse.csr_matrix((9*nQP_new, 1))
    ))

    # Calcul des contraintes
    dSigma = Rtot @ Eps - Eps_th_full

    # Ajout de l'incrément
    Sigma_new = Sigma_new + dSigma

    return Sigma_new

def update_solution(meshing, V, index, nQP, Eps, Sigma, Wa, v, alpha,
                   dTelemmoy, BtotCtotAssemble, Rxi, Rchi):
    """
    Met à jour la solution lors de l'ajout d'une couche.
    """
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, *_ = meshing

    # Calcul des dimensions
    nNodesTot = V.shape[0]//(3*4)
    nNodesTot_new = v.shape[0]//(3*4)
    nQP_new = nQP * nNodesTot_new//nNodesTot

    # Construction de la matrice d'indexation si nécessaire
    if index.nnz == 0:
        index = build_index_matrix(meshing, nNodesTot, nNodesTot_new)

    # Application de l'incrément via la matrice d'indexation
    v_mapped = index @ v

    # Mise à jour des déplacements
    V_new = V #construct_extended_solution(V, v, nNodesTot, nNodesTot_new, nNodes)
    U_new = V_new.reshape((3, 4, nNodesTot_new))

    # Mise à jour déformations/contraintes en maintenant les types sparse
    Eps_new = update_strains(Eps, nQP, nQP_new, BtotCtotAssemble, v_mapped)
    Sigma_new = update_stresses(Sigma, Eps_new, nQP, nQP_new,
                               Wa, Rxi, Rchi, alpha, dTelemmoy)

    # Vérification que tout est bien en sparse
    if not sp.sparse.issparse(Eps_new):
        Eps_new = sp.sparse.csr_matrix(Eps_new)
    if not sp.sparse.issparse(Sigma_new):
        Sigma_new = sp.sparse.csr_matrix(Sigma_new)

    return V_new, U_new, Eps_new, Sigma_new, index


# def update_solution(meshing, V, index, nQP, Eps, Sigma, Wa, v, alpha, dTelemmoy, BtotCtotAssemble, Rxi, Rchi):   # ancienne version
#     """Met à jour la solution après le dépôt d'une couche.
#
#     # previous shape for first element of current layer :
#     V, U, Eps, Sigma
#     # current shape :
#     Wa, v, vUncouple, Eps_thermal, Btot, Ctot, Rxi, Rchi
#     """
#
#     _, _, _, nLayers_h, nLayers_v, nNodes, *_ = meshing
#
#     nNodesTot = V.shape[0]//3//4
#     nNodesTot_new = v.shape[0]//3//4  # nNodes * layer
#     nQP_tot = BtotCtotAssemble.shape[0]//15  # nLayers_h * nLayers_v * (nNodes-1) * 2
#     ## Updating displacement vector
#     V_new = v
#     # V_new = np.ones((v.shape[0],1))#v
#     # for k in range(3):
#     #     for i in range(4):
#     #         V_new[k*4*nNodesTot_new:(k+1)*4*nNodesTot_new][i*nNodesTot_new:i*nNodesTot_new+nNodesTot] *= 0
#     # V = -0.5*np.ones((nNodesTot*3*4,1))
#     if nNodesTot < nNodesTot_new : # only for first element of layer>1
#         # update new layer with previous layer's vertical displacement
#         for i in range(4) : #particule
#             k = 2  # axis b
#             V_new[k*4*nNodesTot_new:(k+1)*4*nNodesTot_new][i*nNodesTot_new+(nNodesTot_new-nNodes):i*nNodesTot_new+nNodesTot_new] += V[k*4*nNodesTot:(k+1)*4*nNodesTot][(i%2)*nNodesTot+nNodesTot-nNodes:(i%2)*nNodesTot+nNodesTot-nNodes+nNodes] #(V[k*4*nNodesTot:(k+1)*4*nNodesTot][(i%2)*nNodesTot+nNodesTot-nNodes:(i%2)*nNodesTot+nNodesTot-nNodes+nNodes]+V[k*4*nNodesTot:(k+1)*4*nNodesTot][(i%2+2)*nNodesTot+nNodesTot-nNodes:(i%2+2)*nNodesTot+nNodesTot-nNodes+nNodes])/2
#         # update larger matrix with previous displacements
#         for i in range(4) : #particule
#             for k in range(3) : #axis
#                 V_new[k*4*nNodesTot_new:(k+1)*4*nNodesTot_new][i*nNodesTot_new:i*nNodesTot_new+nNodesTot] += V[k*4*nNodesTot:(k+1)*4*nNodesTot][i*nNodesTot:i*nNodesTot+nNodesTot]
#
#     else :
#         V_new += V
#
#     U_new = V_new.reshape((3, 4, nNodesTot_new))
#
#     if nNodesTot < nNodesTot_new or index.nnz == 0 :
#         nNodesTot_full = nNodes * nLayers_h * nLayers_v
#         # Extract existing data, rows, and cols from the index COO matrix
#         # update index COO matrix
#         rows = list(np.tile(np.arange(nNodesTot_new), 3*4) + np.kron(np.arange(3*4)*nNodesTot_full, np.ones(nNodesTot_new, dtype=int)))
#
#         # Recreate the COO matrix with updated data
#         index = sp.sparse.coo_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=((3 * 4 * nNodesTot_full, len(rows))))
#
#     # Update generalized strains
#     dEps = BtotCtotAssemble @ sp.sparse.csr_matrix(index @ v)
#     Eps += dEps
#
#     ## Updating generalized stresses
#     Wa0 = sp.sparse.spdiags(Wa.data, 0, nQP_tot, nQP_tot)
#     Rtot = sp.sparse.block_diag((sp.sparse.kron(Rxi, Wa0), sp.sparse.kron(Rchi, Wa0)))
#
#     Eps_thermal_tot = behavior.thermal_eigenstrain(alpha, sp.sparse.coo_matrix((dTelemmoy.A1, (np.arange(nQP),np.zeros(nQP, dtype=int))),shape=(nQP_tot, 1)))
#     dSigma = Rtot @ dEps - sp.sparse.vstack((sp.sparse.kron(Rxi, Wa0) @ Eps_thermal_tot, sp.sparse.csr_matrix((9 * nQP_tot, 1))))
#
#     Sigma += dSigma
#
#     return V_new, U_new, Eps, Sigma, index


#%% additive mixte

def additive_mixte(path, meshing, offset, loading, discretization, material, toPlot, clrmap, scfplot=10, split_bool=False):

    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    E, nu, alpha, optimizedBehavior = material

    ### general code layout

    # generate structural data for full mesh to initialize plot
    meshing, X, Elems, U0, offset, overhang_n, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit, X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF, xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P, Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi = generate_structural_data(
        meshing, offset, discretization, material, split_bool)
    #BtotCtotAssemble = sp.sparse.vstack((Btot, Ctot)) @ Assemble
    index = sp.sparse.coo_matrix(([], ([], [])), shape=((3 * 4 * nNodes * nLayers_h * nLayers_v, 1)))

    # initialize plot
    x, srf, tri, ax, fig = initialize_plot(X, L, scfplot, n, b, Elems, Hn, Hb, split_bool)

    # generate thermal loading history for full structure
    nPasPerLayer, nPas, dTelem, Telem, Tg, nNodesTot_unsplit = generate_thermal_data(path, meshing, discretization, loading, X, X_unsplit, Xunc_unsplit, Elems_unsplit, split_bool)

    # initialize solution for first layer
    V, U, Eps, Sigma = initialize_solution(meshing)

    # for every layer :
        # generate structural data
        # generate thermal data
        # solve mechanical problem additively
        # update solution

    for layer in range(1,nLayers_v+1):
        meshing_n = L, Hn, Hb, nLayers_h, layer, nNodes, beadType, layerType, meshType, zigzag
        offset_n = update_offset(offset, U, layer, nNodes)[:, :layer*nNodes]

        # generate structural data
        meshing_n, X, Elems, U0, offset_n, overhang_n, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit, X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF = generate_mesh_data(meshing_n, offset_n, discretization,
                                                                                                                                                                                                      split_bool)
        xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P = useful_matrices(meshing, X, Elems, nElemsTot, nNodesTot, nUncNodes, Xunc, uncElems, discretization)

        weldDof, extended_weldDof, weldDof_unsplit = generate_weldDof_data(meshing_n, X, offset_n, overhang_n, split_bool)
        Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi, Sw0, Sw, Y = generate_assembled_behavior(meshing_n, offset_n, material, split_bool, Elems, uncElems, nNodesTot, weldDof, extended_weldDof, U0, Ta, P, N, Ds)

        BtotCtotAssemble = sp.sparse.vstack((Btot, Ctot)) @ Assemble

        # collect thermal data
        dTelem_layer, Telem_layer = collect_thermal_data(layer, nNodes, dTelem, Telem)

        if toPlot:
            # initialize plot
            x, srf, tri, ax, fig = initialize_plot(X, L, scfplot, n, b, Elems, Hn, Hb, split_bool)

        for actualTime in np.arange((layer-1)*(nNodes-1), layer*(nNodes-1)):
            Wa, dTelemmoy = activate_element(actualTime, Telem_layer, dTelem_layer, N, Tg, WQ, J)

            # solve mechanical problem additively
            v, vUncouple, Eps_thermal = solve_mechanical_problem(Wa, Rxi, Rchi, Btot, Ctot, Assemble, Y, alpha, dTelemmoy)
            # update solution
            V, U, Eps, Sigma, index = update_solution(meshing, V, index, nQP, Eps, Sigma, Wa, v, alpha, dTelemmoy, BtotCtotAssemble, Rxi, Rchi)

            if toPlot :
                clr = update_plot(U, x, srf, tri, Sigma.todense(), Sni, Hn, Hb, clrmap='stt')
                plt.pause(0.1)

    if Telem.shape[0] > layer*(nNodes-1) :  # if cooling time steps are included after full structure is printed
        for actualTime in np.arange(layer*(nNodes-1), Telem.shape[0]-1):
            Wa, dTelemmoy = activate_element(actualTime, Telem_layer, dTelem_layer, N, Tg, WQ, J)

            # solve mechanical problem additively
            v, vUncouple, Eps_thermal = solve_mechanical_problem(Wa, Rxi, Rchi, Btot, Ctot, Assemble, Y, alpha, dTelemmoy)
            # update solution
            V, U, Eps, Sigma, index = update_solution(meshing, V, index, nQP, Eps, Sigma, Wa, v, alpha, dTelemmoy, BtotCtotAssemble, Rxi, Rchi)

            if toPlot :
                clr = update_plot(U, x, srf, tri, Sigma.todense(), Sni, Hn, Hb, clrmap='stt')
                plt.pause(0.1)

    # update plot
    clr = update_plot(U, x, srf, tri, Sigma.todense(), Sni, Hn, Hb, clrmap='stt')
    plot_colorbar(clr, srf, tri, ax, fig, clrmap)

    elementEnergyDensity = None

    return U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems, clr



def additive_mixte(path, meshing, offset, loading, discretization, material, toPlot, clrmap, scfplot=10, split_bool=False):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    meshing, X, Elems, U0, offset, overhang_n, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit, X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF, xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P, Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi = generate_structural_data(
        meshing, offset, discretization, material, split_bool)

    start_time_temp = time.time()    # thermal loading

    thermal = generate_thermal_data(path, meshing, discretization, loading, X, X_unsplit, Xunc_unsplit, Elems_unsplit, split_bool)
    elapsed_time_temp = time.time() - start_time_temp
    print(f'La definition du chargement thermique a pris {elapsed_time_temp // 60}min et {elapsed_time_temp % 60}s')

    for layer in range(1,nLayers_v+1):
        meshing_n = L, Hn, Hb, nLayers_h, layer, nNodes, beadType, layerType, meshType, zigzag
        offset_n = update_offset(offset, U, layer, nNodes)[:,:layer*nNodes]

        start_time_temp = time.time()    # layer computation
        U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems, clr = additive_classic(layer, thermal, meshing_n, offset_n, discretization, material, toPlot, clrmap, scfplot=10, split_bool=False)
        elapsed_time_temp = time.time() - start_time_temp
        print(f'Le calcul de la couche layer a pris {elapsed_time_temp // 60}min et {elapsed_time_temp % 60}s')


    return 


def additive_classic(layer, thermal, meshing, offset, discretization, material, toPlot, clrmap, scfplot=10, split_bool=False):
    ''' reproduce the usual qwa.additive function with functions from this file to check if the functions deliver what is expected'''
    nPasPerLayer, nPas, dTelem, Telem, Tg, nNodesTot_unsplit = thermal

    start_time_data = time.time()    # mise en donnée

    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    elemOrder, quadOrder = discretization
    E, nu, alpha, optimizedBehavior = material

    meshing, X, Elems, U0, offset, overhang_n, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit, X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF, xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P, Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi = generate_structural_data(
        meshing, offset, discretization, material, split_bool)
    xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P = useful_matrices(meshing, X, Elems, nElemsTot, nNodesTot, nUncNodes, Xunc, uncElems, discretization)

    weldDof, extended_weldDof, weldDof_unsplit = generate_weldDof_data(meshing, X, offset, overhang_n, split_bool)
    Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi, Sw0, Sw, Y = generate_assembled_behavior(meshing, offset, material, split_bool, Elems, uncElems, nNodesTot, weldDof, extended_weldDof, U0, Ta, P, N, Ds)

    elapsed_time_data = time.time() - start_time_data
    print(f'La mise en donnée a pris {elapsed_time_data // 60}min et {elapsed_time_data % 60}s')

    ### Initialization of the QW variables
    ## Generalised strains
    Eps = np.zeros(((6 + 9) * nQP, 1)) # (nInc, nQP) sigma_xi = Rxi @ (Eps_xi - Eps_thermique)

    ## Displacement
    u00 = U0.flatten()[:, np.newaxis].copy()
    freeDOF = np.isnan(U0.flatten())  # True/False table storing remaining DOF
    u00[freeDOF] = 0  # cull NaNs

    V = u00
    U = u00.reshape((3, 4, nNodesTot))

    ## Generalised stresses
    Sigma = np.zeros(((6 + 9) * nQP, 1))  # 6 strain components and 9 curvature components

    elapsed_time_data = time.time() - start_time_data
    print(f'La mise en donnée a pris {elapsed_time_data // 60}min et {elapsed_time_data % 60}s')

    if toPlot :
        ### Plot initialization
        x, srf, tri, ax, fig = initialize_plot(X, L, scfplot, n, b, Elems, Hn, Hb, split_bool)

    # Start time before the computation
    start_time = time.time()
    print(f'computation has started, {nPas} computation remaining')

    for actualTime in np.arange(nPasPerLayer * (layer-1), nPasPerLayer * layer):
        ## Activated element ?

        Wa, dTelemmoy = activate_element(actualTime, Telem, dTelem, N, Tg, WQ, J)

        v, vUncouple, Eps_thermal = solve_mechanical_problem(Wa, Rxi, Rchi, Btot, Ctot, Assemble, Y, alpha, dTelemmoy)
        ## Updating displacement vector
        V = V + v
        U = U + v.reshape((3, 4, nNodesTot))
        ## Updating generalized strains
        dEps = sp.sparse.vstack((Btot, Ctot)) @ vUncouple
        Eps += dEps
        ## Updating generalized stresses
        Rtot = sp.sparse.block_diag((sp.sparse.kron(Rxi, Wa), sp.sparse.kron(Rchi, Wa)))
        dSigma = Rtot @ dEps - sp.sparse.vstack((sp.sparse.kron(Rxi, Wa) @ Eps_thermal, sp.sparse.csr_matrix((9 * nQP, 1)) ))
        Sigma += dSigma
        #Sigma_time[actualTime] = Sigma

        if toPlot :
            ## Plot deformed shape and stress
            ## Update deformed shape
            clr = update_plot(U, x, srf, tri, Sigma, Sni, Hn, Hb, clrmap='stt')

        plt.pause(0.02)

        #print('actualTime:', actualTime)
    # End time after the computation
    end_time = time.time()
    print(f'computation time : {(end_time - start_time)//60} min and {(end_time - start_time)%60} seconds)')

    if toPlot :
        ### Plot parameters
        srf.set_array(np.mean(clr.flatten()[tri], axis=1))
        srf.set_clim(np.nanmin(clr), np.nanmax(clr)) #srf.set_clim(clr.min(), clr.max())  # Masked min/max will ignore NaN values

        ax.set_xlabel('Axe t')
        ax.set_ylabel('Axe n')
        ax.set_zlabel('Axe b')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        colorbar = fig.colorbar(srf, pad=0.15)
        ticks = list(colorbar.get_ticks())+[clr.max()]
        colorbar.set_ticks(ticks)

        if clrmap == "stt" :
            colorbar.set_label('$\sigma_{tt}$ [MPa]')
        elif clrmap == "stn" :
            colorbar.set_label('$\sigma_{tn}$ [MPa]')
        elif clrmap == "stb" :
            colorbar.set_label('$\sigma_{tb}$ [MPa]')
        elif clrmap == "temp" :
            colorbar.set_label('$temperature$ [K]')
        plt.pause(0.1)  # Forces a refresh
        plt.show()


    ###  Reconstruct internal forces
    f1, f2, f3, f4, F1, F2, F3, F4 = forces.internal_forces(Sigma, Hn, Hb)

    # Eps_xi = Eps[:6*nQP]
    # Eps_chi = Eps[6*nQP:]
    # Sigma_s = Sigma[:6*nQP]
    # Sigma_m = Sigma[6*nQP:]

    ### Energy
    Eps_th = sp.sparse.vstack((Eps_thermal, sp.sparse.csr_matrix((9*nQP, 1)))).toarray()
    qpEnergyDensity = behavior.energyDensity(Eps, Eps_th, Sigma, Rtot, quadOrder, 15)
    elementEnergyDensity = qp2elem @ qpEnergyDensity


    return U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems, clr
