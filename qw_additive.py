"""
QuadWire function for calculation during printing
"""
#%% Imports
import time
import csv
import os

import numpy as np
import scipy as sp
from operator import itemgetter

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from modules import mesh, fem, weld, behavior, plot, thermaldata, forces
from shape import shapeOffset, splitBeads


#%% Function

def additive(path, meshing, offset, loading, discretization, material, toPlot=True, clrmap='stt', scfplot=10, split_bool=False):
    start_time_data = time.time()    # mise en donnée

    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    dT, loadType = loading
    elemOrder, quadOrder = discretization
    E, nu, alpha, optimizedBehavior = material

    nNodes_h = nNodes*nLayers_h

    ### Discretization and meshing
    X, Elems, U0 = mesh.generate_trajectory(meshing)
    weldDof = weld.welding_conditions(X, meshing)

    #meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag # Tupple
    X = shapeOffset.mesh_offset(meshing, offset, X, Elems)

    # save unsplit data for thermal loading
    Xunc_unsplit, uncElems_unsplit = mesh.uncouple_nodes(X, Elems)     # save unsplit data for fake temperature loading
    X_unsplit, Elems_unsplit = X, Elems


    ### Extra split beads to model offset
    split_bool *= (nLayers_v > 1) # no split beads if single layer

    if bool(split_bool) :
        ## generate trajectory with extra split beads (as if split beads were regular beads)
        meshing_split = L, Hn, Hb, beadType, layerType, nLayers_h + 1, nLayers_v, nNodes
        #L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing_split
        X_split, Elems_split, U0_split = mesh.generate_trajectory(meshing_split, zigzag)
        #plot.plotpoints(X_split, True)

        ## apply offset (considering respective bead width for split beads)
        offset_new, offset_split, overhang_n = splitBeads.stacking_offset_split_beads(meshing, offset)
        X_split = splitBeads.mesh_offset_split(X_split, Elems_split, meshing_split, offset_split, zigzag)
        #X_split = splitBeads.mesh_offset_split(meshing_split, offset_new, zigzag)  # don't compensate for smaller split beads
        #plot.plotpoints(X_split, True)

        # making split weldDof relies on weldDof weights computed without splitbeads
        extended_weldDof = shapeOffset.make_extended_weldDof(meshing, offset, weldDof)
        extended_weldDof = splitBeads.extended_split_weldDof(meshing_split, extended_weldDof, overhang_n)

        X, Elems, U0 = X_split, Elems_split, U0_split
        meshing = meshing_split
        #offset = offset_split
        offset = offset_new
        weldDof_unsplit = weldDof.copy()
        weldDof = extended_weldDof[:,:4].astype(int)    # ie weldDof_split

    Xunc, uncElems = mesh.uncouple_nodes(X, Elems)

    ### Prevision taille
    nElemsTot = Elems.shape[0]
    nUncNodes = Xunc.shape[0]
    nNodesTot = X.shape[0]
    nParticules = 4
    nCoord = 3
    nNodeDOF = nCoord * nParticules
    nDOF = nNodeDOF * nUncNodes

    ### Useful matrices
    ## Integration matrices
    xiQ, wQ = fem.element_quadrature(quadOrder)
    XIQ, WQ = fem.fullmesh_quadrature(quadOrder, nElemsTot)
    nQP = WQ.shape[0]
    N, Dxi, Ds, J, W, O, qp2elem, elemQ = fem.integration_matrices(X, Elems, elemOrder, quadOrder)
    ## Projection matrices
    T, Tv, Tc, Ta = fem.alpha2beta_matrices(nNodesTot, nUncNodes)
    t, n, b, P = fem.local2global_matrices(Xunc, uncElems, elemOrder)

    ## Welding matrices
    Sni, Sn = weld.weldnode_matrices(Elems, uncElems)

    Sw0 = weld.weldwire_matrix(nNodesTot, weldDof)
    if split_bool:
    #    Sw0 = weld.weldwire_matrix(nNodesTot, weldDof_unsplit)
        Sw = splitBeads.offset_split_weldwire_matrix(meshing, extended_weldDof, Sw0)
    else :
    #    Sw0 = weld.weldwire_matrix(nNodesTot, weldDof)
        Sw = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw0)
    #plot.plotmatrix_sparse_t(Sw, nNodesTot, f'{nLayers_v} couches de {nLayers_h} cordons \n offset {int(offset[-1][-1] * 100)}% (Noeuds pairs, axe t)')
    Y0 = weld.bcs_matrix(U0, Sw0)
    Y = weld.bcs_matrix(U0, Sw)

    ## Assembly matrix node_alpha to qp_beta
    Assemble = Ta @ P.T @ Sn
    elem2node = fem.elem2node(nNodes, nLayers_h, nLayers_v)

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

    elapsed_time_data = time.time() - start_time_data
    print(f'La mise en donnée a pris {elapsed_time_data // 60}min et {elapsed_time_data % 60}s')

    start_time_temp = time.time()    # thermal loading
    ### Thermal eigenstrain    #TODO: Tg should be included in material
    Tbuild = 293.15 - dT
    Tsub = 323.15
    Tamb= 293.15
    Tg = 328 #Tbuild*0.9 #Tbuild means material is active as soon as it starts cooling down ie right away #328 #TODO: Tg must be smaller than 0.95*Tbuild otherwise activeFunction is wrong

    ## Donnees thermique         # temperature is always given first for unsplit setting, then eventually split
    N_unsplit, qp2elem_unsplit = itemgetter(0, 6)(fem.integration_matrices(X_unsplit, Elems_unsplit, elemOrder, quadOrder))
    nNodesTot_unsplit = X_unsplit.shape[0]
    nUncNodes_unsplit = Xunc_unsplit.shape[0]
    nElemsTot_unsplit = Elems_unsplit.shape[0]
    if path is not None :
        ## Time
        nLayers = nLayers_v * nLayers_h    # temperature is given for unsplit setting
        tau_final = 0 #int(np.ceil((nNodes-1)*3/4)) # 0 #nNodes-1 # 10 #
        nPas = (nNodes-1)*nLayers + tau_final
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
        nLayers = nLayers_v * nLayers_h#(-split_bool)
        tau_final = Nth
        nPas = (nNodes-1)*nLayers + tau_final
        ## Data : broadcast on iterative loading
        dTelem_tk = qp2elem_unsplit @ (dTmoy / np.sum(dTmoy) * quadOrder)     # thermal load profile from 'structure', ie at final time step tk=nTimeStep, size (nElemsTot,1)
        dTelem_tk *= dT # normalized with dTmoy / np.sum(dTmoy) so that all cooling add up to dT parameter
        dTelem = sp.sparse.diags(dTelem_tk[::-1], -np.arange(nElemsTot_unsplit)-1, shape=(nPas + 1, nElemsTot_unsplit), format='csr')  # dTelem_tk broadcast and stored on diagonals to account for successive time steps
        #print(dTelem[:,2].todense())     # thermalData[:, i] = temperature of the ith element throughout the simulation
        Telem = sp.sparse.tril(np.ones(((nPas+1, nPas+1)))) @ dTelem#.todense()   # sp.sparse.tril but +Tbuild requires dense format
        dTelem = dTelem.todense() # to use np.repeat in loop on dTelem[actualTime]
        Telem = Tbuild+Telem.todense() # to use np.repeat in loop on np.repeat(Telem[actualTime], 2)
    if bool(split_bool) : # make split beads experience exactly same temperature as primary beads, #TODO: NB.: assuming thinwall
       kron_duplicate = sp.sparse.kron(sp.sparse.eye(nLayers_v),sp.sparse.kron(np.ones((1,2)),sp.sparse.eye(nNodes-1))) # duplicate groups of nElems=nNodes-1 columns to account for split_beads undergoing the same cooling on both halves
       dTelem = dTelem @ kron_duplicate
       Telem = Telem @ kron_duplicate

    elapsed_time_temp = time.time() - start_time_temp
    print(f'La definition du chargement thermique a pris {elapsed_time_temp // 60}min et {elapsed_time_temp % 60}s')

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
    #Sigma_time = np.zeros((nPas, (6+9)*nQP, 1))

    ## Stiffness zero
    tol = 1e-9

    ## Node deposition time
    nodeDeposTime = np.arange(nNodesTot_unsplit) # 0.99 * np.arange(0,nNodesTot,1) #  -20 * np.ones(nNodes)  #
    if bool(split_bool):
        nodeDeposTime = nodeDeposTime @ sp.sparse.kron(sp.sparse.eye(nLayers_v),sp.sparse.kron(np.ones((1,2)),sp.sparse.eye(nNodes)))  # duplicate to enforce both split_bead nodes are deposited at the same time

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


    elapsed_time_data = time.time() - start_time_data
    print(f'La mise en donnée a pris {elapsed_time_data // 60}min et {elapsed_time_data % 60}s')



    # Start time before the computation
    start_time = time.time()
    print(f'computation has started, {nPas} computation remaining')
    ### Time loop
    Un_max = np.zeros((nPas,4)) # previous bead et absolu
    U_data, Eps_data, Sigma_data = [], [], []

    for actualTime in np.arange(nPas):
        ## Activated element ?
        Telem_instant = (N @ np.repeat(Telem[actualTime], 2).T).flatten() #  Telem_instant = (N @ Telem[actualTime][:,np.newaxis][:,[0,0]].flatten().T).T
        activeFunction = tol + (1 - tol) * 1 / (1 + np.exp(1 * (Telem_instant - Tg)))  # size (nQP,)
        # matrice W pondérée par activeFunction (Integral weight matrix including local jacobian)
        Wa = sp.sparse.diags((np.array(activeFunction)*(WQ * J[:, np.newaxis] ).T).flatten())
        #print(activeFunction)

        # dTelemmoy doit être vertical
        dTelemmoy = (N @ np.repeat(dTelem[actualTime], 2).T) # N @ dTelem[actualTime][:,np.newaxis][:,[0,0]].flatten()[:,np.newaxis]
        if len(dTelemmoy.shape) < 2 :  # for some reason shape differs for dataFiles or dTelem_tk
            dTelemmoy = dTelemmoy[:,np.newaxis]

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
        fbc = f #- K @ u00  #u00 is 0 anyways, unless imposed outside load
        yfbc = Y.T @ fbc  # Deleting non useful dof

        ## Solve
        vs = sp.sparse.linalg.spsolve(yKy, yfbc)
        v = Y @ vs[:, np.newaxis]
        vUncouple = Assemble @ v  # node beta configuration

        # vérifier les conditions cinématiques entre les deux premiers cordons superposés
        # print(v.reshape((3, 4, nNodesTot))[:,0,:nNodes] - v.reshape((3, 4, nNodesTot))[:,2,nNodes:2*nNodes])   # condition cinématique entre les deux premiers cordons du thinwall (particule 0 et 2 ie overhang pour un offset négatif)
        # print(v.reshape((3, 4, nNodesTot))[:,3,nNodes:] - (0.5*v.reshape((3, 4, nNodesTot))[:,1,:nNodes]+0.5*v.reshape((3, 4, nNodesTot))[:,0,:nNodes]))   # condition cinématique entre les deux premiers cordons du thinwall (particule 1 et 20-80% de 2 et 3 ie straddle pour un offset négatif de 20%)


        ## Updating displacement vector
        V = V + v
        U = U + v.reshape((3, 4, nNodesTot))
        if actualTime > nNodes and actualTime < (nNodes-1)*nLayers:
            U_nAVG = np.average(U[1], axis=0)
            Un_max[actualTime] = np.argmax(np.abs(U_nAVG)), np.max(np.abs(U_nAVG)), np.argmax(np.abs(U_nAVG)[actualTime-nNodes:actualTime]), np.max(np.abs(U_nAVG)[actualTime-nNodes:actualTime]),
        ## Updating generalized strains
        dEps = sp.sparse.vstack((Btot, Ctot)) @ vUncouple  # sp.sparse.vstack((Btot, Ctot)) @ Assemble @ v
        Eps += dEps

        ## Updating generalized stresses
        Rtot = sp.sparse.block_diag((sp.sparse.kron(Rxi, Wa), sp.sparse.kron(Rchi, Wa)))
        dSigma = Rtot @ dEps - sp.sparse.vstack((sp.sparse.kron(Rxi, Wa) @ Eps_thermal, sp.sparse.csr_matrix((9 * nQP, 1)) ))
        Sigma += dSigma
        #Sigma_time[actualTime] = Sigma

        ## Plot deformed shape and stress
        ## Update deformed shape
        scale = 1   #/(1+split_bool)   #0.5 * np.linalg.norm([Hn, Hb]) / np.max(abs(U), (0, 1, 2)) #0.5 * np.linalg.norm([Hn, Hb]) / np.max(abs(U), (0, 1, 2)) #
        uplot = np.moveaxis(U, (0, 1, 2), (1, 2, 0))
        y = x + scale * uplot

        if actualTime % nNodes == 0 :
            nBeads_printed = actualTime // nNodes
            U_data.append(U), Eps_data.append(Eps), Sigma_data.append(Sigma)

        y = np.moveaxis(y, (0, 1, 2), (0, 2, 1)).reshape(4 * y.shape[0], 3)
        srf.set_verts(y[tri])

        # QuadWire_Elastic version
        if clrmap == "stt" :
            data_plot = Sni.T @ Sigma[0:nQP]/(Hn*Hb)  #sp.sparse.linalg.lsqr(N @ Sni, Sigma[0:nQP])[0][:, None]
        elif clrmap == "stn" :
            data_plot = Sni.T @ Sigma[nQP*3:nQP*4]/(Hn*Hb)  #sp.sparse.linalg.lsqr(N @ Sni, Sigma[nQP*3:nQP*4])[0][:, None]
        elif clrmap == "stb" :
            data_plot = Sni.T @ Sigma[nQP*4:nQP*5]/(Hn*Hb)  #sp.sparse.linalg.lsqr(N @ Sni, Sigma[nQP*4:nQP*5])[0][:, None]
        elif clrmap == "temp" :
            data_plot = Sni.T @ (Telem[actualTime] @ qp2elem).T    # POURQUOI CA FONCTIONNE PAS
        clr = data_plot[:, None] * [[1, 1, 1, 1]]

        if toPlot :
            ## Plot stress
            isDeposed = actualTime/nElemsTot_unsplit*nNodesTot_unsplit > nodeDeposTime

            clr[np.invert(isDeposed)] = np.nan
            ## Mask NaN values in clr
            #clr = np.ma.masked_invalid(clr)

            # Set the color data and clipping limits to ignore NaN values
            srf.set_array(np.mean(clr.flatten()[tri], axis=1))
            srf.set_clim(np.nanmin(clr), np.nanmax(clr)) #srf.set_clim(clr.min(), clr.max())  # Masked min/max will ignore NaN values
            srf.set_cmap("viridis")

            plt.pause(0.02)

        #print('actualTime:', actualTime)
    # End time after the computation
    #print('Un_max :', np.max(Un_max[:,:2], axis=0), 'Un_max previous bead =', np.max(Un_max[:,2:], axis=0))
    end_time = time.time()
    print(f'computation time : {(end_time - start_time)//60} min and {(end_time - start_time)%60} seconds)')

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
    # Eps_chi = Eps[6*nQP:]
    # Sigma_s = Sigma[:6*nQP]
    # Sigma_m = Sigma[6*nQP:]

    ### Energy
    Eps_th = sp.sparse.vstack((Eps_thermal, sp.sparse.csr_matrix((9*nQP, 1)))).toarray()
    qpEnergyDensity = behavior.energyDensity(Eps, Eps_th, Sigma, Rtot, quadOrder, 15)
    elementEnergyDensity = qp2elem @ qpEnergyDensity
    #
    #
    # ### save data to file
    #
    # # Prepare data for CSV
    # data_row = {
    #     "nLayers_v": nLayers_v,
    #     "nNodes": nNodes,
    #     "Elapsed Time Data (min)": elapsed_time_data // 60,
    #     "Elapsed Time Data (sec)": elapsed_time_data % 60,
    #     "Elapsed Time Temp (min)": elapsed_time_temp // 60,
    #     "Elapsed Time Temp (sec)": elapsed_time_temp % 60,
    #     "nPas": nPas,
    #     "Computation Time (min)": (end_time - start_time) // 60,
    #     "Computation Time (sec)": (end_time - start_time) % 60
    # }
    #
    # # File path
    # csv_file = "figures\poc\computation_time.csv"
    #
    # # Write or append data to CSV
    # file_exists = os.path.exists(csv_file)
    # with open(csv_file, mode='a', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=data_row.keys())
    #     # Write header only if file does not exist
    #     if not file_exists:
    #         writer.writeheader()
    #     writer.writerow(data_row)
    #
    # print("Data has been written to", csv_file)

    return U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems, Un_max, U_data, Eps_data, Sigma_data, clr
