""" Module storing function derivatives with respect to offset and respective finite difference validation"""

# %% import packages
import time
from operator import itemgetter

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import qw_additive
from modules import mesh, fem, weld, behavior, plot
from shape import shapeOffset, shapeOptim_module


# %% Copy/Paste qw_structure with specific returns.

def fixed_structure(meshing, loading, discretization, material, additive=False, path=None):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    dT, loadType = loading
    elemOrder, quadOrder = discretization
    E, nu, alpha, optimizedBehavior = material


    ### Discretization and meshing
    X, Elems, U0 = mesh.generate_trajectory(meshing)
    weldDof = weld.welding_conditions(X, meshing)

    # if toPlot :
    #     plot.plotpoints(X)
    if elemOrder == 2:
        X, Elems, U0 = mesh.second_order_discretization(X, Elems, U0)

    Xunc, uncElems = mesh.uncouple_nodes(X, Elems)

    ### Prevision taille
    nElemsTot = Elems.shape[0]
    nUncNodes = Xunc.shape[0]
    nNodesTot = X.shape[0]
    nParticules = 4
    nCoord = 3
    nNodeDOF = nCoord * nParticules

    ### Useful matrices
    ## Integration matrices
    xiQ, wQ = fem.element_quadrature(quadOrder)
    XIQ, WQ = fem.fullmesh_quadrature(quadOrder, nElemsTot)
    nQP = WQ.shape[0]
    N, Dxi, Ds, J, W, O, qp2elem, elemQ = fem.integration_matrices(X, Elems, elemOrder, quadOrder)
    ## Projection matrices
    T, Tv, Tc, Ta = fem.alpha2beta_matrices(nNodesTot, nUncNodes)
    t, n, b, P = fem.local2global_matrices(Xunc, uncElems, elemOrder)

    ## Behavior
    # material parameters
    k0 = E / (3 * (1 - 2 * nu))  # bulk modulus (K)
    mu = E / (2 * (1 + nu))  # shear modulus (G)
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # k0-2/3*mu

    ## Local behavior
    Btot = behavior.derivation_xi(Hn, Hb, N, Ds)
    Ctot = behavior.derivation_chi(Hn, Hb, N, Ds)
    if optimizedBehavior:
        Rxi, Rchi = behavior.optimization_Rxi_Rchi()
    else:
        Rxi = behavior.homogeneization_Rxi(Hn, Hb, lmbda, mu)
        Rchi = behavior.homogenization_Rchi(L, Hn, Hb, lmbda, mu, nNodes)

    ## Local stiffness matrix
    Kxi = behavior.assembly_behavior(Rxi, Btot, W)
    Kchi = behavior.assembly_behavior(Rchi, Ctot, W)
    K = Kxi + Kchi

    if not additive :
        ### Thermal eigenstrain
        dTalpha = behavior.dTfcn(N @ Xunc, dT, loadType, nLayers_h, nLayers_v)
        dTbeta = dTalpha @ T
        dTmoy = dTbeta[:, 0][:, np.newaxis]

    else :
        ### Thermal eigenstrain    #TODO: Tg should be included in material
        Tbuild = 293.15 - dT
        Tsub = 323.15
        Tamb= 293.15
        Tg = 328 #Tbuild*0.9 #Tbuild means material is active as soon as it starts cooling down ie right away #328 #TODO: Tg must be smaller than 0.95*Tbuild otherwise activeFunction is wrong

        ## Donnees thermique         # temperature is always given first for unsplit setting, then eventually split
        if path is not None :
            ## Time
            nLayers = nLayers_v * nLayers_h    # temperature is given for unsplit setting
            tau_final = 0 #int(np.ceil((nNodes-1)*3/4)) # 0 #nNodes-1 # 10 #
            nPas = (nNodes-1)*nLayers + tau_final
            ## Data
            dataFiles = [path + '/temp_' + str(k) + ".txt" for k in range(1, nPas+1)]  # remove .zfill(4) if needed or apply generate_thermal.rename_files for compatible naming
            ## Appeler la fonction pour lire les fichiers
            from modules import thermaldata
            Telem = thermaldata.get_data2tamb(dataFiles, nNodes-1, nLayers_h, nLayers_v, Tbuild) # size (nTimeStep, nElemsTot)
            # Compute the dT_n = T_n - T_n-1
            dTelem = thermaldata.delta_elem_transition_vitreuse(Telem, Tbuild, Tg)
        else :
            ## Data : thermal load
            T = fem.alpha2beta_matrices(nNodesTot, nUncNodes)[0]
            dTalpha = behavior.dTfcn(N @ Xunc, dT, 'uniform', nLayers_h, nLayers_v)            #-split_bool
            dTbeta = dTalpha @ T
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
            dTelem_tk = qp2elem @ (dTmoy / np.sum(dTmoy) * quadOrder)     # thermal load profile from 'structure', ie at final time step tk=nTimeStep, size (nElemsTot,1)
            dTelem_tk *= dT # normalized with dTmoy / np.sum(dTmoy) so that all cooling add up to dT parameter
            dTelem = sp.sparse.diags(dTelem_tk[::-1], -np.arange(nElemsTot)-1, shape=(nPas + 1, nElemsTot), format='csr')  # dTelem_tk broadcast and stored on diagonals to account for successive time steps
            #print(dTelem[:,2].todense())     # thermalData[:, i] = temperature of the ith element throughout the simulation
            Telem = sp.sparse.tril(np.ones(((nPas+1, nPas+1)))) @ dTelem#.todense()   # sp.sparse.tril but +Tbuild requires dense format
            dTelem = dTelem.todense() # to use np.repeat in loop on dTelem[actualTime]
            Telem = Tbuild+Telem.todense() # to use np.repeat in loop on np.repeat(Telem[actualTime], 2)

        return X, Elems, uncElems, nNodesTot, weldDof, U0, Ta, P, K, Btot, Ctot, Rxi, Rchi, W, dTelem, Telem, Tg, nPas, nQP, N, WQ, J

    return X, Elems, uncElems, nNodesTot, weldDof, U0, Ta, P, K, Btot, Ctot, Rxi, Rchi, W, dTmoy

def shape_structure(initialization, meshing, offset, loading, discretization, material, additive):
    """
    Copy/Paste qw_structure with specific returns.
    """
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    dT, loadType = loading
    elemOrder, quadOrder = discretization
    E, nu, alpha, optimizedBehavior = material

    if not additive :
        X, Elems, uncElems, nNodesTot, weldDof, U0, Ta, P, K, Btot, _, Rxi, _, W, dTmoy= initialization
    else :
        X, Elems, uncElems, nNodesTot, weldDof, U0, Ta, P, K, Btot, Ctot, Rxi, Rchi, W, dTelem, Telem, Tg, nPas, nQP, N, WQ, J = initialization
    X = shapeOffset.mesh_offset(meshing, offset, X, Elems)

    ## Welding matrices
    Sni, Sn = weld.weldnode_matrices(Elems, uncElems)

    Sw0 = weld.weldwire_matrix(nNodesTot, weldDof)
    Sw = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw0)

    Y0 = weld.bcs_matrix(U0, Sw0)
    Y = weld.bcs_matrix(U0, Sw)
    # matrice de passage noeud alpha -> qp beta
    Assemble = Ta @ P.T @ Sn

    ### Assembly of K.u = f
    # hard copy is needed to keep the u0 vector unmodified
    u00 = U0.flatten()[:, np.newaxis].copy()
    freeDOF = np.isnan(U0.flatten())  # True/False table storing remaining DOF
    u00[freeDOF] = 0  # cull NaNs

    ## Construction of the global stiffness matrix
    # K = Ta.T @ K @ Ta  # Projection alpha -> beta puis beta -> alpha
    # K = P @ K @ P.T    # Projection global -> local puis local -> global
    # K = Sn.T @ K @ Sn  # Welding nodes
    K = Assemble.T @ K @ Assemble
    yKy = Y.T @ K @ Y  # Deleting non-useful dof

    if not additive :
        ## Construction of the force vector
        # Thermal eigenstrain
        Eps_thermal = behavior.thermal_eigenstrain(alpha, dTmoy)
        f_thermal = Btot.T @ sp.sparse.kron(Rxi, W) @ Eps_thermal    # size nUncNodes * nNodeDOF
        # Assembly of the force vector
        f = f_thermal
        f = Assemble.T @ f

        fbc = f - K @ u00
        yfbc = Y.T @ fbc  # Deleting non-useful dof

        ### Solve K.u = f
        us = sp.sparse.linalg.spsolve(yKy, yfbc)

        ### Reconstruction of the problem's unknowns
        ## Displacement field
        u = Y @ us[:, np.newaxis]

        U = u.reshape((3, 4, nNodesTot))

        return U, us, yKy, Assemble, Y, yfbc, X, Elems
    if additive :
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

        # Start time before the computation
        start_time = time.time()
        print(f'computation has started, {nPas} computation remaining')
        ### Time loop
        for actualTime in np.arange(nPas):
            ## Activated element ?
            Telem_instant = (N @ np.repeat(Telem[actualTime], 2).T).flatten() #  Telem_instant = (N @ Telem[actualTime][:,np.newaxis][:,[0,0]].flatten().T).T
            activeFunction = tol + (1 - tol) * 1 / (1 + np.exp(1 * (Telem_instant - Tg)))  # size (nQP,)
            # matrice W pondérée par activeFunction (Integral weight matrix including local jacobian)
            Wa = sp.sparse.diags((np.array(activeFunction)*(WQ * J[:, np.newaxis] ).T).flatten())

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

            ## Updating displacement vector
            V = V + v
            from shape.shapeOptim_module import u2us
            us = u2us(V, Y)
            U = U + v.reshape((3, 4, nNodesTot))
            ## Updating generalized strains
            dEps = sp.sparse.vstack((Btot, Ctot)) @ vUncouple  # sp.sparse.vstack((Btot, Ctot)) @ Assemble @ v
            Eps += dEps

            ## Updating generalized stresses
            Rtot = sp.sparse.block_diag((sp.sparse.kron(Rxi, Wa), sp.sparse.kron(Rchi, Wa)))
            dSigma = Rtot @ dEps - sp.sparse.vstack((sp.sparse.kron(Rxi, Wa) @ Eps_thermal, sp.sparse.csr_matrix((9 * nQP, 1)) ))
            Sigma += dSigma
        end_time = time.time()
        print(f'computation time : {(end_time - start_time)//60} min and {(end_time - start_time)%60} seconds)')

        return U, us, yKy, Assemble, Y, yfbc, X, Elems


# %% Offset increase for finite differences

def offset_epsilon(meshing, offset, epsilon, i):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nNodes_h = nNodes * nLayers_h
    if i > nLayers_v - 1:
        print('Error: i is greater than number of vertical layers')
    offset_i = 0 * offset
    offset_i[:, i * nNodes_h:(i + 1) * nNodes_h] = 1
    offset_epsilon_i = offset + offset_i * epsilon
    return offset_epsilon_i

def param_epsilon(param, epsilon, i):
    param_i = 0 * param
    param_i[i] = 1
    param_epsilon_i = param + param_i * epsilon
    return param_epsilon_i


# %% weld

def finDiff_Sw(meshing, offset, i, epsilon):
    offset_eps = offset_epsilon(meshing, offset, epsilon, i)

    Sw_offset = weld.offset2weld(meshing, offset)
    Sw_eps = weld.offset2weld(meshing, offset_eps)

    return (Sw_eps - Sw_offset) / epsilon
    
def finDiff_Sw(meshing, offset, weldDof, Sw, i, epsilon):
    offset_eps = offset_epsilon(meshing, offset, epsilon, i)

    Sw_offset = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw)
    Sw_eps = shapeOffset.offset_weldwire_matrix(meshing, offset_eps, weldDof, Sw)

    return (Sw_eps - Sw_offset) / epsilon

def derive_Sw(meshing, offset, weldDof, Sw, i):  # derivative of weight coefficients in Sw with respect to delta_i the offset of layer i
    '''Derive weld.weldwire_matrix'''
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nNodesTot = nNodes * nLayers_h * nLayers_v

    v_connections = shapeOffset.vertical_connections_n(meshing, weldDof)
    v_mask = shapeOffset.connectionType(weldDof)

    def derive_welding_weights(meshing, offset, i):  # derivative of weight coefficients with respect to delta_i the offset of layer i
        '''
        Returns derivative of weights with respect to delta_i the offset of layer i

        Returns
        -------
        weights_derivative : shape (2, nNodesTot, 3)
            derivative of weight coefficients to weld particles to neighbour beads due to offset
            weights[0] are coefficients to t axis and weights[1] are coefficients to n axis
        '''
        L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
        nNodes_h = nNodes * nLayers_h
        nNodesTot = nNodes_h * nLayers_v

        first_node_i = nNodes_h * i                # first node of the i-th layer
        last_node_i = nNodes_h * (i + 1) - 1       # last node of the i-th layer
        first_node_iplus1 = nNodes_h * (i + 1)     # first node of the (i+1)-th layer
        last_node_iplus1 = nNodes_h * (i + 2) - 1  # last node of the (i+1)-th layer

        overhang = offset.copy()  # delta_i
        overhang[:, nNodes_h:] -= offset[:, :-nNodes_h]  # - delta_{i-1}

        ####
        # old version
        mask_delta = (overhang > 0).astype(float)
        mask_delta[overhang == 0] = np.nan
        weights_derivative = np.stack((mask_delta * -1, mask_delta * 0, mask_delta * 1), axis=2) + np.stack(((1 - mask_delta) * 1, (1 - mask_delta) * -1, (1 - mask_delta) * 0), axis=2)

        weights_derivative[:, first_node_iplus1:last_node_iplus1+1] *= -1  # layer i+1 has weights dependant on -delta_i
        weights_derivative[:, :first_node_i] *= 0  # previous layers < i have weights independent of delta_i
        weights_derivative[:, last_node_iplus1+1:] *= 0  # following layers > i have weights independent of delta_i

        # new version

        d_weights_pos_2 = np.ones((2, nNodesTot, 3))
        d_weights_neg_2 = np.ones((2, nNodesTot, 3))
        d_weights_pos_3 = np.ones((2, nNodesTot, 3))
        d_weights_neg_3 = np.ones((2, nNodesTot, 3))

        # data layer i
        data_i = overhang[:, first_node_i:last_node_i+1]
        deltaHn_i, deltaHn2_i, deltaHn3_i = data_i / Hn, np.power(data_i / Hn, 2), np.power(data_i / Hn, 3)
        # data layer i+1
        data_i1 = overhang[:, first_node_iplus1:last_node_iplus1+1]
        deltaHn_i1, deltaHn2_i1, deltaHn3_i1 = data_i1 / Hn, np.power(data_i1 / Hn, 2), np.power(data_i1 / Hn, 3)

        di_deltaHn, di_deltaHn2, di_deltaHn3 = 1 / Hn, 2 * 1 / Hn * data_i / Hn, 3 * 1 / Hn * np.power(data_i / Hn, 2)
        di1_deltaHn, di1_deltaHn2, di1_deltaHn3 = - 1 / Hn, - 2 * 1 / Hn * data_i1 / Hn, - 3 * 1 / Hn * np.power(data_i1 / Hn, 2)     # layer i+1 has weights dependant on -delta_i

        di_weights_pos_2 = np.stack((di_deltaHn - 4 * di_deltaHn2 + 2 * di_deltaHn3, np.multiply(- di_deltaHn, deltaHn2_i) + np.multiply(2 - deltaHn_i, di_deltaHn2), -np.multiply(di_deltaHn, np.power(1 - deltaHn_i, 2))) - np.multiply(deltaHn_i, 2 * (- di_deltaHn) * (1 - deltaHn_i)), axis=2)
        di_weights_neg_2 = np.stack((di_deltaHn - di_deltaHn2 - di_deltaHn3, - di_deltaHn + 2 * di_deltaHn2 + 2 * di_deltaHn3, - (di_deltaHn2 + di_deltaHn3)), axis=2)

        di_weights_pos_3 = np.stack((np.multiply(di_deltaHn, np.power(1 - deltaHn_i, 2)) + np.multiply(1 + deltaHn_i, 2 * (- di_deltaHn) * (1 - deltaHn_i)), np.multiply(di_deltaHn, 1 + 2 * deltaHn_i - 2 * deltaHn2_i) + np.multiply(deltaHn_i, 2 * di_deltaHn - 2 * di_deltaHn2), -np.multiply(- di_deltaHn, deltaHn2_i) - np.multiply(1 - deltaHn_i, di_deltaHn2)) , axis=2)
        di_weights_neg_3 = np.stack((- di_deltaHn - 4 * di_deltaHn2 - 2 * di_deltaHn3, di_deltaHn + 2 * di_deltaHn2 + di_deltaHn3, 2 * di_deltaHn2 + di_deltaHn3), axis=2)

        di1_weights_pos_2 = np.stack((di1_deltaHn - 4 * di1_deltaHn2 + 2 * di1_deltaHn3, np.multiply(- di1_deltaHn, deltaHn2_i1) + np.multiply(2 - deltaHn_i1, di1_deltaHn2), -np.multiply(di1_deltaHn, np.power(1 - deltaHn_i1, 2))) -np.multiply(deltaHn_i1, 2 * (- di1_deltaHn) * (1 - deltaHn_i1)), axis=2)
        di1_weights_neg_2 = np.stack((di1_deltaHn - di1_deltaHn2 - di1_deltaHn3, - di1_deltaHn + 2 * di1_deltaHn2 + 2 * di1_deltaHn3, - (di1_deltaHn2 + di1_deltaHn3)), axis=2)

        di1_weights_pos_3 = np.stack((np.multiply(di1_deltaHn, np.power(1 - deltaHn_i1, 2)) + np.multiply(1 + deltaHn_i1, 2 * (- di1_deltaHn) * (1 - deltaHn_i1)), np.multiply(di1_deltaHn, 1 + 2 * deltaHn_i1 - 2 * deltaHn2_i1) + np.multiply(deltaHn_i1, 2 * di1_deltaHn - 2 * di1_deltaHn2), -np.multiply(- di1_deltaHn, deltaHn2_i1) - np.multiply(1 - deltaHn_i1, di1_deltaHn2)) , axis=2)
        di1_weights_neg_3 = np.stack((- di1_deltaHn - 4 * di1_deltaHn2 - 2 * di1_deltaHn3, di1_deltaHn + 2 * di1_deltaHn2 + di1_deltaHn3, 2 * di1_deltaHn2 + di1_deltaHn3), axis=2)

        # layer i
        d_weights_pos_2[:, first_node_i:last_node_i+1] = di_weights_pos_2
        d_weights_neg_2[:, first_node_i:last_node_i+1] = di_weights_neg_2
        d_weights_pos_3[:, first_node_i:last_node_i+1] = di_weights_pos_3
        d_weights_neg_3[:, first_node_i:last_node_i+1] = di_weights_neg_3
        # layer i + 1
        d_weights_pos_2[:, first_node_iplus1:last_node_iplus1+1] = di1_weights_pos_2
        d_weights_neg_2[:, first_node_iplus1:last_node_iplus1+1] = di1_weights_neg_2
        d_weights_pos_3[:, first_node_iplus1:last_node_iplus1+1] = di1_weights_pos_3
        d_weights_neg_3[:, first_node_iplus1:last_node_iplus1+1] = di1_weights_neg_3


        bool_pos = np.repeat(((overhang > 0) * 1)[:,:,np.newaxis], 3, axis=2)
        bool_neg = np.repeat(((overhang < 0) * 1)[:,:,np.newaxis], 3, axis=2)
        d_weights_2 = np.multiply(d_weights_pos_2, bool_pos) + np.multiply(d_weights_neg_2, bool_neg)
        d_weights_3 = np.multiply(d_weights_pos_3, bool_pos) + np.multiply(d_weights_neg_3, bool_neg)

        d_weights_2[:, :first_node_i] *= 0  # previous layers < i have weights independent of delta_i
        d_weights_3[:, :first_node_i] *= 0  # previous layers < i have weights independent of delta_i

        d_weights_2[:, last_node_iplus1+1:] *= 0  # following layers > i have weights independent of delta_i
        d_weights_3[:, last_node_iplus1+1:] *= 0  # following layers > i have weights independent of delta_i
        #####

        return d_weights_2, d_weights_3, overhang

    offset_weights_derivative = derive_welding_weights(meshing, offset, i)
    offset_weights_derivative0 = derive_welding_weights(meshing, offset*0, i)

    extended_weldDof_weights_derivative0 = shapeOffset.make_welding_connections_n(meshing, offset_weights_derivative0, weldDof, v_connections, v_mask)
    extended_v_mask = shapeOffset.welding_type(extended_weldDof_weights_derivative0)[:, -1].astype(bool)  # adds column with 0 for horizontal connection and 1 for vertical connection
    extended_weldDof_weights_derivative = shapeOffset.make_welding_connections_n(meshing, offset_weights_derivative, weldDof, v_connections, v_mask)  # derivative of weigths with respect to delta_i
    primary_weldDof_weights_derivative = shapeOffset.make_primary_leaders(extended_weldDof_weights_derivative, Sw)

    # vertical connections are updated with weight derivatives, horizontal connections are set to 0 weights because not dependent on delta_i (0*Sw)
    Sw_derivative = shapeOffset.update_weldwire_matrix_weights(nNodesTot, primary_weldDof_weights_derivative[extended_v_mask], 0 * Sw)

    Sw_derivative = finDiff_Sw(meshing, offset, weldDof, Sw, i, 1e-7)     #TODO: fix derive_welding_weights (hint: depends on coefficients whether optimal weights or not)
    return Sw_derivative


# check derivation with finite difference
# X, Elems, U0 = mesh.generate_trajectory(meshing)
# weldDof = weld.welding_conditions(X, meshing)
# Sw = weld.weldwire_matrix(nNodesTot, weldDof)  # welding matrix without offset
# Sw = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw)
# for i in range(nLayers_v) :
#     Sw_derivative = derive_Sw(meshing, offset, weldDof, Sw, i)
#     Sw_finDiff = finDiff_Sw(meshing, offset, weldDof, Sw, i, epsilon)
#     print(np.sqrt(np.sum((Sw_derivative - Sw_finDiff).power(2))))
#     #plot.plotmatrix_sparse_t(Sw_derivative, nNodesTot)
#     #plot.plotmatrix_sparse_t(Sw_finDiff, nNodesTot)
#     plot.plotmatrix_sparse_t((Sw_derivative - Sw_finDiff), nNodesTot, f'i={i}')

def derive_Y(U0, Sw, Sw_derivative):
    '''Derive weld.bcs_matrix'''
    # Reshape U0 (3, 4, nNodes*nLayers) matrix as a vector
    u0 = U0.flatten()
    # True/False table storing remaining DOF
    freeDOF = np.isnan(u0)
    # Boundary conditions matrix
    B = sp.sparse.diags(freeDOF * 1)  # B matrix is made of Bt Bn Bb, Bi is made of B0i B1i B2i B3i (particles 0 1 2 3), B0i is made of B0i^0 B0i^1 B0i^2... ie B0i^k for k in range(nNodesTot) in order of mesh appearance
    # Delete DOFs
    Y = Sw @ B
    isZero = np.sum(Y, 0) == 0
    Y = Y[:, np.array(np.invert(isZero)).flatten()]

    Y_derivative = Sw_derivative @ B
    Y_derivative = Y_derivative[:, np.array(np.invert(isZero)).flatten()]
    return Y_derivative

def finDiff_Y(U0, Sw_offset, Sw_epsilon, epsilon):
    Y_offset = weld.bcs_matrix(U0, Sw_offset)
    Y_epsilon = weld.bcs_matrix(U0, Sw_epsilon)
    Y_diff = (Y_epsilon - Y_offset) / epsilon

    return Y_diff

def finDiff_Y(meshing, offset, i, epsilon):
    offset_eps = offset_epsilon(meshing, offset, epsilon, i)

    y_offset = weld.offset2bcs(meshing, offset)
    y_eps = weld.offset2bcs(meshing, offset_eps)

    return (y_eps - y_offset) / epsilon

# # check derivation with finite difference
# X, Elems, U0 = mesh.generate_trajectory(meshing)
# weldDof = weld.welding_conditions(X, meshing)
# Sw = weld.weldwire_matrix(nNodesTot, weldDof)
#
# for i in range(nLayers):
#     offset_eps = offset_epsilon(meshing, offset, epsilon, i)
#     Sw_derivative = derive_Sw(meshing, offset, weldDof, Sw, i)
#     Sw_offset = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw)
#     Sw_eps = shapeOffset.offset_weldwire_matrix(meshing, offset_eps, weldDof, Sw)
#
#     Y_derivative = derive_Y(U0, Sw, Sw_derivative)
#     Y_finDiff = finDiff_Y(U0, Sw_offset, Sw_eps, epsilon)
#     print(np.sum((Y_derivative - Y_finDiff)))
#     # plot.plotmatrix_sparse_t_evenNodes(Y_derivative, nNodesTot)
#     # plot.plotmatrix_sparse_t_evenNodes(Y_finDiff, nNodesTot)
#     # plot.plotmatrix_sparse_t_evenNodes((Y_derivative - Y_finDiff), nNodesTot, f'i={i}')



# %% shapeOptim

def derive_yKy(meshing, offset, loading, discretization, material, i):  # dérivée de yKy par rapport à delta_i le offset de la couche i
    '''
    copy of structure function except for _derivative terms (Sw_derivative inducing Y_derivative and yKy_derivative)
    '''
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    dT, loadType = loading
    elemOrder, quadOrder = discretization
    E, nu, alpha, optimizedBehavior = material

    ### Discretization and meshing
    X, Elems, U0 = mesh.generate_trajectory(meshing)
    weldDof = weld.welding_conditions(X, meshing)
    X = shapeOffset.mesh_offset(meshing, offset, X, Elems)
    if elemOrder == 2:
        X, Elems, U0 = mesh.second_order_discretization(X, Elems, U0)
    Xunc, uncElems = mesh.uncouple_nodes(X, Elems)

    ### Prevision taille
    #nElemsTOT = Elems.shape[0]
    nUncNodes = Xunc.shape[0]
    nNodesTOT = X.shape[0]
    #nParticules = 4
    #nCoord = 3
    #nNodeDOF = nCoord * nParticules

    ### Useful matrices
    ## Integration matrices
    #xiQ, wQ = fem.element_quadrature(quadOrder)
    #XIQ, WQ = fem.fullmesh_quadrature(quadOrder, nElemsTOT)
    #nQP = WQ.shape[0]
    N, Dxi, Ds, J, W, O, qp2elem, elemQ = fem.integration_matrices(X, Elems, elemOrder, quadOrder)
    ## Projection matrices
    T, Tv, Tc, Ta = fem.alpha2beta_matrices(nNodesTOT, nUncNodes)
    t, n, b, P = fem.local2global_matrices(Xunc, uncElems, elemOrder)
    ## Welding matrices
    Sni, Sn = weld.weldnode_matrices(Elems, uncElems)
    Sw0 = weld.weldwire_matrix(nNodesTOT, weldDof)
    Sw = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw0)
    #Y0 = weld.bcs_matrix(U0, Sw0)
    Y = weld.bcs_matrix(U0, Sw)

    Sw_derivative = derive_Sw(meshing, offset, weldDof, Sw, i)
    Y_derivative = derive_Y(U0, Sw, Sw_derivative)

    # matrice de passage noeud alpha -> qp beta
    Assemble = Ta @ P.T @ Sn

    ### Behavior
    ## PLA material parameters
    mu = E / (2 * (1 + nu))  # shear modulus (G)
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # k0-2/3*mu

    ## Local behavior
    Btot = behavior.derivation_xi(Hn, Hb, N, Ds)
    Ctot = behavior.derivation_chi(Hn, Hb, N, Ds)

    if optimizedBehavior:
        Rxi, Rchi = behavior.optimization_Rxi_Rchi()
    else:
        Rxi = behavior.homogeneization_Rxi(Hn, Hb, lmbda, mu)
        Rchi = behavior.homogenization_Rchi(L, Hn, Hb, lmbda, mu, nNodes)

    ## Local stiffness matrix
    Kxi = behavior.assembly_behavior(Rxi, Btot, W)
    Kchi = behavior.assembly_behavior(Rchi, Ctot, W)
    K = Kxi + Kchi

    # ### Thermal eigenstrain
    # dTalpha = behavior.dTfcn(N @ Xunc, dT, loadType, nLayers_h, nLayers_v)
    # dTbeta = dTalpha @ T
    # dTmoy = dTbeta[:, 0][:, np.newaxis]
    #
    # ### Assembly of K.u = f
    # # hard copy is needed to keep the u0 vector unmodified
    # u00 = U0.flatten()[:, np.newaxis].copy()
    # freeDOF = np.isnan(U0.flatten())  # True/False table storing remaining DOF
    # u00[freeDOF] = 0  # cull NaNs

    ## Construction of the global stiffness matrix
    # K = Ta.T @ K @ Ta # Projection alpha -> beta puis beta -> alpha
    # K = P @ K @ P.T    # Projection global -> local puis local -> global
    # K = Sn.T @ K @ Sn  # Welding nodes
    K = Assemble.T @ K @ Assemble
    #yKy = Y.T @ K @ Y  # Deleting non-useful dof
    yKy_derivative = Y_derivative.T @ K @ Y + Y.T @ K @ Y_derivative

    return yKy_derivative


def finDiff_yKy(meshing, offset, loading, discretization, material, epsilon, i):
    offset_epsilon_i = offset_epsilon(meshing, offset, epsilon, i)
    yKy = shape_structure(meshing, offset, loading, discretization, material)[2]
    yKy_epsilon_i = shape_structure(meshing, offset_epsilon_i, loading, discretization, material)[2]
    yKy_i = (yKy_epsilon_i - yKy) / epsilon
    return yKy_i

# # check derivation with finite difference
# for i in range(nLayers_v) :
#     yKy_derivative = derive_yKy(meshing, offset, loading, discretization, material, i)
#     yKy_finDiff = finDiff_yKy(meshing, offset, loading, discretization, material, epsilon, i)
#     print(np.sqrt(np.sum(((yKy_derivative - yKy_finDiff)/np.sqrt(np.sum(yKy_derivative.power(2)))).power(2))))
#     #plot.plotmatrix_sparse_t_evenNodes(yKy_derivative, nNodesTot, f'i={i} yKy dérivée exacte')
#     #plot.plotmatrix_sparse_t_evenNodes(yKy_finDiff, nNodesTot, f'i={i} yKy différence finie epsilon={epsilon}')
#
#     #plot.plotmatrix_sparse_t_evenNodes((yKy_derivative - yKy_finDiff), nNodesTot, f'i={i} yKy dérivée - yKy diff finie')
#     #plot.plotmatrix_sparse_t_evenNodes((yKy_derivative - yKy_finDiff)/np.sqrt(np.sum(yKy_derivative.power(2))), nNodesTot, f'i={i} (yKy dérivée - yKy diff finie)/||yKy dérivée||_2')

def derive_yfbc(meshing, offset, loading, discretization, material, i):
    '''
    ## copy of structure function except for _derivative terms (Sw_derivative inducing Y_derivative and yKy_derivative)
    '''
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    dT, loadType = loading
    elemOrder, quadOrder = discretization
    E, nu, alpha, optimizedBehavior = material

    ### Discretization and meshing
    X, Elems, U0 = mesh.generate_trajectory(meshing)
    weldDof = weld.welding_conditions(X, meshing)
    X = shapeOffset.mesh_offset(meshing, offset, X, Elems)
    if elemOrder == 2:
        X, Elems, U0 = mesh.second_order_discretization(X, Elems, U0)
    Xunc, uncElems = mesh.uncouple_nodes(X, Elems)

    ### Prevision taille
    #nElemsTOT = Elems.shape[0]  # Only used in unused XIQ, WQ calculation
    nNodesTOT = X.shape[0]    # nb beads = nNodesTOT//nNodes
    nUncNodes = Xunc.shape[0] #(nNodesTOT//nNodes)*(nNodes-1)*2
    #nParticules = 4
    #nCoord = 3
    #nNodeDOF = nCoord * nParticules

    ### Useful matrices
    ## Integration matrices
    #xiQ, wQ = fem.element_quadrature(quadOrder)
    #XIQ, WQ = fem.fullmesh_quadrature(quadOrder, nElemsTOT)  # Not used in subsequent calculations
    #nQP = WQ.shape[0]  # Not used
    N, Dxi, Ds, J, W, O, qp2elem, elemQ = fem.integration_matrices(X, Elems, elemOrder, quadOrder)
    ## Projection matrices
    T, Tv, Tc, Ta = fem.alpha2beta_matrices(nNodesTOT, nUncNodes)
    t, n, b, P = fem.local2global_matrices(Xunc, uncElems, elemOrder)
    ## Welding matrices
    Sni, Sn = weld.weldnode_matrices(Elems, uncElems)
    Sw0 = weld.weldwire_matrix(nNodesTOT, weldDof)
    Sw = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw0)
    #Y0 = weld.bcs_matrix(U0, Sw0)  # Only used for reference
    Y = weld.bcs_matrix(U0, Sw)

    Sw_derivative = derive_Sw(meshing, offset, weldDof, Sw, i)
    Y_derivative = derive_Y(U0, Sw, Sw_derivative)

    # matrice de passage noeud alpha -> qp beta
    Assemble = Ta @ P.T @ Sn

    ### Behavior
    ## PLA material parameters
    mu = E / (2 * (1 + nu))  # shear modulus (G)
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # k0-2/3*mu

    ## Local behavior
    Btot = behavior.derivation_xi(Hn, Hb, N, Ds)
    Ctot = behavior.derivation_chi(Hn, Hb, N, Ds)

    if optimizedBehavior:
        Rxi, Rchi = behavior.optimization_Rxi_Rchi()
    else:
        Rxi = behavior.homogeneization_Rxi(Hn, Hb, lmbda, mu)
        Rchi = behavior.homogenization_Rchi(L, Hn, Hb, lmbda, mu, nNodes)

    ## Local stiffness matrix
    Kxi = behavior.assembly_behavior(Rxi, Btot, W)
    Kchi = behavior.assembly_behavior(Rchi, Ctot, W)
    K = Kxi + Kchi

    ### Thermal eigenstrain
    dTalpha = behavior.dTfcn(N @ Xunc, dT, loadType, nLayers_h, nLayers_v)
    dTbeta = dTalpha @ T
    dTmoy = dTbeta[:, 0][:, np.newaxis]

    ### Assembly of K.u = f
    # # hard copy is needed to keep the u0 vector unmodified
    # u00 = U0.flatten()[:, np.newaxis].copy()
    # freeDOF = np.isnan(U0.flatten())  # True/False table storing remaining DOF
    # u00[freeDOF] = 0  # cull NaNs

    ## Construction of the global stiffness matrix
    #K = Ta.T @ K @ Ta # Projection alpha -> beta puis beta -> alpha  # Commented instruction
    #K = P @ K @ P.T    # Projection global -> local puis local -> global  # Commented instruction
    #K = Sn.T @ K @ Sn  # Welding nodes  # Commented instruction
    #K = Assemble.T @ K @ Assemble
    #yKy = Y.T @ K @ Y  # Deleting non-useful dof  # Not used in final result
    #yKy_derivative = Y_derivative.T @ K @ Y + Y.T @ K @ Y_derivative  # Only used for reference

    ## Construction of the force vector
    # Thermal eigenstrain
    Eps_thermal = behavior.thermal_eigenstrain(alpha, dTmoy)
    f_thermal = Btot.T @ sp.sparse.kron(Rxi, W) @ Eps_thermal    # shape (nUncNodes * 12,1)
    # Assembly of the force vector
    f = f_thermal
    f = Assemble.T @ f

    fbc = f #- K @ u00    #maintain sparse structure and u00 is zero anyway
    #yfbc = Y.T @ fbc  # Deleting non-useful dof  # Not used in final result
    yfbc_derivative = Y_derivative.T @ fbc

    return yfbc_derivative


def finDiff_yfbc(meshing, offset, loading, discretization, material, epsilon, i):

    offset_epsilon_i = offset_epsilon(meshing, offset, epsilon, i)
    yfbc = shape_structure(meshing, offset, loading, discretization, material)[5]
    yfbc_epsilon_i = shape_structure(meshing, offset_epsilon_i, loading, discretization, material)[5]
    yfbc_i = (yfbc_epsilon_i - yfbc) / epsilon
    return yfbc_i

# # check derivation with finite difference
# X, Elems, U0 = mesh.generate_trajectory(meshing)
# weldDof = weld.welding_conditions(X, meshing)
# Sw = weld.weldwire_matrix(nNodesTot, weldDof)  # welding matrix without offset
# offset = shapeOffset.stacking_offset(meshing, "linear", 0, -0.5)  # offset between successive layers along t and n (2, nLayers_v*nNodes_h)
# Sw = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw)
# for i in range(nLayers):
#     offset_eps = offset_epsilon(meshing, offset, epsilon, i)
#     Sw_derivative = derive_Sw(meshing, offset, weldDof, Sw, i)
#     Sw_offset = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw)
#     Sw_eps = shapeOffset.offset_weldwire_matrix(meshing, offset_eps, weldDof, Sw)
#
#     yfbc_derivative = derive_yfbc(meshing, offset, loading, discretization, material, i)
#     yfbc_finDiff = finDiff_yfbc(meshing, offset, loading, discretization, material, epsilon, i)
#     print(yfbc_derivative - yfbc_finDiff)


def finDiff_offset2uslike(meshing, offset, loading, discretization, material, epsilon, i):
    U, us, yKy, Assemble, Y, yfbc, X, Elems = shape_structure(meshing, offset, loading, discretization, material)
    offset_epsilon_i = offset_epsilon(meshing, offset, epsilon, i)
    U_epsilon_i, us_epsilon_i, yKy_epsilon_i, Assemble_epsilon_i, Y_epsilon_i, yfbc_epsilon_i, X_epsilon_i, Elems_epsilon_i = shape_structure(meshing, offset, loading, discretization, material)

    offset2uslike = shapeOptim_module.offset2uslike(meshing, offset, Y, Assemble)
    offset2uslike_epsilon_i = shapeOptim_module.offset2uslike(meshing, offset_epsilon_i, Y_epsilon_i, Assemble_epsilon_i)
    offset2uslike_i = (offset2uslike_epsilon_i - offset2uslike) / epsilon
    return offset2uslike_i
