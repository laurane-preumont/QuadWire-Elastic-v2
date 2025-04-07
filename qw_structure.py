""" 
QuadWire function for calculating printed structures 
"""
# %% Imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from modules import mesh, fem, weld, behavior, plot
from shape import shapeOffset, splitBeads


# %% Function

def structure(meshing, offset, loading, discretization, material, toPlot, clrmap="stt", scfplot=10, split_bool=False):
    """
    Implementation of the mechanical calculation core of the QuadWire model for a pre-printed structure.
    Wires are ordered  :   TL -- TR
                           |     |
                           BL -- BR
    Particules in the article are enumerated such as [TL,TR,BL,BR] = [3,1,4,2] 
    Primary variables: U_alpha = [[uTL_t,uTR_t,uBL_t,uBR_t],[uTL_n,uTR_n,uBL_n,uBR_n],[uTL_b,uTR_b,uBL_b,uBR_b]]
    
    Parameters
    ----------
    L : int
         Length of the structure in tangential direction t.
    Hn : int
        Characteristic length of the bead section in n direction.
    Hb : int
        Characteristic length of the bead section in b direction.
    beadType : str
        Geometry of the first bead. Options available are linear, circular, square, quarter_circular, quarter_square, sinus
    layerType : str
        Geometry of the first layer. Options available are "duplicate" or "normal".
        Default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
    offset : array
        Offset of layer compared to previous one in t and n direction. Size (nLayers_v, 2)
    nNodes : int
        Number of nodes discretising the length L.
    nLayers_h : int
        Number of layers of the structure in the horizontal direction.
    nLayers_v : int
        Number of layers of the structure in the build direction.
    dT : float
       Temperature variation
    loadType : str
        Type of thermal loading.
        Options are "uniform", "linear", "quad" or "random"
    elemOrder : int
         Element order : number of nodes that forms an element 
         Options are 1 for two-nodes P1 element and 2 for three-nodes P2 element
    quadOrder : int
        Element quadrature order.
    optimizedBehavior : bool
        If True, the material behavior used is obtained through an optimisation procedure as described in the article.
        If False, the material behavior used is obtained through an homogeneization procedure.
    toPlot : bool
        If True, several outputs are ploted.
        If False, no outputs are ploted
    clrmap : str, optional
        Option for the plot function.
        Choice of the quantity displayed in the colormap of the printed structure.
        The default is "stt". Options are "stt" for sigma_tt and "nrg" for the energy
    scfplot : int, optional
        Aspect ratio between the length and height (or width) of the section. 
        Used for display functions. 
        The default is 10.

    Returns
    -------
    U : array of float of size (nCoord, nParticules, nNodesTot) 
        Nodal displacement result.
        Coordinates are ordered [t, n, b]
        Particules are ordered [TL,TR,BL,BR] = [3,1,4,2] 
        Nodes are ordered by layers.
    Eps : array of size (nInc*nQP, 1)
        QW generalized strains.
        k*nQP to (k+1)*nQP gives the k^th generalized strain at each quadrature point
    Sigma : array of size (nInc*nQP, 1)
        QW generalized stresses.
        k*nQP to (k+1)*nQP gives the k^th generalized stress at each quadrature point
    nrg : array of size (nElemsTot,)
        Linear elastic energy density of each elements.
        Elements are ordered by layer.

    """
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    dT, loadType = loading
    elemOrder, quadOrder = discretization
    E, nu, alpha, optimizedBehavior = material

    ### Discretization and meshing
    X, Elems, U0 = mesh.mesh_first_bead(L, nNodes, beadType, meshType)
    X, Elems, U0 = mesh.mesh_first_layer(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, layerType, zigzag)
    X, Elems, U0 = mesh.mesh_structure(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, zigzag)
    weldDof = weld.welding_conditions(X, meshing)

    #meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag # Tupple
    X = shapeOffset.mesh_offset(meshing, offset, X, Elems)

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

        # making split weldDof rely on weldDof weights computed without splitbeads
        extended_weldDof = shapeOffset.make_extended_weldDof(meshing, offset, weldDof)
        extended_weldDof_split = splitBeads.extended_split_weldDof(meshing_split, extended_weldDof, overhang_n)

        X, Elems, U0 = X_split, Elems_split, U0_split
        meshing = meshing_split
        #offset = offset_split
        offset = offset_new
        weldDof_unsplit = weldDof.copy()
        weldDof = extended_weldDof_split[:,:4].astype(int)    # ie weldDof_split

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

    ## Welding matrices
    Sni, Sn = weld.weldnode_matrices(Elems, uncElems)

    Sw0 = weld.weldwire_matrix(nNodesTot, weldDof)
    if bool(split_bool):
        Sw = splitBeads.offset_split_weldwire_matrix(meshing, extended_weldDof_split, Sw0)
    else :
        Sw = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw0)

    #plot.plotmatrix_sparse_t(Sw, nNodesTot, f'{nLayers_v} couches de {nLayers_h} cordons \n offset {int(offset[-1][-1] * 100)}% (Noeuds pairs, axe t)')

    Y0 = weld.bcs_matrix(U0, Sw0)
    Y = weld.bcs_matrix(U0, Sw)
    # matrice de passage noeud alpha -> qp beta
    Assemble = Ta @ P.T @ Sn

    ## Behavior
    # material parameters
    k0 = E / (3 * (1 - 2 * nu))  # bulk modulus (K)
    mu = E / (2 * (1 + nu))  # shear modulus (G)
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # k0-2/3*mu

    ## Local behavior
    if split_bool :
        Hn /= 2      # assume offset = 50%
    Btot = behavior.derivation_xi(Hn, Hb, N, Ds)
    Ctot = behavior.derivation_chi(Hn, Hb, N, Ds)
    if optimizedBehavior:
        Rxi, Rchi = behavior.optimization_Rxi_Rchi()
    else:
        Rxi = behavior.homogeneization_Rxi(Hn, Hb, lmbda, mu)
        Rchi = behavior.homogenization_Rchi(L, Hn, Hb, lmbda, mu, nNodes)

    # if split_bool :
    #     Hn *= 2

    ## Local stiffness matrix
    Kxi = behavior.assembly_behavior(Rxi, Btot, W)
    Kchi = behavior.assembly_behavior(Rchi, Ctot, W)
    K = Kxi + Kchi

    ### Thermal eigenstrain
    dTalpha = behavior.dTfcn(N @ Xunc, dT, loadType, nLayers_h, nLayers_v)
    dTbeta = dTalpha @ T
    dTmoy = dTbeta[:, 0][:, np.newaxis]

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

    ## Construction of the force vector
    # Thermal eigenstrain
    Eps_thermal = behavior.thermal_eigenstrain(alpha, dTmoy)
    f_thermal = Btot.T @ sp.sparse.kron(Rxi, W) @ Eps_thermal
    # Assembly of the force vector
    f = np.zeros(nUncNodes * nNodeDOF)
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
    # nodal displacement 
    uUncouple = Assemble @ u  # beta configuration
    uUncouple_alpha =  P.T @ Sn @ u # alpha configuration

    ## Strains
    # Initialization
    Eps = np.zeros(((6 + 9) * nQP, 1)) # (nInc, nQP) sigma_xi = Rxi @ (Eps_xi - Eps_thermique)
    Eps_xi = np.zeros((6*nQP, 1))
    Eps_chi = np.zeros((9*nQP, 1))
    # Update
    Eps += sp.sparse.vstack((Btot, Ctot)) @ uUncouple  
    Eps_xi += Btot @ uUncouple
    Eps_chi += Ctot @ uUncouple
    
    
    ## Stresses
    # Initialization
    Sigma = np.zeros(((6 + 9) * nQP, 1))  # 6 strain components and 9 curvature components
    Sigma_s = np.zeros((6*nQP, 1))
    Sigma_m = np.zeros((9*nQP, 1))
    # Rtot : Mauvais nom
    Rtot = sp.sparse.block_diag((sp.sparse.kron(Rxi, W), sp.sparse.kron(Rchi, W)))
    # Update
    Sigma +=  Rtot @ Eps
    Sigma -=  sp.sparse.vstack((sp.sparse.kron(Rxi, W) @ Eps_thermal, sp.sparse.csr_matrix((9 * nQP, 1)) ))
    Sigma = np.array(Sigma) ## changement de type de variable donc il faut la repasser en ndarray wtf ?

    Sigma_s += sp.sparse.kron(Rxi, W) @ (Eps_xi - Eps_thermal)
    Sigma_m +=  sp.sparse.kron(Rchi, W) @ Eps_chi

    # Sigmatt elem
    sigmatt_elem = qp2elem @ Sigma[:nQP]/(Hn*Hb)

    ### Energy
    Eps_th = sp.sparse.vstack((Eps_thermal, sp.sparse.csr_matrix((9*nQP, 1)))).toarray()

    qpEnergyDensity = behavior.energyDensity(Eps, Eps_th, Sigma, Rtot, quadOrder, 15)
    nrg = qp2elem @ qpEnergyDensity
    
    ### Plot
    
    # mean basis vectors on nodes
    Me = sp.sparse.csr_matrix((np.ones(Elems.size), (Elems.flatten(), np.arange(Elems.size))))
    nm = Me @ n;
    nm = nm / np.linalg.norm(nm, axis=1)[:, np.newaxis]
    bm = Me @ b;
    bm = bm / np.linalg.norm(bm, axis=1)[:, np.newaxis]

    # undeformed shape
    x0 = X[:, :, np.newaxis] + 0.5 * (
            Hn * nm[:, :, np.newaxis] * np.array([[[-1, 1, -1, 1]]]) + Hb * bm[:, :, np.newaxis] * np.array(
        [[[1, 1, -1, -1]]]))
    # deformed shape
    if np.max(abs(U), (0, 1, 2)) > 0 :
        scale = .25 * np.linalg.norm([Hn, Hb]) / np.max(abs(U), (0, 1, 2))
    else :
        scale = 0
    uplot = np.moveaxis(U, (0, 1, 2), (1, 2, 0))
    x = x0 + scale * uplot

    if toPlot :
        fig = plt.figure()
        ax = plt.axes(projection='3d', proj_type='ortho')
        # ax.set_axis_off()

        ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scfplot), np.ptp(x[:, 2])*scfplot))

        if clrmap=="stt":
            clr = sigmatt_elem[:,0]
            # sigmatt = sp.sparse.linalg.lsqr(N @ Sni, Sigma[0:nQP])[0][:, None]
            # clr = sigmatt[:, None] * [[1, 1, 1, 1]]
        elif clrmap=="nrg" :
            clr = nrg
        elif clrmap=="temp" :
            clr = nrg
        # clr = sigmatt_elem[:,0]
        # srf.set_array(np.mean(clr.flatten()[tri], axis=1))
        # srf.set_clim(np.nanmin(clr), np.nanmax(clr))  

        srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='black', clrfun=clr, outer=False)
        colorbar = plt.colorbar(srf, pad=0.15)

        if clrmap=="stt":
            colorbar.set_label('$\Sigma_{tt}$ [N]')
        elif clrmap=="nrg" :
            colorbar.set_label("$\psi$ [J.mm$^{-1}$]")

        ax.set_xlabel('Axe t')
        ax.set_ylabel('Axe n')
        ax.set_zlabel('Axe b')

        plt.show()

    # if toPlot :
    #     plot.plot_sigma(Sigma, qp2elem, nQP, x, Elems, L, Hn, Hb, 'stt', scfplot)
    #     plot.plot_sigma(Sigma, qp2elem, nQP, x, Elems, L, Hn, Hb, 'stn', scfplot)
    #     plot.plot_sigma(Sigma, qp2elem, nQP, x, Elems, L, Hn, Hb, 'stb', scfplot)

    return U, Eps, Sigma, nrg, qp2elem, nQP, x0, Elems
