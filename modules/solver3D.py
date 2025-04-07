import numpy as np
import scipy as sp


def cauchyTM(L, Hn, Hb, nNodes, nLayers, beadType, thermalLoading):
    E = 3e3  # 210e3  # Young modulus /!\ in MPa !
    nu = 0.3  # poisson ratio

    k0 = E / (3 * (1 - 2 * nu))  # bulk modulus (K)
    mu = E / (2 * (1 + nu))  # shear modulus (G)
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # k0-2/3*mu

    alpha = 1.13e-5  # 12e-6  # thermal expansion coefficient
    dT = -60  # -1000  #  temperature increase

    # 3DTM discretization
    ds = np.sqrt(Hn ** 2 + Hb ** 2) / 10  # approximate section discretization length
    nHn = int(np.ceil(Hn / ds)) + 1
    nHb = int(np.ceil(Hb / ds)) + 1

    weldDistTol = ds / 1000  # node welding distance tolerance

    debug = False  # display debugging elements

    if beadType == "linear":
        # Single straight bead welded on the platform
        X = np.linspace(0, L, nNodes)[:, np.newaxis] * [1, 0, 0]  # straight line along X1
        Elems = np.arange(nNodes - 1)[:, np.newaxis] + [0, 1]  # simple elements
        U0 = np.tile([[[np.nan], [np.nan], [0], [0]]], (3, 1, nNodes))  # clamp all {BL,BR} wires

    elif beadType == "circular":
        # Single circular bead welded on the platform
        theta = np.arange(nNodes)[:, np.newaxis] / nNodes * 2 * np.pi
        R = L / 2 / np.pi
        X = np.hstack((R * np.cos(theta), R * np.sin(theta), theta * 0))  # straight line along X1
        Elems = np.mod(np.arange(nNodes)[:, np.newaxis] + [0, 1], nNodes)  # simple elements
        U0 = np.tile([[[np.nan], [np.nan], [0], [0]]], (3, 1, nNodes))  # clamp all {BL,BR} wires

    elif beadType == "sinus":
        # Single sinusoidal bead welded on the platform
        x1 = np.linspace(0, L, nNodes)[:, np.newaxis]
        X = np.hstack((x1, .25 * L * np.sin(1.0 * x1 / L * 2 * np.pi), x1 * 0))
        Elems = np.arange(nNodes - 1)[:, np.newaxis] + [0, 1]  # simple elements
        U0 = np.tile([[[np.nan], [np.nan], [0], [0]]], (3, 1, nNodes))  # clamp all {BL,BR} wires

    # Cantilever beam
    # /!\ not working because the model does not include bending !
    # X = np.linspace(0,L,nNodes)[:,np.newaxis]*[0,1,0] # straight line along X1
    # Elems = np.arange(nNodes-1)[:,np.newaxis]+[0,1] # simple elements
    # U0 = np.tile(np.array([[[0],[0],[0],[0]]])*np.concatenate([[0],np.nan*np.zeros((nNodes-1))])[np.newaxis,np.newaxis,:],(3,1,1)) # clamp all wires of the first node

    # ## Stack the subsequent layers
    # "weldDOF" list of welded dofs (nWeldedDOFs,[nod0,{TL,TR,BL,BR}_0,nod1,{TL,TR,BL,BR}_1])
    # replicate the bead with z offset
    X = np.tile(X, (nLayers, 1)) + np.repeat(np.arange(nLayers)[:, np.newaxis] * [0, 0, Hb], nNodes, axis=0)
    Elems = np.tile(Elems, (nLayers, 1)) + nNodes * np.repeat(np.arange(nLayers)[:, np.newaxis], Elems.shape[0], axis=0)
    # let the new layers free
    U0 = np.concatenate((U0, np.tile(np.nan * U0, (1, 1, nLayers - 1))), axis=2)

    # ## Build the 3D mesh

    # Node tangents as mean over connected elements
    t = np.diff(X[Elems, :], 1, axis=1).reshape((Elems.shape[0], X.shape[1]))  # element tangent vector
    t = t / np.sqrt(np.sum(t ** 2, 1))[:, np.newaxis]  # normalization
    e2n = sp.sparse.csr_matrix((np.ones((Elems.size,)), (Elems.flatten(), np.repeat(np.arange(Elems.shape[0]), Elems.shape[1], 0))))  # elem-node connectivity matrix
    t = e2n @ t

    # Node frames
    t = t / np.sqrt(np.sum(t ** 2, 1))[:, np.newaxis]  # re-normalization
    n = t @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # normal vector rotated of 90° in the XY plane
    b = np.cross(t, n)

    # Replicate the 1D mesh to make it 3D
    # replicate along the normal
    Elems = np.tile(np.hstack((Elems, Elems[:, [1, 0]] + X.shape[0])), (nHn - 1, 1)) + X.shape[0] * np.repeat(np.arange(nHn - 1)[:, np.newaxis], Elems.shape[0], axis=0)
    X = np.tile(X, (nHn, 1)) + Hn * np.repeat(np.linspace(-.5, .5, nHn)[:, np.newaxis], X.shape[0], axis=0) * np.tile(n, (nHn, 1))
    # replicate along the B-vector
    Elems = np.tile(np.hstack((Elems, Elems + X.shape[0])), (nHb - 1, 1)) + X.shape[0] * np.repeat(np.arange(nHb - 1)[:, np.newaxis], Elems.shape[0], axis=0)
    X = np.tile(X, (nHb, 1)) + Hb * np.repeat(np.linspace(0, 1, nHb)[:, np.newaxis], X.shape[0], axis=0) * np.tile(b, (nHn * nHb, 1))

    # Replicate boundary conditions
    # wire "U0" (nCoord,{TL,TR,BL,BR},n1DNodes)
    # 3D "U0" (n3DNodes,nCoord)
    n1DNodes = int(X.shape[0] / nHb / nHn)
    U0_1D = np.moveaxis(U0, (0, 1, 2), (1, 2, 0))  # (n1DNodes,nCoord,{TL,TR,BL,BR})
    U0_3D = np.nan * np.ones(X.shape)  # initialize
    # Node indices in the section, helps wire->3D BC conversion
    iXn = np.tile(np.repeat(np.arange(nHn), n1DNodes, 0), (nHb,))
    iXb = np.repeat(np.arange(nHb), n1DNodes * nHn, 0)
    # Coupled wire BCs on section faces
    U0_3D[iXb == 0, :] = np.tile(U0_1D[:, :, 2] + U0_1D[:, :, 3], (nHn, 1))  # Bottom face
    U0_3D[iXb == nHb - 1, :] = np.tile(U0_1D[:, :, 0] + U0_1D[:, :, 1], (nHn, 1))  # Top face
    U0_3D[iXn == 0, :] = np.tile(U0_1D[:, :, 0] + U0_1D[:, :, 2], (nHb, 1))  # Left face
    U0_3D[iXn == nHn - 1, :] = np.tile(U0_1D[:, :, 1] + U0_1D[:, :, 3], (nHb, 1))  # Right face
    # Copy individual wire BCs on section edges
    U0_3D[np.logical_and(iXn == 0, iXb == 0), :] = U0_1D[:, :, 2]  # BL edge
    U0_3D[np.logical_and(iXn == nHn - 1, iXb == 0), :] = U0_1D[:, :, 3]  # BR edge
    U0_3D[np.logical_and(iXn == 0, iXb == nHb - 1), :] = U0_1D[:, :, 0]  # TL edge
    U0_3D[np.logical_and(iXn == nHn - 1, iXb == nHb - 1), :] = U0_1D[:, :, 1]  # TR edge

    # Weld duplicated nodes within a tolerance
    tree = sp.spatial.KDTree(X)
    nodepairs = np.array(list(tree.query_pairs(weldDistTol)))

    if nodepairs.size != 0:
        lonenodes = np.setdiff1d(np.arange(X.shape[0]), nodepairs.flatten())
        newNodeIdx = np.arange(X.shape[0])
        newNodeIdx[lonenodes] = np.arange(lonenodes.size)
        newNodeIdx[nodepairs[:, 0]] = lonenodes.size + np.arange(nodepairs.shape[0])
        newNodeIdx[nodepairs[:, 1]] = newNodeIdx[nodepairs[:, 0]]
        S = sp.sparse.csr_matrix((np.ones(X.shape[0]), (newNodeIdx, np.arange(X.shape[0]))))  # welding matrix
        X = (S @ X) / np.array(np.sum(S, 1))  # mean over welded nodes
        U0_3D = (S @ U0_3D)  # /np.array(np.sum(S,1)) # mean over welded BCs
        Elems = newNodeIdx[Elems]  # new element-node connectivity

    # ## Finite Element Definition

    # Two-nodes P1 element
    xiN = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])  # node reference coordinates

    # Shape Functions and derivatives in reference coordinates
    def ShapeFun(xi):
        return np.concatenate([((1 - xi[:, 0]) * (1 - xi[:, 1]) * (1 - xi[:, 2]))[:, np.newaxis], (xi[:, 0] * (1 - xi[:, 1]) * (1 - xi[:, 2]))[:, np.newaxis], (xi[:, 0] * xi[:, 1] * (1 - xi[:, 2]))[:, np.newaxis],
            ((1 - xi[:, 0]) * xi[:, 1] * (1 - xi[:, 2]))[:, np.newaxis], ((1 - xi[:, 0]) * (1 - xi[:, 1]) * xi[:, 2])[:, np.newaxis], (xi[:, 0] * (1 - xi[:, 1]) * xi[:, 2])[:, np.newaxis], (xi[:, 0] * xi[:, 1] * xi[:, 2])[:, np.newaxis],
            ((1 - xi[:, 0]) * xi[:, 1] * xi[:, 2])[:, np.newaxis]], 1)

    def dShapeFun_dxi(xi):
        return [np.concatenate([  # df_dxi1
            (-(1 - xi[:, 1]) * (1 - xi[:, 2]))[:, np.newaxis], ((1 - xi[:, 1]) * (1 - xi[:, 2]))[:, np.newaxis], (xi[:, 1] * (1 - xi[:, 2]))[:, np.newaxis], (-xi[:, 1] * (1 - xi[:, 2]))[:, np.newaxis],
            (-(1 - xi[:, 1]) * xi[:, 2])[:, np.newaxis], ((1 - xi[:, 1]) * xi[:, 2])[:, np.newaxis], (xi[:, 1] * xi[:, 2])[:, np.newaxis], (-xi[:, 1] * xi[:, 2])[:, np.newaxis]], 1), np.concatenate([  # df_dxi2
            (-(1 - xi[:, 0]) * (1 - xi[:, 2]))[:, np.newaxis], (-xi[:, 0] * (1 - xi[:, 2]))[:, np.newaxis], (xi[:, 0] * (1 - xi[:, 2]))[:, np.newaxis], ((1 - xi[:, 0]) * (1 - xi[:, 2]))[:, np.newaxis],
            (-(1 - xi[:, 0]) * xi[:, 2])[:, np.newaxis], (-xi[:, 0] * xi[:, 2])[:, np.newaxis], (xi[:, 0] * xi[:, 2])[:, np.newaxis], ((1 - xi[:, 0]) * xi[:, 2])[:, np.newaxis]], 1), np.concatenate([  # df_dxi3
            (-(1 - xi[:, 0]) * (1 - xi[:, 1]))[:, np.newaxis], (-xi[:, 0] * (1 - xi[:, 1]))[:, np.newaxis], (-xi[:, 0] * xi[:, 1])[:, np.newaxis], (-(1 - xi[:, 0]) * xi[:, 1])[:, np.newaxis],
            ((1 - xi[:, 0]) * (1 - xi[:, 1]))[:, np.newaxis], (xi[:, 0] * (1 - xi[:, 1]))[:, np.newaxis], (xi[:, 0] * xi[:, 1])[:, np.newaxis], ((1 - xi[:, 0]) * xi[:, 1])[:, np.newaxis]], 1)]

    # Element quadrature
    # xiQ = np.array([[.5,.5,.5]]) # quadrature point reference coordinates
    # wQ = np.array([1]) # quadrature weights
    xiQ = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * 0.5 / np.sqrt(3) + np.array([[.5, .5, .5]])  # quadrature point reference coordinates
    wQ = 1 / 8 * np.array([1, 1, 1, 1, 1, 1, 1, 1])  # quadrature weights

    # ## Interpolation matrices
    nNodes3D = X.shape[0]  # previously, nNodes was meant BY 1D LAYER, from now on it refers to all nodes in the 3D mesh
    nCoord = X.shape[1]  # nCoord = 3
    nElems = Elems.shape[0]
    nNodesByElem = Elems.shape[1]
    nUncNodes = nElems * nNodesByElem

    # Full mesh quadrature
    XIQ = np.tile(xiQ, (nElems, 1))  # points
    WQ = np.tile(wQ[:, np.newaxis], (nElems, 1))  # weights
    elemQ = np.repeat(np.arange(nElems), len(xiQ))  # corresponding element index
    nQP = WQ.size  # number of quadrature points

    # Sparse matrix indices
    ii = np.tile(np.arange(nQP)[:, np.newaxis], (1, nNodesByElem))  # row indices
    jj = Elems[elemQ, :]  # column indices

    # Function interpolation
    # f(Xqp) = N.f(Xunc)
    nn = ShapeFun(XIQ)  # fill values (nElems,nUncNodes)
    N = sp.sparse.csr_matrix((nn.flatten(), (ii.flatten(), jj.flatten())))

    # Function partial derivatives
    # df_dxi[c](Xqp) = Dxi[c].f(Xunc) derivatives in REFERENCE coordinates
    # df_dX[c](Xqp) = Dx[c].f(Xunc) derivatives in GLOBAL coordinates
    # where df_ds = J^{-1}.df_dxi, J = ds_dxi
    gg = dShapeFun_dxi(XIQ)  # fill values (nQP,nNodesByElem)
    Dxi = [sp.sparse.csr_matrix((gg[comp].flatten(), (ii.flatten(), jj.flatten()))) for comp in range(nCoord)]  # ref derivation matrices nCoord*(nQP,nUncNodes)
    dX_dxi = [Dxi[comp].dot(X) for comp in range(nCoord)]  # ref coordinate derivatives (nQP,nCoord)
    # jacobian and inverse iJ = 1/detJ*comatJ.T
    J = np.concatenate([dX_dxi[comp][:, :, np.newaxis] for comp in range(nCoord)], 2)  # Jacobian matrices (nQP,nGlobalCoord,nLocalCoord==3)
    detJ = (J[:, 1, 0] * J[:, 2, 0] - J[:, 2, 0] * J[:, 2, 0]) * J[:, 0, 2] - (J[:, 0, 0] * J[:, 2, 1] - J[:, 2, 0] * J[:, 0, 1]) * J[:, 1, 2] + (J[:, 0, 0] * J[:, 1, 1] - J[:, 1, 0] * J[:, 0, 1]) * J[:, 2, 2]
    iJ = (1 / detJ[:, np.newaxis, np.newaxis]) * np.moveaxis(np.array(  # next lines define the comatrix components
        [[J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1], -J[:, 1, 0] * J[:, 2, 2] + J[:, 1, 2] * J[:, 2, 0], J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0]],
         [-J[:, 0, 1] * J[:, 2, 2] + J[:, 0, 2] * J[:, 2, 1], J[:, 0, 0] * J[:, 2, 2] - J[:, 0, 2] * J[:, 2, 0], -J[:, 0, 0] * J[:, 2, 1] + J[:, 0, 1] * J[:, 2, 0]],
         [J[:, 0, 1] * J[:, 1, 2] - J[:, 0, 2] * J[:, 1, 1], -J[:, 0, 0] * J[:, 1, 2] + J[:, 0, 2] * J[:, 1, 0], J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]]]), (0, 1, 2), (2, 1, 0))  # includes the comatrix transpose
    Dx = [np.sum([sp.sparse.diags(iJ[:, cXI, cX]).dot(Dxi[cXI]) for cXI in range(nCoord)]) for cX in range(nCoord)]  # global derivation matrices nCoord*(nQP,nUncNodes)
    if debug: print('derivative error: ', np.sqrt(np.sum(abs(Dx[0] @ X - [1, 0, 0]) ** 2) / nQP), ',', np.sqrt(np.sum(abs(Dx[1] @ X - [0, 1, 0]) ** 2) / nQP), ',', np.sqrt(np.sum(abs(Dx[2] @ X - [0, 0, 1]) ** 2) / nQP))

    # Integral weight matrix including local jacobian
    W = sp.sparse.diags((WQ * detJ[:, np.newaxis]).flatten())  # (nQP,nQP)
    if debug: print('mesh volume V: ', np.sum(abs(W.diagonal())), '(==', L * Hb * Hn, ')')  # L = integral of the jacobian

    # null matrix
    O = sp.sparse.csr_matrix(N.shape)  # (nQP,nUncNodes)

    # ## System matrices
    nDOF = nCoord * nNodes3D

    # Thermo-mechanical equilibrium : \int{eps*C*(eps-alpha*dT*Id)} = 0
    # eps = [E11 E22 E33 2*E13 2*E23 2*E12] = B @ [u1 u2 u3]
    B = sp.sparse.bmat([[Dx[0], O, O], [O, Dx[1], O], [O, O, Dx[2]], [Dx[2], O, Dx[0]], [O, Dx[2], Dx[1]], [Dx[1], Dx[0], O]])  # strain matrix (6*nQP,nDOF)
    C = (E / (1 + nu) / (1 - 2 * nu)) * np.array([[1 - nu, nu, nu, 0, 0, 0], [nu, 1 - nu, nu, 0, 0, 0], [nu, nu, 1 - nu, 0, 0, 0], [0, 0, 0, .5 - nu, 0, 0], [0, 0, 0, 0, .5 - nu, 0], [0, 0, 0, 0, 0, .5 - nu], ])  # "material stiffness"
    Cw = sp.sparse.kron(sp.sparse.csr_matrix(C), W)
    K = B.T @ Cw @ B  # left-hand side (nDOF,nDOF)

    if thermalLoading == "uniform":
        # homogeneous cooling
        dTmoy = dT * np.ones((1, nQP))
    elif thermalLoading == "linear":
        # linear cooling
        dTmoy = N @ (X[:, 0] / L * dT)
    elif thermalLoading == "quad":
        dTmoy = N @ (dT * (np.ones(X.shape[0]) + X[:, 0] * 2 / L) * (np.ones(X.shape[0]) - X[:, 0] * 2 / L))

    f = B.T @ (Cw @ np.kron(np.array([[alpha, alpha, alpha, 0, 0, 0]]).T, dTmoy).reshape(6 * nQP, 1))  # right-hand side (nUncDOF,1)

    # ## Boundary Conditions
    u0 = U0_3D.T.flatten()
    freeDOF = np.where(np.isnan(u0))[0]  # remaining DOF indices
    Y = sp.sparse.csr_matrix((np.ones(freeDOF.size), (freeDOF, np.arange(freeDOF.size))), shape=(nDOF, freeDOF.size))

    # ## Solve the system

    # Modify right-hand-side with BCs
    u00 = u0[:, np.newaxis].copy()  # hard copy is needed to keep the u0 vector unmodified
    u00[freeDOF] = 0  # cull NaNs
    f = f - K @ u00

    # Delete BC DOFs
    K = Y.T @ K @ Y
    f = Y.T @ f

    # Solve
    us = sp.sparse.linalg.spsolve(K, f)  # énergie us.T @ f
    U_dof = us[:, np.newaxis]

    return U_dof, B, Y, C, nQP, dTmoy, xiQ, elemQ


def psi_3dtm(U_dof, B, Y, C, nQP, dTmoy, xiQ, elemQ, alpha, Hn, Hb, nNodes, nLayers, circ=0):
    # 3DTM discretization
    ds = np.sqrt(Hn ** 2 + Hb ** 2) / 10  # approximate section discretization length
    nHn = int(np.ceil(Hn / ds)) + 1
    nHb = int(np.ceil(Hb / ds)) + 1

    # size (6, 8 * (nHb-1) * (nHn-1) * (nNodesIni-1))
    epsilon_3D = np.reshape(B @ Y @ U_dof, (6, nQP))
    alpha_dT_3D = np.reshape(np.kron(np.array([[alpha, alpha, alpha, 0, 0, 0]]).T, dTmoy), (6, nQP))
    # size nQP X 1
    energy_3dtm = (0.5 * np.sum(epsilon_3D * (C @ epsilon_3D), axis=0)[:, np.newaxis] - np.sum(epsilon_3D * (C @ alpha_dT_3D), axis=0)[:, np.newaxis])

    # M4 takes the sum of 4 gauss points (eta = 0.5 + 0.5 / np.sqrt(3) and eta = 0.5 - 0.5 / np.sqrt(3))
    # size (2 * (nHb-1) * (nHn-1) * (nNodesIni-1), nQP)
    M4 = (1 / 4) * (Hn / (nHn - 1)) * (Hb / (nHb - 1)) * sp.sparse.csr_matrix((np.concatenate((np.tile((xiQ[:, 0] == 0.5 - 0.5 / np.sqrt(3)).astype(int), nQP // 8), np.tile(1 - (xiQ[:, 0] == 0.5 - 0.5 / np.sqrt(3)).astype(int), nQP // 8))),
                                                                               ((2 * elemQ.flatten() + [[0], [1]]).flatten(), np.tile(np.arange(nQP), 2))))

    energy_3dtm = (M4 @ energy_3dtm)
    energy_3dtm = energy_3dtm.reshape(((nHb - 1) * (nHn - 1), 2 * (nNodes - 1 + circ) * nLayers))
    energy_3dtm = np.sum(energy_3dtm, axis=0).T.reshape((-1, 2))
    energy_3dtm = (energy_3dtm[:, 0] + energy_3dtm[:, 1]) / 2
    return energy_3dtm
