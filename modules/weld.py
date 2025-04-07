""" Module containing the welding functions of the QuadWire model """
# %% import packages
import numpy as np
import scipy as sp
from scipy import sparse


# %% function welding_conditions is moved from mesh module to weld module
def welding_conditions(X, meshing):
    """
    Function that returns a matrix of wire welding conditions of the structure

    Parameters
    ----------
    X : array of size (nNodes, 3)
        Coordinates of each node in the global reference (t,n,b).
    meshing : bool
        If True then successive beads are printed in back and forth directions.

    Returns
    -------
    weldDof : array of size (nNodes*(nLayers-1)*2, 4) - number of layer interface * two welding conditions
        Array listing the welding conditions between the wires
        The k^th welding condition is given by : weldDof[k] = [indexNode1, node1Particule, indexNode2, node2Particule] which means that the node1Particule chosen is welded to the node2Particule
        There are nInterface*nNodes*2 welding conditions in between wires
    """
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing

    if nLayers_h == 1:  # thin wall
        weldNodes = np.arange(nNodes)[:, np.newaxis] * [1, 1] + [0, nNodes]
        if meshing:
            weldNodes[:, 0] = weldNodes[::-1, 0]
        weldNodes = np.tile(weldNodes, (nLayers_v - 1, 1)) + np.repeat(nNodes * np.arange(nLayers_v - 1)[:, np.newaxis], nNodes, axis=0)
        ## welding TL to BL and TR to BR: weldDof = [ [[iNode,0,iNode+nNodes,2],[iNode,1,iNode+nNodes,3]] for iNode in range(nNodes)]
        weldDof = np.tile(weldNodes[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 0, 0, 2], [0, 1, 0, 3]], weldNodes.shape[0], axis=0)  # vertical welding (on top of each other)

    elif nLayers_v == 1:  # carpet
        weldNodes = np.arange(nNodes)[:, np.newaxis] * [1, 1] + [0, nNodes]
        if meshing:
            weldNodes[:, 0] = weldNodes[::-1, 0]
        weldNodes = np.tile(weldNodes, (nLayers_h - 1, 1)) + np.repeat(nNodes * np.arange(nLayers_h - 1)[:, np.newaxis], nNodes, axis=0)
        ## welding TR to TL and BR to BL: weldDof = [ [[iNode,1,iNode+nNodes,0],[iNode,3,iNode+nNodes,2]] for iNode in range(nNodes)]
        weldDof = np.tile(weldNodes[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 1, 0, 0], [0, 3, 0, 2]], weldNodes.shape[0], axis=0)  # horizontal welding (next to each other)

    else:  # any nLayers_h x nLayers_v
        # Calculate the total number of nodes in each horizontal layer
        nNodes_h = nNodes * nLayers_h

        # Calculate nodes for horizontal welding
        # First layer carpet
        weldNodes_h = np.arange(nNodes)[:, np.newaxis] * [1, 1] + [0, nNodes]
        if meshing:
            weldNodes_h[:, 0] = weldNodes_h[::-1, 0]
        weldNodes_h = np.tile(weldNodes_h, (nLayers_h - 1, 1)) + np.repeat(nNodes * np.arange(nLayers_h - 1)[:, np.newaxis], nNodes, axis=0)

        # Calculate degrees of freedom for horizontal welding
        # Identifying nodes next to each other for welding
        weldDof_h = np.tile(weldNodes_h[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 1, 0, 0], [0, 3, 0, 2]], weldNodes_h.shape[0], axis=0)

        # Horizontal welding (next to each other)
        weldDof = weldDof_h

        # Expand horizontal welding for multiple vertical layers
        # All horizontal layers repeated for each vertical layer
        weldNodes_h = np.tile(weldNodes_h, (nLayers_v, 1)) + np.repeat(nLayers_h * nNodes * np.arange(nLayers_v)[:, np.newaxis], (nLayers_h - 1) * nNodes, axis=0)
        weldDof_h = np.tile(weldNodes_h[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 1, 0, 0], [0, 3, 0, 2]], weldNodes_h.shape[0], axis=0)

        # Identify nodes for vertical welding
        # Sort nodes by X and Y coordinates to group nodes needing vertical welding
        xyzNodes = np.concatenate((X, np.arange(len(X))[:, np.newaxis]), axis=1)
        xxyyzNodes = xyzNodes[np.lexsort((xyzNodes[:, 1], xyzNodes[:, 0]))]
        xxyyzzNodes = np.reshape(xxyyzNodes[:, -1:], (nNodes_h, nLayers_v))
        xxyyzzNNodes = np.reshape(np.repeat(xxyyzzNodes, 2, axis=1)[:, 1:-1], (nNodes_h, 2, nLayers_v - 1))

        # Calculate nodes for vertical welding
        weldNodes_v = np.reshape(xxyyzzNNodes, (nNodes_h * (nLayers_v - 1), 2))

        # Calculate degrees of freedom for vertical welding
        # Identify pairs of nodes to be welded vertically
        weldDof_v = np.tile(weldNodes_v[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 0, 0, 2], [0, 1, 0, 3]], weldNodes_v.shape[0], axis=0)

        # Concatenate horizontal and vertical welding degrees of freedom
        weldDof = np.concatenate((weldDof_h, weldDof_v), axis=0)
        weldDof = weldDof.astype(int)  # TODO: corriger weldDof zigzag pour avoir les colomnes 0,1 leader et 2,3 suiveur
    return weldDof


def weldnode_matrices(Elems, uncElems):
    """
    Function that builds the node welding matrices 

    Parameters
    ----------
    Elems : array of size  (nElemTot, elemOrder +1)
        Index of the pair of nodes forming an element.
    uncElems : array of size or (nElemTot, elemOrder +1)
        Index of the pair of nodes forming an uncoupled element.

    Returns
    -------
    Sni : sparse matrix of shape (nUncNodes, nNodesTot)
        Welding matrix for the node functions.
    Sn : sparse matrix of shape (nUncNodes*nNodeDof, nNodesTot*nNodeDof) 
        Welding matrix for the degrees of freedom 3*4*Sni.

    """
    nElems = Elems.shape[0]
    nNodesByElem = Elems.shape[1]

    # Weld uncoupled nodes
    Sni = sp.sparse.csr_matrix((np.ones((nElems * nNodesByElem,)), (uncElems.flatten(), Elems.flatten())))

    Sn = sp.sparse.kron(sp.sparse.eye(12), Sni).tocsr()

    return Sni, Sn


def weldwire_matrix(nNodesTot, weldDof):
    """
    Function that builds the wire welding matrices

    Parameters
    ----------
    nNodesTot : int
        Number of total nodes discretising the structures.
        nNodesTot = X.shape[0]
    weldDof : array of size (nNodes*(nLayers-1)*2, 4) - number of layer interface * two welding conditions on particles
        Array listing the welding conditions between the wires 
        The k^th welding condition is given by : weldDof[k] = [indexNode1, node1Particule, indexNode2, node2Particule] which means that the node1Particule chosen is welded to the node2Particule
        There are nInterface*nNodes*2 welding conditions in between wires.

    Returns
    -------
    Sw : sparse matrix of shape (nNodesTot*nNodeDof, nNodesTot*nNodeDof)
        Inter-wire welding matrix / wire connectivity matrix 

    """
    # There are welding conditions to apply
    if np.array(weldDof).size != 0:
        # DOF indexes
        wDOFidx = weldDof[:, [0, 2]] + nNodesTot * weldDof[:, [1, 3]]  # le couple nœud/particule des colonnes 0 et 1 est LEADER du couple nœud/particule des colonnes 2 et 3 qui est SUIVEUR
        # Repeat for all 3 coordinates
        wDOFidx = np.tile(wDOFidx, (3, 1)) + np.repeat(4 * nNodesTot * np.arange(3)[:, np.newaxis], wDOFidx.shape[0], axis=0)
        # Leader particles and follower particles
        wDOFidx_lead = wDOFidx[:, 0]
        wDOFidx_follow = wDOFidx[:, 1]
        noWeld = np.argwhere(np.invert(np.isin(np.arange(12 * nNodesTot), wDOFidx.flatten())))
        # Build welding matrix
        ii_lead = wDOFidx_lead
        jj_lead = ii_lead  # leader particles remain on diagonal
        ii_follow = wDOFidx_follow
        jj_follow = jj_lead  # follower particles are shifted to their respective leader particle's column

        Sw_follow = sp.sparse.csr_matrix((np.ones(ii_follow.shape), (ii_follow, jj_follow)), shape=(12 * nNodesTot, 12 * nNodesTot))
        Sw_lead = sp.sparse.csr_matrix((np.ones(12 * nNodesTot), (np.arange(12 * nNodesTot), np.arange(12 * nNodesTot))), shape=(12 * nNodesTot, 12 * nNodesTot))  # identité CSR
        Sw_lead[ii_follow, ii_follow] = 0  # any particle that is a follower is not a leader

        Sw = Sw_follow + Sw_lead
        # Particles following follow particles should be attached to primary leader particle : Sw @ Sw allows this
        Sw = Sw @ Sw
        # Particles following two particles need to be normalized : set all values to 1
        Sw = Sw.sign()

    # No welded DOFs, gives Identity matrix
    else:
        Sw = sp.sparse.eye(12 * nNodesTot)

    return Sw


def bcs_matrix(U0, Sw):
    """
    Function that builds the dof removing matrix.

    Parameters
    ----------
    U0 : array of size (nCoord, nParticule, nNodes*nLayers)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. Particules are indexed as {TL,TR,BL,BR}. 
    Sw : sparse matrix of shape (nNodesTot*nNodeDof, nNodesTot*nNodeDof)
        Inter-wire welding matrix / wire connectivity matrix.

    Returns
    -------
    Y : sparse matrix of size (nNodesTot*nNodeDof, nNodesTot*nNodeDof/2) 
        /2 because two particules are used to weld two wires
        Degrees of freedom removing matrix.

    """
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
    return Y

def offset2weld(meshing, offset):
    from modules import mesh
    X, Elems, U0 = mesh.generate_trajectory(meshing)
    nNodesTot = X.shape[0]

    weldDof = welding_conditions(X, meshing)
    Sw0 = weldwire_matrix(nNodesTot, weldDof)
    from shape import shapeOffset
    Sw = shapeOffset.offset_weldwire_matrix(meshing, offset, weldDof, Sw0)
    return Sw

def offset2bcs(meshing, offset):
    from modules.mesh import generate_trajectory
    X, Elems, U0 = generate_trajectory(meshing)
    nNodesTot = X.shape[0]

    weldDof = welding_conditions(X, meshing)
    Sw0 = weldwire_matrix(nNodesTot, weldDof)
    from shape.shapeOffset import offset_weldwire_matrix
    Sw = offset_weldwire_matrix(meshing, offset, weldDof, Sw0)
    Y = bcs_matrix(U0, Sw)
    return Y