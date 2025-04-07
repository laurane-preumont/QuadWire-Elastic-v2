
""" Module defining the finite element components and the transition matrices between different function evaluation locations (node, quadrature points) and coordinate systems (primary, secondary) of the QuadWire program """


#%% import packages
import numpy as np
import scipy as sp
from modules.mesh import uncouple_nodes

#%% Finite Element Definition
    
def shape_function(xi, elemOrder):
    """
    Function that returns the shape fonction (or interpolation fonction)
    f(Xqp) = N.f(Xunc)
    
    Parameters
    ----------
    xi : array of size ((nNodes-1)*nLayers, 1)
        Coordinates of the quadrature points.
    elemOrder : int
        Element order : number of nodes that form an element
        Options are 1 for two-nodes P1 element and 2 for three-nodes P2 element

    Returns
    -------
    array of size (nQP, nNodesByElem, 1)
        Shape function.
    
    NB : nQP = quadOrder*(nNodes-1)*nLayers
    """
       
    if elemOrder == 1:
        return np.concatenate([1 - xi[:, np.newaxis], xi[:, np.newaxis]], 1)  
    
    elif elemOrder ==2:
        return np.concatenate([1 - 3 * xi[:, np.newaxis] + 2 * xi[:, np.newaxis] ** 2, 4 * (xi[:, np.newaxis] - xi[:, np.newaxis] ** 2),
             -1 * xi[:, np.newaxis] + 2 * xi[:, np.newaxis] ** 2], 1)

def shape_function_derivative(xi, elemOrder):
    """
    Derivation of the shape function d_Shape/d_xi in the reference coordinates 
    df_dxi(Xqp) = Dxi.f(Xunc)
    
    Parameters
    ----------
    xi : array of size ((nNodes-1)*nLayers, 1)
        Coordinates of the quadrature points.
    elemOrder : int, optional
        Element order : number of nodes that form an element
        Options are 1 for two-nodes P1 element and 2 for three-nodes P2 element

    Returns
    -------
    array of size (nQP, nNodesByElem)
        Derivation of the shape function.

    """
    
    if elemOrder == 1:
        return np.concatenate([-np.ones((xi.size, 1)), np.ones((xi.size, 1))], 1)
    
    elif elemOrder ==2:
        return np.concatenate([-3 + 4 * xi[:, np.newaxis], 4 * (1 - 2 * xi[:, np.newaxis]), -1 + 4 * xi[:, np.newaxis]], 1)

def element_quadrature(quadOrder):
    """
    Function that returns an element quadrature

    Parameters
    ----------
    quadOrder : int
        Element quadrature order.

    Returns
    -------
    xiQ : array of size (quadOrder, )
        Coordinates of the quadrature point reference coordinates.
    wQ : array of size (quadOrder, )
        Quadrature weights.

    """
    if quadOrder == 1:
        xiQ = np.array([.5])
        wQ = np.array([1])
    elif quadOrder == 2:
        xiQ = (1 + np.array([-1, 1]) / np.sqrt(3)) / 2
        wQ = np.array([.5, .5])
    elif quadOrder == 3:
        xiQ = (1 + np.array([-1, 0, 1]) * np.sqrt(.6)) / 2
        wQ = np.array([5, 8, 5]) / 18
    return xiQ, wQ

def fullmesh_quadrature(quadOrder, nElemsTot):
    """
    Function that returns the full mesh quadrature

    Parameters
    ----------
    quadOrder : int
        Element quadrature order.
    nElemsTot : int
        Number of element in the whole structure.

    Returns
    -------
    XIQ : array of size (quadOrder*(nNodes-1)*nLayers, 1) or (nQP, 1)
        Coordinates of the quadrature points.
    WQ : array of size (quadOrder*(nNodes-1)*nLayers, 1) or (nQP, 1)
        Weight of the quadrature points.

    """
    # Single element quadrature
    xiQ, wQ = element_quadrature(quadOrder)
    # Full mesh quadrature
    XIQ = np.tile(xiQ[:, np.newaxis], (nElemsTot, 1))  # points
    WQ = np.tile(wQ[:, np.newaxis], (nElemsTot, 1))  # weights
    return XIQ, WQ

def integration_matrices(X, Elems, elemOrder, quadOrder):
    """
    Return some useful sparse matrix that will be used in the FEM solver.
    Reminder : nQP = quadOrder*(nNodes-1)*nLayers = quadOrder * nElemTot
               nUncNodes = 2*(nNodes -2)*nLayers + nLayers*2
    
    Parameters
    ----------
    X : array of size (nNodes*nLayers, nCoord)
        Coordinates of each node in the global reference (t,n,b).
    Elems : array of size ((nNodes - 1)*nLayers, elemOrder +1 ) or (nElemTot, elemOrder +1)
        Index of the pair of nodes forming an element.
    elemOrder : int
        Element order : number of nodes that forms an element 
        Options are 1 for two-nodes P1 element and 2 for three-nodes P2 element
    quadOrder : int
        Element quadrature order.

    Returns
    -------
    N : sparse matrix of shape (nQP, nUncNodes) 
        Function of interpolation at the quadrature points : f(Xqp) = N.f(Xunc)
    Dxi : sparse matrix of shape (nQP, nUncNodes)
        Function of derivation in REFERENCE coordinates : df_dxi(Xqp) = Dxi.f(Xunc) .
    Ds : sparse matrix of shape (nQP, nUncNodes)
        Function derivation in LENGTH coordinates : df_ds(Xqp) = Ds.f(Xunc).
    J : array of size (nQP,)
        Jacobian = norm(dX_dxi)
    W : sparse matrix of shape (nQP, nQP)
        Integral weight matrix including local jacobian.
    O : sparse matrix of shape (nQP, nUncNodes)
        Null matrix. 
    qp2elem : sparse matrix of shape (nElemTOT, nQP)
        Transition matrix between the quadrature points and the elements
        f(elem) = qp2elem @ f(x_qp) 

    """
    nElems = Elems.shape[0]
    nNodesByElem = Elems.shape[1]
    
    xiQ, WQ = element_quadrature(quadOrder)
    XIQ, WQ = fullmesh_quadrature(quadOrder, nElems)
    nQP = WQ.shape[0]
    
    # Uncoupled coordonnates
    Xunc, uncElems = uncouple_nodes(X, Elems)
    
    # Quadrature points corresponding element index
    elemQ = np.repeat(np.arange(nElems), len(xiQ))  
    
    # QP to Element transfer matrix
    qp2elem = sp.sparse.csr_matrix((WQ.flatten(), (elemQ.flatten(), np.arange(0, nQP) )))

    # Sparse matrix indices
    ii = np.tile(np.arange(nQP)[:, np.newaxis], (1, nNodesByElem))  # row indices
    jj = uncElems[elemQ, :]  # column indices
    
    # Function interpolation -- f(Xqp) = N.f(Xunc)
    nn = shape_function(XIQ, elemOrder)  # fill values (nElems,nUncNodes)
    N = sp.sparse.csr_matrix((nn.flatten(), (ii.flatten(), jj.flatten())))
    
    # Function derivation in REFERENCE coordinates -- df_dxi(Xqp) = Dxi.f(Xunc)
    gg = shape_function_derivative(XIQ, elemOrder)  # fill values (nQP,nNodesByElem)
    Dxi = sp.sparse.csr_matrix((gg.flatten(), (ii.flatten(), jj.flatten())))  # ref derivation matrix (nQP,nUncNodes)
    
    # Function derivation in LENGTH coordinates -- df_ds(Xqp) = Ds.f(Xunc)
    dX_dxi = Dxi.dot(Xunc)  # ref coordinate derivative (nQP,nCoord)
    J = np.sqrt(np.sum(dX_dxi ** 2, 1))  # Jacobian = norm(dX_dxi)  (nQP,)
    Ds = sp.sparse.diags(J ** -1).dot(Dxi)  # length derivation matrix (nQP,nUncNodes)
    
    # Integral weight matrix including local jacobian
    W = sp.sparse.diags((WQ * J[:, np.newaxis]).flatten())  # (nQP,nQP)
    
    # Null matrix
    O = sp.sparse.csr_matrix(N.shape)  # (nQP,nUncNodes)
    
    return N, Dxi, Ds, J, W, O, qp2elem, elemQ

def alpha2beta_matrices(nNodesTot, nUncNodes):
    """
    Function that returns the matrices that enable the primary to secondary transfer for vector variables

    Parameters
    ----------
    nNodesTot : int
        Number of coupled nodes in the mesh
    nUncNodes : int
        Number uncoupled nodes in the mesh.

    Returns
    -------
    Tv : sparse matrix of shape (nNodeDOF, nNodeDOF) = (12, 12)
        Transition matrix between the alpha and beta coordinate systems for vectorized variables 
        u_beta = vec(U_beta) = Tv.vec(u_alpha) 
    
    Tc : sparse matrix of shape (nNodeDOF*nNodesTot, nNodeDOF*nNodesTot)
        Transition matrix between the alpha and beta coordinate systems for coupled vectors
    
    Ta : sparse matrix of shape (nNodeDOF*nUncNodes, nNodeDOF*nUncNodes)
        Transition matrix between the alpha and beta coordinate systems for uncoupled vectors

    """
    
    # Primary->Secondary transfer f_beta = f_alpha*T scalar
    T = (1/4)*np.array([[1,1,1,1],[-2,2,-2,2],[2,2,-2,-2],[-4,4,4,-4]]).T
    # with vectorized variables: u_beta = vec(U_beta) = Tv.vec(u_alpha) with Tv vector
    Tv = sp.sparse.kron(sp.sparse.eye(3), T)
    # with all coupled Nodes
    Tc = sp.sparse.kron(Tv.T, sp.sparse.eye(nNodesTot)).tocsr()
    # with all uncoupled Nodes
    Ta = sp.sparse.kron(Tv.T, sp.sparse.eye(nUncNodes)).tocsr() #nUncNodes
    #Ta = sp.sparse.kron(sp.sparse.eye(nUncNodes), Tv) #nUncNodes

    return T, Tv, Tc, Ta

def local2global_matrices(Xunc, uncElems, elemOrder):
    """
    Function which returns the evaluation of the base vectors at each decoupled node
    and the matrix for passing from local to global coordinates
    
    Parameters
    ----------
    Xunc : array of size (nUncNodes, nCoord)
        Coordinates of each uncoupled nodes in the global reference (t,n,b).
    uncElems : array of size (nElemTot, elemOrder + 1)
        Index of the pair of nodes forming an uncoupled element.
    elemOrder : int, optional
        Element order in the FEM discretization. 
        Options are 1 for two-nodes P1 element and 2 for three-nodes P2 element

    Returns
    -------
    t : array of size (nUncNodes, nCoord)
        Coordinates of the local tangent vector evaluated at each uncoupled node.
    n : array of size (nUncNodes, nCoord)
        Coordinates of the local horizontal normal vector evaluated at each uncoupled node.
    b : array of size (nUncNodes, nCoord)
        Coordinates of the local vertical normal vector evaluated at each uncoupled node.
    P : sparse matrix of shape (nNodeDOF*nUncNodes, nNodeDOF*nUncNodes) = (nDOF, nDOF)
        Matrix that transfers variables expressed in a local coordinates to global coordinates
    """
        
    nElems = uncElems.shape[0]
    nNodesByElem = uncElems.shape[1]
    nUncNodes = nElems * nNodesByElem
    
    # node reference coordinates
    if elemOrder == 1 :
        xiN = np.array([0, 1])  
    elif elemOrder == 2 : 
        xiN = np.array([0, .5, 1])
    
    # Coordinate derivatives at element nodes reference coordinates
    XIN = np.tile(xiN, (nElems,))
    elemN = np.repeat(np.arange(nElems), len(xiN))
    ii = np.tile(np.arange(nUncNodes)[:, np.newaxis], (1, nNodesByElem))  # row indices
    jj = uncElems[elemN, :]  # column indices
    gg = shape_function_derivative(XIN, elemOrder)  # fill values
    Dxitan = sp.sparse.csr_matrix((gg.flatten(), (ii.flatten(), jj.flatten())))
    
    # tangent vector
    t = Dxitan @ Xunc
    t = t / np.sqrt(np.sum(t ** 2, 1))[:, np.newaxis]  # normalization
    
    # normal vector (nQP,nCoord)
    n = t @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])  
    
    # third vector (nQP,nCoord)
    b = np.cross(t, n)  

    if (np.sum(t, axis=0)[1:3] == np.zeros((1,2))).all() : #short cut for rectilinear thinwall
        P = sp.sparse.eye(12*nUncNodes, format="csr")
    else :
        # P is a matrix made of (3x3) diagonal blocks of size 12*nQP
        P = np.hstack((t, n, b))  # all frames (nUncNodes,3*3)
        P = np.tile(P, (4, 1))  # replicate for {TL,TR,BL,BR} (4*nUncNodes,3*3)
        P = np.split(P, 9, 1)  # ((4*nUncNodes),3*3)
        P = [sp.sparse.diags(p.flatten()) for p in P]  # ((4*nUncNodes,4*nUncNodes),3*3)
        P = sp.sparse.bmat(np.array(P).reshape((3, 3)).T).tocsr()  # (12*nUncNodes,12*nUncNodes)

    return t, n, b, P

def elem2node(nNodes, nLayers_h, nLayers_v):
    """
    Function that returns the element to node transition matrix.

    Parameters
    ----------
    nNodes : int
        Number of nodes per layer.
    nLayers : int
        Number of layer in the structure.

    Returns
    -------
    elem2node : sparse matrix of shape (nElemTOT, nNodesTOT)
        Transition matrix between the element and the nodes

    """
    nLayers = nLayers_h * nLayers_v
    
    nn = np.ones(2*(nNodes-2) + 2)
    nn[1:-1] = 0.5
    ii = np.arange(0, nNodes-1)[:,np.newaxis] * [1,1]
    elem2node = sp.sparse.csr_matrix(( nn , (ii.flatten(), ii.flatten() + np.array([0,1]*(nNodes-1)))))
    elem2node = sp.sparse.kron(sp.sparse.eye(nLayers), elem2node)
    return elem2node
