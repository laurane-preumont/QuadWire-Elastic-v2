"""Module defining the mechanical behaviour of the QuadWire model"""

#%% import packages
import numpy as np
import scipy as sp
#%% Comportement

def derivation_xi(Hn, Hb, N, Ds): #strain matrix ? B_chi B_xi
    """
    Function that returns the first gradient derivation matrix 

    Parameters
    ----------
    Hn : float
        Length of the section of the bead in the n direction (width).
    Hb : float
        Length of the section of the bead in the b direction (height).
    N : sparse matrix of shape (nQP, nUncNodes) 
        Function of interpolation at the quadrature points : f(Xqp) = N.f(Xunc)
    Ds : sparse matrix of shape (nQP, nUncNodes)
        Function derivation in LENGTH coordinates : df_ds(Xqp) = Ds.f(Xunc).
    Returns
    -------
    Btot : sparse matrix of shape (6*nQP, nNodeDOF*nQP)
        First gradient derivation matrix.

    """
    # Initialisation of arrays of size (6,12) 
    B0, B1 = np.zeros((6, 12)), np.zeros((6, 12)) 
    
    # Fill non zero values
    B0[1, 5], B0[2, 10], B0[3, 1], B0[4, 2], B0[5, 6], B0[5, 9] = 1/Hn, 1/Hb, 0.5/Hn, 0.5/Hb, 0.5/Hb, 0.5/Hn
    B1[0, 0], B1[3, 4], B1[4, 8] = 1, 0.5, 0.5
    # Assembling (6*nQP, nUncDOF=12*nQP)
    Btot = sp.sparse.kron(B0, N) + sp.sparse.kron(B1, Ds)   

    return Btot

def derivation_chi(Hn, Hb, N, Ds):
    """
    Function that returns the second gradient derivation matrix

    Parameters
    ----------
    Hn : float
        Length of the section of the bead in the n direction (width).
    Hb : float
        Length of the section of the bead in the b direction (height).
    N : sparse matrix of shape (nQP, nUncNodes) 
        Function of interpolation at the quadrature points : f(Xqp) = N.f(Xunc)
    Ds : sparse matrix of shape (nQP, nUncNodes)
        Function derivation in LENGTH coordinates : df_ds(Xqp) = Ds.f(Xunc).

    Returns
    -------
    Ctot : sparse matrix of shape (9*nQP, nNodeDOF*nQP)
        Second gradient derivation matrix.

    """
    # Initialisation of arrays of size (9,12)
    C0, C1 = np.zeros((9, 12)), np.zeros((9, 12))  # (9x12)
    
    # Fill non zero values
    C0[6, 3], C0[7, 7], C0[8, 11] = 1/(Hn*Hb), 1/(Hn*Hb), 1/(Hn*Hb)
    C1[0, 1], C1[1, 5], C1[2, 9], C1[3, 2], C1[4, 6], C1[5, 10] = 1/Hn, 1/Hn, 1/Hn, 1/Hb, 1/Hb, 1/Hb
    # Assembling 
    Ctot = sp.sparse.kron(C0, N) + sp.sparse.kron(C1, Ds)  # (9*nQP, nUncDOF=12*nQP)
    
    return Ctot

def homogeneization_Rxi(Hn, Hb, lmbda, mu):
    """
    Function that returns the elementary material behavior matrix for deformation obtained through the homogenisation process

    Parameters
    ----------
    Hn : float
        Length of the section of the bead in the n direction.
    Hb : float
        Length of the section of the bead in the b direction.
    lmbda : float
        First Lamé coefficient.
    mu : float
        Second Lamé coefficient or shear modulus.

    Returns
    -------
    Rxi : sparse matrix of shape (6, 6)
        Elementary material behavior matrix for deformation obtained through the homogenisation process.

    """
    Rxi = Hn * Hb * sp.linalg.block_diag(lmbda * np.ones((3, 3)) + 2 * mu * np.eye(3), 4 * mu * np.eye(3))  # (6x6)
    Rxi = sp.sparse.csr_matrix(Rxi)

    return Rxi

def homogenization_Rchi(L, Hn, Hb, lmbda, mu, nNodes):
    """
    Function that returns the elementary material behavior matrix for curvature obtained through the homogenisation process

    Parameters
    ----------
    L : int
        Length of the structure in tangential direction t.
    Hn : float
        Length of the section of the bead in the n direction.
    Hb : float
        Length of the section of the bead in the b direction.
    lmbda : float
        First Lamé coefficient.
    mu : float
        Second Lamé coefficient or shear modulus.
    nNodes : int
        Number of nodes discretising the length L.

    Returns
    -------
    Rchi : sparse matrix of shape (9, 9)
        Elementary material behavior matrix for curvature obtained through the homogenisation process.
    """
    
    ln = Hn / np.sqrt(12)
    lb = Hb / np.sqrt(12)
    lt = L / (np.sqrt(12)*(nNodes - 1))
    
    Rchi_t = mu * np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 2., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 1., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 1., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 2., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    Rchi_t += lmbda * np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0., 1., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    Rchi_t *= Hn * Hb * lt ** 2              # Hn * Hb * L² / (12*(nNodes - 1)²)
    Rchi_n = mu * np.array([[2., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 2.]])
    Rchi_n += lmbda * np.array([[1., 0., 0., 0., 0., 0., 0., 0., 1.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [1., 0., 0., 0., 0., 0., 0., 0., 1.]])
    Rchi_n *= Hn * Hb * ln ** 2              # Hn * Hb * Hn² / 12
    Rchi_b = mu * np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 2., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 1., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 1., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 2., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    Rchi_b += lmbda * np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 1., 0., 0., 0., 1., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 1., 0., 0., 0., 1., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    Rchi_b *= Hn * Hb * lb ** 2              # Hn * Hb * Hb² / 12
    Rchi = Rchi_t + Rchi_n + Rchi_b  # (9,9)
    
    Rchi = sp.sparse.csr_matrix(Rchi)

    return Rchi

def optimization_Rxi_Rchi(material="PLA"):
    """
    Function that returns the behavior obtained through the optimization process for PLA

    Parameters
    ----------
    material : str
        Name of the studied material.
        For now "PLA" is the only option.
        
    Returns
    -------
    Rchi : sparse matrix of shape (6, 6)
        Elemental material behavior matrix for deformation obtained through an optimization process.
    Rchi : sparse matrix of shape (9, 9)
        Elemental material behavior matrix for curvature obtained through an optimization process.

    """
    if material == "PLA" :
        Rxi=np.array([[40.38459531, 17.30755737, 17.30755737,  0. ,   0.  ,  0.], 
                      [17.30755737, 40.38456331, 17.30758124,  0. ,   0.  ,  0.],
                      [17.30755737, 17.30758124, 40.38456331,  0. ,   0.  ,  0.],
                      [ 0. ,    0.  ,  0.  ,   46.15384384, 0.          , 0.   ],
                      [ 0. ,    0.  ,  0.  ,    0.        , 46.15384384 , 0.   ],
                      [ 0. ,    0.  ,  0.  ,    0.        , 0.    , 46.15384615]])
        
    
        Rchi = np.array([[0.02590884, 0., 0., 0., 0., 0., 0., 0., 0.00407345],
                          [0., 0.00976768, 0., 0., 0., 0.00027718, 0., 0., 0., ],
                          [0., 0., 0.00939231, 0., 0.00115104, 0., 0.00921195, 0., 0., ],
                          [0., 0., 0., 0.02590884, 0., 0., 0., 0.00407345, 0., ],
                          [0., 0., 0.00115104, 0., 0.00939231, 0., 0.00921195, 0., 0., ],
                          [0., 0.00027718, 0., 0., 0., 0.00976768, 0., 0., 0., ],
                          [0., 0., 0.00921195, 0., 0.00921195, 0., 0.05593367, 0., 0., ],
                          [0., 0., 0., 0.00407345, 0., 0., 0., 0.07302198, 0., ],
                          [0.00407345, 0., 0., 0., 0., 0., 0., 0., 0.07302198]])
    return Rxi, Rchi

def assembly_behavior(R, B, W):
    """
    Function that assembles the elementary behaviour matrix into a global matrix (for the entire mesh).
    It has been contracted with the derivation matrix B so that Sigma = K U

    Parameters
    ----------
    R : sparse matrix of shape (nInc, nInc)
        Elementary material behavior matrix
        nInc can be of different sizes : 6 for Rxi, 9 for Rchi, 3 for Rgamma

    B : sparse matrix of shape (nInc*nQP, nNodeDOF)
       Derivation matrix : first gradient for xi, second gradient for chi, third for gamma      
        
    W : sparse matrix of shape (nQP, nQP)
        Integral weight matrix including local jacobian.

    Returns
    -------
    K : sparse matrix of shape (nDOF, nDOF) nUncDOF = nUncNodes*nNodeDOF
        Assembled behavior matrix.

    """
    K = B.T @ sp.sparse.kron(R, W) @ B  
    return K.tocsr()

#%% Thermal

def dTfcn(x, dT, loadType, nLayers_h, nLayers_v):
    """
    Function which returns the thermal loading at the quadrature points in the alpha configuration.
    This fonction is used to compute the entire structure undergoing a variation of temperature.

    Parameters
    ----------
    x : array of size (nQP, nCoord)
        Coordinates of each quadrature points in the global reference (t,n,b)..
    dT : float
       Temperature variation
    loadType : str
        Type of thermal loading. Options available are "uniform" and "linear".
    nLayers : int
        Number of layers in the printed structure.

    Returns
    -------
    dTalpha array of size (nQP, nParticules)
        Variation of temperature at each quadrature points for each particule.

    """

    nLayers = nLayers_h * nLayers_v
    nQP = x.shape[0]

    if loadType == "uniform" :
        dTalpha = dT * np.ones((x.shape[0], 4))
           
    elif loadType == "linear" :
        #dTalpha = dT*np.linspace(0,1,nNodes)
        #dTalpha = np.tile(dTalpha, nLayers)
        dTalpha = dT*np.linspace(0,1,nQP)
        dTalpha = np.repeat(dTalpha[:,np.newaxis],4, axis=1)  
        
    elif loadType == "random":
        dTalpha = np.random.randint( dT, 0, nQP)
        dTalpha = np.repeat(dTalpha[:,np.newaxis],4, axis=1)
        
    elif loadType == "quad":
        L = x[-1,0]
        dTalpha = dT * (np.ones(nQP) + x[:nQP,0]*2/L)*(np.ones(nQP) - x[:nQP,0]*2/L)
        dTalpha = np.repeat(dTalpha[:,np.newaxis],4, axis=1)

    elif loadType == "poc" :
        nQPlayer = nQP//nLayers  # nQP on given layer
        Ncooling = np.max((1,nQPlayer//2//2)) # takes half of elements to cool down linearly to 1% of dT
        dTalpha = np.zeros((nQP))
        dTalpha[-Ncooling:] = dT*np.linspace(0.01,1,Ncooling)
        dTalpha = np.repeat(dTalpha[:,np.newaxis],4, axis=1)
        
    return dTalpha


def thermal_eigenstrain(alpha, dT):
    """
    Function that returns the thermal eigenstrain.

    Parameters
    ----------
    alpha : float
        Thermal expansion coefficient.
    dT : array of size (nQP, 1)
        Array of the variation of temperature at the quadrature points

    Returns
    -------
    Eps_thermal : sparse matrix of size (nQP*6, 1) 
        Thermal eigenstrain.

    """
    return alpha * sp.sparse.kron(np.array([[1, 1, 1, 0, 0, 0]]).T, dT) 

#%% Energy

def energyDensity(Eps, Eps_th, Sigma, Rtot, quadOrder, nInc):
    """
    Function that computes the free energy density at each quadrature points
    nInc gives the variables on which the energy is computed : 9 for first gradient components, 6 for second gradient components, 15 for both
    
    Parameters
    ----------
    Eps : array of size (nInc*nQP, 1)
        QW generalized strains.
        k*nQP to (k+1)*nQP gives the k^th generalized strain at each quadrature point
    Eps_th : array of size (nInc*nQP, 1)
        Thermal eigenstrain.
    Sigma : array of size (nInc*nQP, 1)
        QW generalized stresses.
        k*nQP to (k+1)*nQP gives the k^th generalized stress at each quadrature point
    Rtot : sparse matrix of shape (nInc*nQP, nInc*nQP)
        Derivation matrix : first gradient for xi, second gradient for chi, third for gamma.
    quadOrder : int
        Element quadrature order.
    nInc : int
        Number of QW degrees of freedom given for Eps.

    Returns
    -------
    nrg : array of size (nQP,)
        Linear elastic energy density expressed at the quadrature points.

    """
    nQP = int(Eps.shape[0]/nInc)
    nrg = 0.5 * np.sum (Eps.reshape((nInc, nQP)) * (Sigma - Rtot @ Eps_th).reshape((nInc, nQP)), axis=0)
    return nrg