""" Module containing functions for tracing internal forces and reconstructing a cauchy stress tensor """
# %% import packages
import numpy as np
import scipy as sp


# %%

def internal_forces(Sigma, Hn, Hb):
    """
    Function that returns the QuadWire internal forces

    Parameters
    ----------
    Sigma : array of size (nInc*nQP, 1)
        QW generalized stresses.
        k*nQP to (k+1)*nQP gives the k^th generalized stress at each quadrature points.
    Hn : float
        Length of the section of the bead in the n direction.
    Hb : float
        Length of the section of the bead in the b direction.

    Returns
    -------
    f1 : array of size (nCoord, nQP, 1)
        Internal force per length unit of particule 1 expressed at the quadrature points.
    f2 : array of size (nCoord, nQP, 1)
        Internal force per length unit of particule 2 expressed at the quadrature points.
    f3 : array of size (nCoord, nQP, 1)
        Internal force per length unit of particule 3 expressed at the quadrature points.
    f4 : array of size (nCoord, nQP, 1)
        Internal force per length unit of particule 4 expressed at the quadrature points.
        
    F1 : array of size (nCoord, nQP, 1)
        Internal force of particule 1 expressed at the quadrature points.
    F2 : array of size (nCoord, nQP, 1)
        Internal force of particule 2 expressed at the quadrature points.
    F3 : array of size (nCoord, nQP, 1)
        Internal force of particule 3 expressed at the quadrature points.
    F4 : array of size (nCoord, nQP, 1)
        Internal force of particule 4 expressed at the quadrature points.


    """
    nQP = int(Sigma.shape[0] / 15)

    s1 = Sigma[nQP * 0:nQP * 1]
    s2 = Sigma[nQP * 1:nQP * 2]
    s3 = Sigma[nQP * 2:nQP * 3]
    s4 = Sigma[nQP * 3:nQP * 4]
    s5 = Sigma[nQP * 4:nQP * 5]
    s6 = Sigma[nQP * 5:nQP * 6]

    Mn_t = Sigma[nQP * 6:nQP * 7]
    Mn_n = Sigma[nQP * 7:nQP * 8]
    Mn_b = Sigma[nQP * 8:nQP * 9]

    Mb_t = Sigma[nQP * 9:nQP * 10]
    Mb_n = Sigma[nQP * 10:nQP * 11]
    Mb_b = Sigma[nQP * 11:nQP * 12]

    mx_t = Sigma[nQP * 12:nQP * 13]
    mx_n = Sigma[nQP * 13:nQP * 14]
    mx_b = Sigma[nQP * 14:nQP * 15]

    F_t = s1
    F_n = s4
    F_b = s5

    mn_t = - s4
    mn_n = - s2
    mn_b = - s6

    mb_t = - s5
    mb_n = - s6
    mb_b = - s3

    f1t = mn_t / (2 * Hn) + mb_t / (2 * Hb) + mx_t / (Hn * Hb)
    f2t = mn_t / (2 * Hn) - mb_t / (2 * Hb) - mx_t / (Hn * Hb)
    f3t = -mn_t / (2 * Hn) + mb_t / (2 * Hb) - mx_t / (Hn * Hb)
    f4t = -mn_t / (2 * Hn) - mb_t / (2 * Hb) + mx_t / (Hn * Hb)

    f1n = mn_n / (2 * Hn) + mb_n / (2 * Hb) + mx_n / (Hn * Hb)
    f2n = mn_n / (2 * Hn) - mb_n / (2 * Hb) - mx_n / (Hn * Hb)
    f3n = -mn_n / (2 * Hn) + mb_n / (2 * Hb) - mx_n / (Hn * Hb)
    f4n = -mn_n / (2 * Hn) - mb_n / (2 * Hb) + mx_n / (Hn * Hb)

    f1b = mn_b / (2 * Hn) + mb_b / (2 * Hb) + mx_b / (Hn * Hb)
    f2b = mn_b / (2 * Hn) - mb_b / (2 * Hb) - mx_b / (Hn * Hb)
    f3b = -mn_b / (2 * Hn) + mb_b / (2 * Hb) - mx_b / (Hn * Hb)
    f4b = -mn_b / (2 * Hn) - mb_b / (2 * Hb) + mx_b / (Hn * Hb)

    F1t = F_t / 4 + Mn_t / (2 * Hn) + Mb_t / (2 * Hb)
    F2t = F_t / 4 + Mn_t / (2 * Hn) - Mb_t / (2 * Hb)
    F3t = F_t / 4 - Mn_t / (2 * Hn) + Mb_t / (2 * Hb)
    F4t = F_t / 4 - Mn_t / (2 * Hn) - Mb_t / (2 * Hb)

    F1n = F_n / 4 + Mn_n / (2 * Hn) + Mb_n / (2 * Hb)
    F2n = F_n / 4 + Mn_n / (2 * Hn) - Mb_n / (2 * Hb)
    F3n = F_n / 4 - Mn_n / (2 * Hn) + Mb_n / (2 * Hb)
    F4n = F_n / 4 - Mn_n / (2 * Hn) - Mb_n / (2 * Hb)

    F1b = F_b / 4 + Mn_b / (2 * Hn) + Mb_b / (2 * Hb)
    F2b = F_b / 4 + Mn_b / (2 * Hn) - Mb_b / (2 * Hb)
    F3b = F_b / 4 - Mn_b / (2 * Hn) + Mb_b / (2 * Hb)
    F4b = F_b / 4 - Mn_b / (2 * Hn) - Mb_b / (2 * Hb)

    # return [f1t, f1n, f1b], [f2t, f2n, f2b], [f3t, f3n, f3b], [f4t, f4n, f4b], [F1t, F1n, F1b], [F2t, F2n, F2b], [F3t, F3n, F3b], [F4t, F4n, F4b]
    return np.array([f1t, f1n, f1b]), np.array([f2t, f2n, f2b]), np.array([f3t, f3n, f3b]), np.array([f4t, f4n, f4b]), np.array([F1t, F1n, F1b]), np.array([F2t, F2n, F2b]), np.array([F3t, F3n, F3b]), np.array([F4t, F4n, F4b])


def sigma3D(Sigma, Hn, Hb):
    """
    Function that reconstructs the 3D cauchy stress tensor

    Parameters
    ----------
    Sigma : array of size (nInc*nQP, 1)
        QW generalized stresses.
        k*nQP to (k+1)*nQP gives the k^th generalized stress at each quadrature points.
    Hn : float
        Length of the section of the bead in the n direction.
    Hb : float
        Length of the section of the bead in the b direction.

    Returns
    -------
    Sigma3D_tt : TYPE
        DESCRIPTION.
    Sigma3D_tn : TYPE
        DESCRIPTION.
    Sigma3D_tb : TYPE
        DESCRIPTION.

    """
    f1, f2, f3, f4, F1, F2, F3, F4 = internal_forces(Sigma, Hn, Hb)

    Sigma3D_tt = (f1[0] + f2[0] + f3[0] + f4[0]) / (Hn * Hb)
    Sigma3D_tn = (f1[1] + f2[1] + f3[1] + f4[1]) / (Hn * Hb)
    Sigma3D_tb = (f1[2] + f2[2] + f3[2] + f4[2]) / (Hn * Hb)

    return Sigma3D_tt, Sigma3D_tn, Sigma3D_tb


def delamination(Sigma, Hn, Hb, nLayers_v, nLayers_h, nNodes, axis):
    '''
    Evaluate delamination forces between beads, be it vertical delamination (axis=2) between successive layers or horizontal delamination (axis=1) between successive beads in a given layer
    Delamination forces are defined as :  (assuming no ZigZag, otherwise would need to consider connection table)
    option 1 : difference of average linear internal forces between successive beads
    option 2 : difference of top/bottom (resp. left/right) linear internal forces between successive beads on top of each other (resp. next to each other)
    Parameters
    ----------
    Sigma :
    Hn :
    Hb :
    nLayers_v :
    nLayers_h :
    nNodes :
    axis :

    Returns
    -------

    '''
    nQP = int(Sigma.shape[0] / 15)
    nLayers = nLayers_v * nLayers_h  # nBeads
    # number of QP per bead and layer
    nQP_bead = nQP // nLayers  # number of QP per bead
    nQP_layer = nQP_bead * nLayers_h  # number of QP per horizontal layer

    f1, f2, f3, f4, F1, F2, F3, F4 = internal_forces(Sigma, Hn, Hb)

    # option 1
    dela = (f1[axis] + f2[axis] + f3[axis] + f4[axis])/4
    dela = dela.reshape(nQP_bead, nLayers_h, nLayers_v)

    if axis == 2:
        delta_dela = dela[1:]  # all QP except first layer
        delta_dela -= dela[:-1]  # all QP except last layer
    if axis == 1:
        delta_dela = dela[:, 1:, :]  # all QP except first bead of each layer
        delta_dela -= dela[:, :-1, :]  # all QP except last bead of each layer

    # option 2
    dela_top = (f1[axis] + f3[axis])/2
    dela_bottom = (f2[axis] + f4[axis])/2
    dela_left = (f4[axis] + f3[axis])/2
    dela_right = (f2[axis] + f2[axis])/2

    if axis == 2:
        delta_dela = dela_bottom[1:]  # all QP except first layer
        delta_dela -= dela_top[:-1]  # all QP except last layer
    if axis == 1:
        delta_dela = dela_left[:, 1:, :]  # all QP except first bead of each layer
        delta_dela -= dela_right[:, :-1, :]  # all QP except last bead of each layer

    return delta_dela


def comp_sigma3D_t(f, Hn, Hb):
    comp_Sigma_3D_t = f / (Hn * Hb)
    return comp_Sigma_3D_t
