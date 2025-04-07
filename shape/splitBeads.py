""" Module implementing split beads to model overhang due to offset"""

# %% import packages

import numpy as np
import scipy as sp
from modules import mesh
from shape import shapeOffset


# %% Meshing


def stacking_offset_split_beads(meshing, offset):  # takes meshing and offset, not meshing_split
    """
        modify stacking offset to shift split beads to their respective positions, knowing their width depends on overhang
        offset_new is only layer dependant while offset_split is specific to beads
    """
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nNodes_h = nNodes * (nLayers_h)

    offset_n = offset[1, ::nNodes_h]
    overhang_n = offset_n[1:] - offset_n[:-1]
    overhang_n = np.hstack((-overhang_n[0], overhang_n))
    final_offset_n = np.repeat(offset_n[:, np.newaxis], nLayers_h + 1, axis=1)
    offset_new = np.vstack((np.hstack((offset[0], [0] * (nNodes * (nLayers_v)))), np.repeat(final_offset_n, nNodes)))  # increase size of empty offset[0] to match new shape with split beads (layerwise offset, not beadwise)

    mask = overhang_n > 0

    final_offset_n[mask, -1:] -= 1 / 2
    final_offset_n[mask, -2:] -= np.repeat((np.abs(overhang_n[mask]) / 2)[:, np.newaxis], 2, axis=1)

    final_offset_n[~mask] -= 1  # rattraper le décalage des cordons split pour le décalage négatif
    final_offset_n[~mask, :1] += 1 / 2
    final_offset_n[~mask, :2] += np.repeat((np.abs(overhang_n[~mask]) / 2)[:, np.newaxis], 2, axis=1)

    offset_split = np.vstack((np.hstack((offset[0], [0] * (nNodes * (nLayers_v)))), np.repeat(final_offset_n, nNodes)))  # increase size of empty offset[0] to match new shape with split beads

    return offset_new, offset_split, overhang_n


def mesh_offset_split(X_split, Elems_split, meshing_split, offset_split, zigzag=False):
    X_split = shapeOffset.mesh_offset(meshing_split, offset_split, X_split, Elems_split)
    return X_split


# %% Behavior (pas du tout réalisé pour des Hn variables)

def assembly_behavior_xi(Hn, Hb, N, Ds, lmbda, mu, W, offset):
    '''
    Merge of behavior.derivation_xi (Btot) and behavior.homogeneization_Rxi (Rxi) to return behavior.assembly_behavior(Rxi, Btot, W)
    Other goal is to avoid 1//Hn factors to prevent conditionning issues for very small Hn values. To this end, Hn*Btot and Rxi/Hn are computed instead of Btot and Rxi. 
    '''
    ## Btot*Hn
    # Initialisation of arrays of size (6,12)
    B0xHn, B1xHn = np.zeros((6, 12)), np.zeros((6, 12))

    # Fill non zero values
    B0xHn[1, 5], B0xHn[2, 10], B0xHn[3, 1], B0xHn[4, 2], B0xHn[5, 6], B0xHn[5, 9] = 1, Hn / Hb, 0.5, 0.5 * Hn / Hb, 0.5 * Hn / Hb, 0.5
    B1xHn[0, 0], B1xHn[3, 4], B1xHn[4, 8] = 1, 0.5, 0.5
    # Assembling (6*nQP, nUncDOF=12*nQP)
    BtotxHn = sp.sparse.kron(B0xHn, N) + sp.sparse.kron(Hn * B1xHn, Ds)

    ## Rxi/Hn
    Rxi_Hn = Hb * sp.linalg.block_diag(lmbda * np.ones((3, 3)) + 2 * mu * np.eye(3), 4 * mu * np.eye(3))  # (6x6)
    Rxi_Hn = sp.sparse.csr_matrix(Rxi_Hn)

    K_xi = 1 / Hn * (BtotxHn.T @ sp.sparse.kron(Rxi_Hn, W) @ BtotxHn)  # factorisation par 1/Hn, #TODO: à contrôler si Hn est trop petit

    return K_xi


def assembly_behavior_chi(Hn, Hb, N, Ds, lmbda, mu, L, nNodes, W, offset):
    '''
    Merge of behavior.derivation_chi (Ctot) and behavior.homogeneization_Rchi (Rchi) to return behavior.assembly_behavior(Rchi, Ctot, W)
    Other goal is to avoid 1//Hn factors to prevent conditionning issues for very small Hn values. To this end, Hn*Ctot and Rchi/Hn are computed instead of Ctot and Rchi.
    '''
    ## Ctot*Hn
    # Initialisation of arrays of size (9,12)
    C0xHn, C1xHn = np.zeros((9, 12)), np.zeros((9, 12))  # (9x12)

    # Fill non zero values
    C0xHn[6, 3], C0xHn[7, 7], C0xHn[8, 11] = 1 / Hb, 1 / Hb, 1 / Hb
    C1xHn[0, 1], C1xHn[1, 5], C1xHn[2, 9], C1xHn[3, 2], C1xHn[4, 6], C1xHn[5, 10] = 1, 1, 1, Hn / Hb, Hn / Hb, Hn / Hb
    # Assembling
    CtotxHn = sp.sparse.kron(C0xHn, N) + sp.sparse.kron(C1xHn, Ds)  # (9*nQP, nUncDOF=12*nQP)

    ##Rchi/Hn
    ln = Hn / np.sqrt(12)
    lb = Hb / np.sqrt(12)
    lt = L / (np.sqrt(12) * (nNodes - 1))

    Rchi_t_Hn = mu * np.array(
        [[1., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 2., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 1., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 2., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    Rchi_t_Hn += lmbda * np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 1., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    Rchi_t_Hn *= Hb * lt ** 2              # Hb * L² / (12*(nNodes - 1)²)
    Rchi_n_Hn = mu * np.array(
        [[2., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 2.]])
    Rchi_n_Hn += lmbda * np.array(
        [[1., 0., 0., 0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0., 0., 0., 1.]])
    Rchi_n_Hn *= Hb * ln ** 2              # Hb * Hn² / 12
    Rchi_b_Hn = mu * np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 2., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 1., 0., 0.], [0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 1., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 2., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    Rchi_b_Hn += lmbda * np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    Rchi_b_Hn *= Hb * lb ** 2              # Hb * Hb² / 12
    Rchi_Hn = Rchi_t_Hn + Rchi_n_Hn + Rchi_b_Hn  # (9,9)

    Rchi_Hn = sp.sparse.csr_matrix(Rchi_Hn)

    K_chi = 1 / Hn * (CtotxHn.T @ sp.sparse.kron(Rchi_Hn, W) @ CtotxHn)  # factorisation par 1/Hn, #TODO: à contrôler si Hn est trop petit

    return K_chi


# %% Welding

def generate_extendedData(meshing_split, weldDof, overhang_n):
    '''
        weldDof [1,2,3,4] + connection type [5] + split leader [5] + split follower [6] + positive shift leader [7] + positive shift follower [8]

        '''
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing_split
    nNodes_h = nLayers_h * nNodes  # split configuration

    mask_pos = overhang_n > 0
    # Indices of layers with positive overhang
    ind_v_pos = np.arange(nLayers_v)[mask_pos]
    # Indices of nodes in split beads with positive overhang
    split_nodes_pos = np.repeat(((ind_v_pos + 1) * nNodes_h - nNodes)[:, np.newaxis], nNodes, axis=1) + np.repeat(np.arange(nNodes)[:, np.newaxis], np.sum(mask_pos), axis=1).T
    # Indices of layers with negative overhang
    ind_v_neg = np.arange(nLayers_v)[~mask_pos]
    # Indices of nodes in split beads with negative overhang
    split_nodes_neg = np.repeat((ind_v_neg * nNodes_h)[:, np.newaxis], nNodes, axis=1) + np.repeat(np.arange(nNodes)[:, np.newaxis], np.sum(~mask_pos), axis=1).T

    # Combine positive and negative split nodes
    split_nodes = np.vstack((split_nodes_pos, split_nodes_neg))

    # Mask for leader nodes that are split beads
    mask_split_nodes_leader = np.isin(weldDof[:, 0], split_nodes)
    # Mask for follower nodes that are split beads
    mask_split_nodes_follower = np.isin(weldDof[:, 2], split_nodes)
    # Combined mask for split nodes
    mask_split_nodes = np.logical_or(mask_split_nodes_leader, mask_split_nodes_follower)

    # Mask for nodes in layers shifted right (positive overhang)
    mask_leader_pos = mask_pos[weldDof[:, 0] // nNodes_h]
    mask_follower_pos = mask_pos[weldDof[:, 2] // nNodes_h]

    # Get connection type using the welding_type function
    weldDof_connectionType = shapeOffset.welding_type(weldDof)

    # Combine all extended data
    weldDof_extendedData = np.hstack((weldDof_connectionType, mask_split_nodes_leader[:, np.newaxis], mask_split_nodes_follower[:, np.newaxis], mask_leader_pos[:, np.newaxis], mask_follower_pos[:, np.newaxis]))
    return weldDof_extendedData


def get_split_edges(meshing_split, overhang_n):
    ''' return split edges ie node/particle of split beads'''
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing_split
    nNodes_h = nLayers_h * nNodes  # split configuration

    mask_pos = overhang_n > 0

    # Indices of layers with positive overhang
    ind_v_pos = np.arange(nLayers_v)[mask_pos]
    # Indices of nodes in split beads with positive overhang
    split_nodes_pos = np.repeat(((ind_v_pos + 1) * nNodes_h - nNodes)[:, np.newaxis], nNodes, axis=1) + np.repeat(np.arange(nNodes)[:, np.newaxis], np.sum(mask_pos), axis=1).T
    # Indices of layers with negative overhang
    ind_v_neg = np.arange(nLayers_v)[~mask_pos]
    # Indices of nodes in split beads with negative overhang
    split_nodes_neg = np.repeat((ind_v_neg * nNodes_h)[:, np.newaxis], nNodes, axis=1) + np.repeat(np.arange(nNodes)[:, np.newaxis], np.sum(~mask_pos), axis=1).T

    new_split_pos = np.vstack((np.repeat(split_nodes_pos, 2, axis=1).flatten(), np.repeat(np.repeat(np.array([[0, 2]]), nNodes, axis=0), len(split_nodes_pos), axis=0).flatten())).T           # new split node : follower
    old_split_pos = np.vstack((np.repeat(split_nodes_pos - nNodes, 2, axis=1).flatten(), np.repeat(np.repeat(np.array([[1, 3]]), nNodes, axis=0), len(split_nodes_pos), axis=0).flatten())).T  # previously unsplit node : leader

    new_split_neg = np.vstack((np.repeat(split_nodes_neg, 2, axis=1).flatten(), np.repeat(np.repeat(np.array([[1, 3]]), nNodes, axis=0), len(split_nodes_neg), axis=0).flatten())).T           # new split node : leader
    old_split_neg = np.vstack((np.repeat(split_nodes_neg + nNodes, 2, axis=1).flatten(), np.repeat(np.repeat(np.array([[0, 2]]), nNodes, axis=0), len(split_nodes_neg), axis=0).flatten())).T  # previously unsplit node : follower

    return new_split_pos, old_split_pos, new_split_neg, old_split_neg


def weld_split_beads(meshing_split, weldDof, overhang_n):
    '''hypothese : zigzag=False ie premier cordon à gauche (vers les xn négatifs) et dernier à droite (vers les xn positifs)

    '''
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing_split  # split configuration
    nNodes_h = nLayers_h * nNodes

    weldDof_extendedData = generate_extendedData(meshing_split, weldDof, overhang_n)

    # shift leader node to account for shifted layers in opposite directions
    weldDof_shift = weldDof.copy()
    mask_shift_leader_plus_nNodes = np.logical_and(weldDof_extendedData[:, 8], weldDof_extendedData[:, 8] != weldDof_extendedData[:, 7])  # follower à droite et leader à gauche
    mask_shift_leader_minus_nNodes = np.logical_and(weldDof_extendedData[:, 7], weldDof_extendedData[:, 8] != weldDof_extendedData[:, 7])  # leader à droite et follower à gauche
    weldDof_shift[mask_shift_leader_plus_nNodes, 0] += nNodes  # shift leader node to account for shifted layer
    weldDof_shift[mask_shift_leader_minus_nNodes, 0] -= nNodes  # shift leader node to account for shifted layer

    weldDof_shift_extendedData = generate_extendedData(meshing_split, weldDof_shift, overhang_n)

    # remove vertical connections for split nodes
    mask_split_nodes_vertical_follower = np.logical_and(weldDof_shift_extendedData[:, 4], weldDof_shift_extendedData[:, 6])  # connection is vertical and leader is a split node
    weldDof_consoles = weldDof_shift[~mask_split_nodes_vertical_follower]

    def shift_leader_irrelevant_split_beads(weldDof_consoles, overhang_n):
        mask_pos = overhang_n > 0

        A = np.eye(len(overhang_n))
        np.fill_diagonal(A[1:, :-1], 1)
        mask_same_direction_as_previous_layer = A @ mask_pos != 1
        mask_pos_same_direction_as_previous_layer = np.logical_and(mask_pos, mask_same_direction_as_previous_layer)
        mask_neg_same_direction_as_previous_layer = np.logical_and(~mask_pos, mask_same_direction_as_previous_layer)

        # Indices of layers with positive overhang and same_direction_as_previous_layer
        ind_v_pos_same_direction_as_previous_layer = np.arange(nLayers_v)[mask_pos_same_direction_as_previous_layer]

        # Indices of nodes in split beads with positive overhang and same_direction_as_previous_layer
        split_nodes_pos_same_direction_as_previous_layer = np.repeat(((ind_v_pos_same_direction_as_previous_layer + 1) * nNodes_h - 2 * nNodes)[:, np.newaxis], nNodes, axis=1) + np.repeat(np.arange(nNodes)[:, np.newaxis],
                                                                                                                                                                                            np.sum(mask_pos_same_direction_as_previous_layer),
                                                                                                                                                                                            axis=1).T
        # Indices of layers with negative overhang and same_direction_as_previous_layer
        ind_v_neg_same_direction_as_previous_layer = np.arange(nLayers_v)[mask_neg_same_direction_as_previous_layer]
        # Indices of nodes in split beads with negative overhang and same_direction_as_previous_layer
        split_nodes_neg_same_direction_as_previous_layer = np.repeat((ind_v_neg_same_direction_as_previous_layer * nNodes_h + nNodes)[:, np.newaxis], nNodes, axis=1) + np.repeat(np.arange(nNodes)[:, np.newaxis],
                                                                                                                                                                                  np.sum(mask_neg_same_direction_as_previous_layer), axis=1).T

        v_con_13 = np.sum(weldDof_consoles[:, [1, 3]] == [1, 3], axis=1) == 2  # vertical connection between particles 1 and 3
        v_con_02 = np.sum(weldDof_consoles[:, [1, 3]] == [0, 2], axis=1) == 2  # vertical connection between particles 0 and 2
        weldDof_consoles[np.logical_and(np.isin(weldDof_consoles[:, 2], split_nodes_pos_same_direction_as_previous_layer), v_con_13), 0] += nNodes  # leader node is shifted to next bead
        weldDof_consoles[np.logical_and(np.isin(weldDof_consoles[:, 2], split_nodes_neg_same_direction_as_previous_layer), v_con_02), 0] -= nNodes  # leader node is shifted to previous bead

        return weldDof_consoles

    weldDof = shift_leader_irrelevant_split_beads(weldDof_consoles, overhang_n)

    return weldDof

def extended_split_weldDof(meshing_split, extended_weldDof, overhang_n) :
    ''' convert weldDof from offset configuration to splitbeads by :
        1. converting node numbers to extra split node
        2. adding horizontal connections between split beads
        3. adding linear interpolation constraint to split edge top particle
        4. switching vertical connection from last top particle to new split bead top particle
        '''
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing_split
    nNodes_h = nLayers_h * nNodes
    nNodes_h_unsplit = nNodes_h - nNodes
    nNodesTot = nNodes * nLayers_h * nLayers_v
    nNodesTot_unsplit = nNodesTot - nNodes * nLayers_v
    mask_pos = overhang_n > 0
    new_split_pos, old_split_pos, new_split_neg, old_split_neg = get_split_edges(meshing_split, overhang_n)
    extended_weldDof_split = extended_weldDof[extended_weldDof[:, 4] != 0].copy()  # not considering null weights

    # %% 1. converting node number to split configuration : add nNodes for every layer, current layer only if overhang <0
    convert = np.arange(nNodesTot_unsplit) + (np.arange(nNodesTot_unsplit) // nNodes_h_unsplit + np.repeat(~mask_pos * 1, nNodes_h_unsplit)) * nNodes
    extended_weldDof_split[:, 0], extended_weldDof_split[:, 2] = convert[extended_weldDof_split[:, 0].astype(int)], convert[extended_weldDof_split[:, 2].astype(int)]

    # %% 2. adding horizontal connections between split beads (wrong leader/follower relationship but needed this way for 4.)
    weldDof_h = np.vstack((np.hstack((old_split_pos, new_split_pos)), np.hstack((new_split_neg, old_split_neg))))
    extended_weldDof_h = np.hstack((weldDof_h, np.ones((weldDof_h.shape[0], 1), dtype=int)))

    # %% 3. switching vertical connection from last top particle to new split bead top particle
    # top split edge cannot lead with weight !=1 -> partial leaders switch to new split node
    mask_to_switch_pos = np.logical_and(np.isin(extended_weldDof_split[:, 0], old_split_pos[:, 0]), np.isin(extended_weldDof_split[:, 1], old_split_pos[:, 1]))
    mask_to_switch_neg = np.logical_and(np.isin(extended_weldDof_split[:, 0], old_split_neg[:, 0]), np.isin(extended_weldDof_split[:, 1], old_split_neg[:, 1]))
    extended_weldDof_split[mask_to_switch_pos, 0] += nNodes
    extended_weldDof_split[mask_to_switch_neg, 0] -= nNodes

    # comment out extended_weldDof_interpolation in result because
    # 1. particles end up following too many particles ?? #TODO check if this leads to issues in Sw
    # 2. such kinematic contraints seem unnecessary and would over rigidify the system #TODO check if this leads to a more rigid structure
    # %% 4. adding linear interpolation constraint to split edge top particle #TODO: fix this and check relevance
    weldDof_interpolation_1 = weldDof_h[weldDof_h[:, 1] == 1]  # top particle
    weldDof_interpolation_0 = weldDof_h[weldDof_h[:, 1] == 0]  # top particle
    extended_weldDof_interpolation_1 = np.hstack((np.vstack(
        (np.hstack((weldDof_interpolation_1[:, [2, 3]] + [0, 1], weldDof_interpolation_1[:, [0, 1]])), np.hstack((weldDof_interpolation_1[:, [0, 1]] - [0, 1], weldDof_interpolation_1[:, [0, 1]])))),
                                                  np.zeros((weldDof_interpolation_1.shape[0] * 2, 1))))
    extended_weldDof_interpolation_0 = np.hstack((np.vstack(
        (np.hstack((weldDof_interpolation_0[:, [2, 3]] + [0, 1], weldDof_interpolation_0[:, [0, 1]])), np.hstack((weldDof_interpolation_0[:, [0, 1]] - [0, 1], weldDof_interpolation_0[:, [0, 1]])))),
                                                  np.zeros((weldDof_interpolation_0.shape[0] * 2, 1))))
    extended_weldDof_interpolation = np.vstack((extended_weldDof_interpolation_1, extended_weldDof_interpolation_0))

    Delta = overhang_n[extended_weldDof_interpolation[:, 2].astype(int) // nNodes_h]
    weights = np.vstack((1 - np.abs(Delta), np.abs(Delta)))
    extended_weldDof_interpolation[:, 4] = 1 * (Delta > 0) * (weights[0] * extended_weldDof_interpolation[:, 1] + weights[1] * (1 - extended_weldDof_interpolation[:, 1])) + 1 * (Delta < 0) * (
                weights[1] * extended_weldDof_interpolation[:, 1] + weights[0] * (1 - extended_weldDof_interpolation[:, 1]))

    # %% 2. adding horizontal connections between split beads (proper leader/follower relationship now that 4. is over)
    weldDof_h = np.vstack((np.hstack((old_split_pos, new_split_pos)), np.hstack((old_split_neg, new_split_neg))))
    extended_weldDof_h = np.hstack((weldDof_h, np.ones((weldDof_h.shape[0], 1), dtype=int)))

    #extended_weldDof = np.vstack((extended_weldDof_split, extended_weldDof_h, extended_weldDof_interpolation)) #TODO: %% 4. is commented out is extended_weldDof_interpolation
    extended_weldDof = np.vstack((extended_weldDof_split, extended_weldDof_h))

    return extended_weldDof

def offset_split_weldwire_matrix(meshing, extended_weldDof_split, Sw):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nNodesTot = nNodes * nLayers_h * nLayers_v

    if nLayers_v == 1:
        return Sw

    from shape.shapeOffset import welding_type, make_primary_leaders, update_weldwire_matrix_weights
    extended_v_mask = welding_type(extended_weldDof_split)[:, -1].astype(bool)  # adds column with 0 for horizontal connection and 1 for vertical connection
    primary_weldDof = make_primary_leaders(extended_weldDof_split, Sw)  # shift all follower particles to their primary leader
    Sw_update = update_weldwire_matrix_weights(nNodesTot, primary_weldDof[extended_v_mask], Sw)  # only vertical connections need to be updated

    return Sw_update