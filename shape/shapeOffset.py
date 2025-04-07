""" Module applying stacking offset to a QuadWire FEM geometry"""

# %% import packages
import time

import numpy as np
import scipy as sp
from modules import mesh


# %% Meshing

def stacking_offset(meshing, offsetType="linear", tcoef=0, ncoef=1):
    """
    offset from initial position along t and n. Given as a percentage of lc and Hn. (la dérivée de la fonction doit être inférieure à 1 pour éviter la lévitation)
    Returns offset of size (2, nLayers_v*nNodes_h)
    """
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nNodes_h = nNodes * nLayers_h
    if nLayers_v == 1 :
        return np.zeros((2, nLayers_v*nNodes_h))
    offset = np.ones((2, nLayers_v))
    if offsetType == "linear":
        offset *= np.arange(nLayers_v) / (nLayers_v - 1)
    elif offsetType == "quadratic":
        offset *= (1 / nLayers_v) * np.arange(nLayers_v) ** 2
    elif offsetType == "sinus":
        offset *= L * np.ones(nLayers_v) * np.sin(2 * np.pi / nLayers_v * np.linspace(0,nLayers_v-1,nLayers_v))
    elif offsetType == "gauss":
        mu = nLayers_v * 0.4
        sigma = nLayers_v/4
        x = np.arange(nLayers_v)
        offset *= 2 * L * np.exp(-((x-mu)/sigma)**2)


    offset *= [[tcoef], [ncoef]]
    offset = np.repeat(offset, nNodes_h, axis=1)

    return offset


def mesh_offset(meshing, offset, X, Elems):
    """
    Function that applies a stacking offset between successive layers of the whole structure in the build direction (offset compared to initial X position)

    Parameters
    ----------
    offset : array of size (nLayers_v*nNodes_h, 2)
        offset between successive layers along t and n direction
    X : array of size (nNodes, 3)
        Coordinates of each node in the global reference (t,n,b).

    Returns
    -------
    X : array of size (nNodes*nLayers, 3)
        Coordinates of each node in the global reference (t,n,b).
    """
    L, Hn, _, nLayers_h, nLayers_v, nNodes, _, layerType, _, zigzag = meshing

    if offset is None :
        return X
    ## Apply offset in the building direction
    lc = L / (nNodes - 1)  # length of elements
    dX = offset.T * [lc, Hn]

    if layerType == "normal":
        nNodes_h = nNodes * nLayers_h
        t, n = mesh.local_axis(X, Elems)

        if zigzag:  # if zigzag, t and n change sign at every bead, prevent this by changing signs manually
            signs = np.repeat(np.array([[1, -1]]), nNodes, axis=1).flatten()  # alternating minus signs for two consecutive beads
            signs = np.tile(signs, nLayers_h // 2)  # alternating minus signs for all horizontal beads
            if nLayers_h % 2:  # if number of beads per layer is uneven, add one last row of nNodes positive signs
                signs = np.concatenate((signs, np.ones(nNodes)), axis=0)
            signs = np.tile(signs, nLayers_v)  # alternating minus signs for all vertical beads
            signs.reshape(nLayers_v, nNodes_h)[1::2, :] *= -1  # changing signs for uneven layers since successive layers have different orientations

            t *= signs[:, np.newaxis]
            n *= signs[:, np.newaxis]

        dX = dX[:, 0][:, np.newaxis] * t[:, :-1] + dX[:, 1][:, np.newaxis] * n[:, :-1]

    X[:, :-1] += dX  # add offset
    return X


def generate_trajectory(meshing, offset):
    ''' generate trajectory with given offset '''
    X, Elems, U0 = mesh.generate_trajectory(meshing)
    X = mesh_offset(meshing, offset, X, Elems)
    return X, Elems, U0

# %% Welding

def welding_type(weldDof):
    """
    Analyzes the weld degrees of freedom to determine the type of welding connection between particles : horizontal or vertical.

    Parameters:
    - weldDof: A NumPy array representing the connection table of weld degrees of freedom.
               Each row represents a connection between two particles, with columns representing:
               - Column 0: Leader Node
               - Column 1: Leader particle
               - Column 2: Follower Node
               - Column 3: Follower particle

    Returns:
    - weldDof_connectionType: A modified weldDof array with an additional column indicating
                               whether each entry corresponds to a vertical connection (True/False).
    """
    # Define arrays representing possible vertical and horizontal connections
    vertical_connections = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    horizontal_connections = np.array([[0, 1], [1, 0], [2, 3], [3, 2]])

    # Extract leader and follower columns from weldDof
    weldDof_connections = weldDof[:, [1, 3]]

    # Reshape vertical_connections and weldDof to broadcast against each other
    broadcasted_vc = np.tile(vertical_connections, (len(weldDof_connections), 1, 1))
    broadcasted_wd = np.tile(weldDof_connections[:, np.newaxis, :], (1, len(vertical_connections), 1))

    # Check for equality along the last axis (columns) to identify vertical connections
    is_in_vc = np.any(np.all(broadcasted_vc == broadcasted_wd, axis=-1), axis=1) * 1

    # Add a boolean column indicating vertical connections
    weldDof_connectionType = np.hstack((weldDof, is_in_vc[:, np.newaxis]))

    return weldDof_connectionType


def get_welding_conditions(nNodesTot, Sw):
    '''
    Get welding conditions from welding matrix, so only collecting primary leader particles.
    Return weldDof equivalent to welding_conditions in weld module : colum 0 1 are leader node/particle to follower node/particle in column 2 3
    '''
    wMatrixidx = np.vstack((Sw.nonzero()[0][:4 * nNodesTot], Sw.nonzero()[1][:4 * nNodesTot]))
    diagonal_indices = wMatrixidx[0] == wMatrixidx[1]
    nondiagonal_indices = ~diagonal_indices

    ii_lead = wMatrixidx[0][diagonal_indices]
    jj_lead = ii_lead
    ii_follow = wMatrixidx[0][nondiagonal_indices]
    jj_follow = wMatrixidx[1][nondiagonal_indices]

    lead_nodes = jj_follow % nNodesTot
    lead_particles = jj_follow // nNodesTot
    follow_nodes = ii_follow % nNodesTot
    follow_particles = ii_follow // nNodesTot

    get_weldDof = np.vstack((lead_nodes, lead_particles, follow_nodes, follow_particles)).T

    return get_weldDof


def welding_weights(meshing, offset):
    '''
    Returns weight coefficients to weld particles 2 and 3 to neighbour beads due to offset
    
    Parameters
    ----------
    X : array of size (nNodesTot, 3)
        Coordinates of each node in the global reference (t,n,b).
    offset :  array of size (2, nNodesTot)
        offset between successive layers along t and n direction

    Returns
    -------
    weigths : shape (2, nNodesTot, 3)
        weight coefficients to weld particles to neighbour beads due to offset
        weights[0] are coefficients to t axis and weights[1] are coefficients to n axis
        weights[:,:,0] is weight coefficient to former particle (without offset), 1 is weight coefficient to new node same particle, 2 is same node new particle
    '''
    overhang_max = 1
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nNodes_h = nNodes * nLayers_h

    overhang = np.hstack((np.zeros((2, nNodes_h)), offset[:, nNodes_h:] - offset[:, :-nNodes_h]))
    # cantilever weights
    weights_cantilever = np.stack((1 - np.abs(overhang), np.maximum(0, -overhang), np.maximum(0, overhang)), axis=2)

    # straddle weights
    deltaHn, deltaHn2, deltaHn3 = overhang, np.power(overhang, 2), np.power(overhang, 3)
    # weights for follower particle 3 (ie n°2 in article notation)
    weights_pos_3 = np.stack((1 + deltaHn - 4 * deltaHn2 + 2 * deltaHn3, 2 * deltaHn2 - deltaHn3, -(deltaHn - 2 * deltaHn2 + deltaHn3)), axis=2)
    weights_neg_3 = np.stack((1 + deltaHn - deltaHn2 - deltaHn3, - deltaHn + 2 * deltaHn2 + 2 * deltaHn3, - (deltaHn2 + deltaHn3)), axis=2)
    # weights for follower particle 2 (ie n°4 in article notation)
    weights_pos_2 = np.stack((1 - deltaHn - deltaHn2 + deltaHn3, deltaHn + 2 * deltaHn2 - 2 * deltaHn3, -(deltaHn2 - deltaHn3)), axis=2)
    weights_neg_2 = np.stack((1 - deltaHn - 4 * deltaHn2 - 2 * deltaHn3, deltaHn + 2 * deltaHn2 + deltaHn3, 2 * deltaHn2 + deltaHn3), axis=2)

    bool_pos = np.repeat(((overhang > 0) * 1)[:, :, np.newaxis], 3, axis=2)
    bool_neg = np.repeat(((overhang < 0) * 1)[:, :, np.newaxis], 3, axis=2)
    weights_2 = np.multiply(weights_pos_2, bool_pos) + np.multiply(weights_neg_2, bool_neg)
    weights_3 = np.multiply(weights_pos_3, bool_pos) + np.multiply(weights_neg_3, bool_neg)

    weights_straddle = np.stack((weights_2, weights_3), axis=0)
    if np.any(np.abs(overhang) > overhang_max):  # define overhang_max outside function
        # return 1 / 0  # Error: "overhang cannot exceed bead width"
        print("overhang should not exceed bead width")
        weights_cantilever[:, np.where(np.abs(overhang) > 1)] *= 0
        weights_straddle[:, np.where(np.abs(overhang) > 1)] *= 0

    return weights_cantilever, weights_straddle, overhang


def connection_table_n(meshing, weldDof):
    """
    Analyzes weldDof to return dictionaries containing follower nodes and leader nodes
    to each node of the mesh with type of connection.

    Parameters:
    - weldDof: A NumPy array representing the connection table of weld degrees of freedom.
               Each row represents a connection between two particles, with columns representing:
               - Column 0: Leader Node
               - Column 1: Leader particle
               - Column 2: Follower Node
               - Column 3: Follower particle

    Returns:
    - follower_nodes: A dictionary containing follower nodes for each node. The keys are node IDs,
                      and the values are dictionaries with keys 'horizontal' and 'vertical',
                      containing lists of node IDs connected horizontally and vertically, respectively.
    - leader_nodes: A dictionary containing leader nodes for each node. The keys are node IDs,
                      and the values are dictionaries with keys 'horizontal' and 'vertical',
                      containing lists of node IDs connected horizontally and vertically, respectively.
    """

    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nNodesTot = nNodes * nLayers_h * nLayers_v

    weldDof_connectionType = welding_type(weldDof)  # adds columns with 0 if horizontal connection and 1 for vertical connection

    # Extract leader, follower, and connection type columns
    leaders = weldDof_connectionType[:, 0]
    followers = weldDof_connectionType[:, 2]
    connection_types = weldDof_connectionType[:, 4]

    # Initialize dictionaries to store horizontal and vertical connections
    follower_nodes = {node: {'horizontal': [], 'vertical': []} for node in range(nNodesTot)}
    leader_nodes = {node: {'horizontal': [], 'vertical': []} for node in range(nNodesTot)}

    for leader, follower, connection_type in zip(leaders, followers, connection_types):
        # Add connections based on type
        if connection_type == 0:  # Horizontal connection
            follower_nodes[follower]['horizontal'].append(leader)
            leader_nodes[leader]['horizontal'].append(follower)
        elif connection_type == 1:  # Vertical connection
            follower_nodes[follower]['vertical'].append(leader)
            leader_nodes[leader]['vertical'].append(follower)

    # Convert lists to sets to remove duplicates
    for node in follower_nodes:
        follower_nodes[node]['horizontal'] = list(set(follower_nodes[node]['horizontal']))
        follower_nodes[node]['vertical'] = list(set(follower_nodes[node]['vertical']))
    for node in leader_nodes:
        leader_nodes[node]['horizontal'] = list(set(leader_nodes[node]['horizontal']))
        leader_nodes[node]['vertical'] = list(set(leader_nodes[node]['vertical']))

    return follower_nodes, leader_nodes


def vertical_connections_n(meshing, weldDof):
    """
    Analyzes the weldDof connection table to return dictionaries containing final vertical connections
    to each node where both direct follower vertical connections and indirect connections are considered.

    Parameters:
    - weldDof: A NumPy array representing the connection table of weld degrees of freedom.
               Each row represents a connection between two particles, with columns representing:
               - Column 0: Leader Node
               - Column 1: Leader particle
               - Column 2: Follower Node
               - Column 3: Follower particle

    Returns:
    - vertical_connections: A dictionary containing final vertical connections for each node.
                            The keys are node IDs, and the values are sets of node IDs representing
                            both direct and indirect follower vertical connections.
    """
    # Get follower_nodes and leader_nodes dictionaries from connection_table_n
    follower_nodes, leader_nodes = connection_table_n(meshing, weldDof)
    # Initialize vertical_connections dictionary
    vertical_connections = {node: set() for node in follower_nodes}
    horizontal_connections = {node: set() for node in follower_nodes}

    for node in follower_nodes:
        # Add direct horizontal connections (both follower and leader)
        horizontal_connections[node].update(follower_nodes[node]['horizontal'])
        horizontal_connections[node].update(leader_nodes[node]['horizontal'])

        # Add direct vertical connections
        vertical_connections[node].update(follower_nodes[node]['vertical'])
        # Add indirect vertical connections via intermediate horizontal connections (both leader and follower)
        # # old version (issue with split beads)
        # for intermediate_node in follower_nodes[node]['horizontal'] + leader_nodes[node]['horizontal']:
        # vertical_connections[node].update(follower_nodes[intermediate_node]['vertical'])
        # new version
        direct_vertical_neighbours = np.array(list(vertical_connections.get(node)))
        for vertical_neighbour in direct_vertical_neighbours:
            vertical_connections[node].update(horizontal_connections[vertical_neighbour])

    return vertical_connections


def connectionType(weldDof):
    '''
    Returns mask of 0 for horizontal connection and 1 for vertical connection
    '''
    weldDof_connectionType = welding_type(weldDof)  # adds column with 0 for horizontal connection and 1 for vertical connection
    v_mask = weldDof_connectionType[:, 4].astype(bool)
    return v_mask


## save old version with simple welding weights
# def make_welding_connections_n(meshing, offset_weights, weldDof, v_connections, v_mask):
#     '''
#     sets weights to connections between particles based on offset
#
#     Parameters
#     ----------
#     meshing :
#     provide all mesh parameters to define L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes, meshType
#     offset_weights :
#
#     weldDof :
#     v_connections :
#     v_mask :
#
#     Returns
#     -------
#
#     '''
#     L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
#
#     weldDof_weights_vertical = np.ones((weldDof.shape[0], weldDof.shape[1] + 1), dtype=int)
#     weldDof_weights_vertical[:, :-1] = weldDof  # initialize weights to 1 (no offset)
#     if nLayers_v == 1:  # no offset possible for a single layer
#         return weldDof_weights_vertical
#     v_welds = weldDof_weights_vertical[v_mask].astype(float)  # vertical connections
#     h_welds = weldDof_weights_vertical[~v_mask]  # horizontal connections
#     # new_welds: beads have three vertical neighbours (2 new ones) except first and last beads of layers that only have two vertical neighbours (end particle only has 1 new one)
#     new_welds = np.zeros((v_welds.shape[0] * 2 - 4 * (nLayers_v - 1), 5)).astype(float)
#     new_welds = np.zeros((v_welds.shape[0] * 2, 5)).astype(float)
#
#     new_row = 0
#
#     for row in range(v_welds.shape[0]):
#         node, particle = v_welds[row, [2, 3]].astype(int)  # follower node/particle
#         # if np.isin([node, particle], split_particles)
#         v_con = v_connections.get(node)  # nodes potentially vertically connected to this node as leaders (vertical neighbors)
#         main_leader = {v_welds[row, 0].astype(int)}  # initial leader node
#         new_leader = v_con.difference(main_leader)  # new leader nodes due to offset
#         if len(v_con) == 3:  # if node has 3 vertical neighbors (any bead except first and last of the layer)
#             v_welds[row, 4] = offset_weights[node, 0]  # weight of initial leader node/particle
#             new_welds[new_row, 4] = offset_weights[node, 1]  # weight of new leader node/particle
#             new_welds[new_row + 1, 4] = offset_weights[node, 2]
#             new_welds[new_row, [1, 2, 3]] = v_welds[row, [1, 2, 3]]  # weld particles are all identical
#             new_welds[new_row + 1, [1, 2, 3]] = v_welds[row, [1, 2, 3]]
#             new_welds[new_row, 0] = list(new_leader)[0]  # new leader nodes are updated
#             new_welds[new_row + 1, 0] = list(new_leader)[1]
#             new_row += 2
#         elif len(v_con) == 2:  # if node only has 2 vertical neighbors (first or last bead of the layer)
#             if list(main_leader) > list(new_leader):  # node is last of layer
#                 if particle == 2:
#                     v_welds[row, 4] = offset_weights[node, 0]  # weight of initial leader node
#                     new_welds[new_row, 4] = offset_weights[node, 1]  # weight of new leader node/particle
#                     new_welds[new_row + 1, 4] = offset_weights[node, 2]  # weight of new leader node/particle
#                     new_welds[new_row, [1, 2, 3]] = v_welds[row, [1, 2, 3]]  # weld particles are updated
#                     new_welds[new_row + 1, [1, 2, 3]] = v_welds[row, [1, 2, 3]] + [1, 0, 0]  # weld particles are updated
#                     new_welds[new_row, 0] = list(new_leader)[0]  # new leader nodes are updated
#                     new_welds[new_row + 1, 0] = list(main_leader)[0]
#                     new_row += 2
#                 elif particle == 3:
#                     v_welds[row, 4] = offset_weights[node, 0] + offset_weights[node, 2]  # weight of initial leader node/particle
#                     new_welds[new_row, 4] = offset_weights[node, 1]  # weight of new leader node/particle
#                     new_welds[new_row, [1, 2, 3]] = v_welds[row, [1, 2, 3]]  # weld particles are all identical
#                     new_welds[new_row, 0] = list(new_leader)[0]  # new leader node is updated
#                     new_row += 1
#             if list(main_leader) < list(new_leader):  # node is first of layer
#                 if particle == 2:
#                     v_welds[row, 4] = offset_weights[node, 0] + offset_weights[node, 1]  # weight of initial leader node/particle
#                     new_welds[new_row, 4] = offset_weights[node, 2]  # weight of new leader node/particle
#                     new_welds[new_row, [1, 2, 3]] = v_welds[row, [1, 2, 3]]  # weld particles are all identical
#                     new_welds[new_row, 0] = list(new_leader)[0]  # new leader node is updated
#                     new_row += 1
#                 elif particle == 3:
#                     v_welds[row, 4] = offset_weights[node, 0]  # weight of initial leader node
#                     new_welds[new_row, 4] = offset_weights[node, 1]  # weight of new leader node/particle
#                     new_welds[new_row + 1, 4] = offset_weights[node, 2]  # weight of new leader node/particle
#                     new_welds[new_row, [1, 2, 3]] = v_welds[row, [1, 2, 3]] - [1, 0, 0]  # weld particles are updated
#                     new_welds[new_row + 1, [1, 2, 3]] = v_welds[row, [1, 2, 3]]  # weld particles are updated
#                     new_welds[new_row, 0] = next(iter(main_leader)) # new leader nodes are updated
#                     new_welds[new_row + 1, 0] = list(new_leader)[0]
#                     new_row += 2
#         elif len(v_con) == 1:  # node only has 1 vertical neighbor (thinwall)
#             if particle == 2:
#                 v_welds[row, 4] = offset_weights[node, 0] + offset_weights[node, 1]  # weight of initial leader node/particle
#                 new_welds[new_row, 4] = offset_weights[node, 2]  # weight of new leader node/particle
#                 new_welds[new_row, [0, 1, 2, 3]] = v_welds[row, [0, 1, 2, 3]] + [0, 1, 0, 0]  # new leader node/particle
#             elif particle == 3:
#                 v_welds[row, 4] = offset_weights[node, 0] + offset_weights[node, 2]  # weight of initial leader node/particle
#                 new_welds[new_row, 4] = offset_weights[node, 1]  # weight of new leader node/particle
#                 new_welds[new_row, [0, 1, 2, 3]] = v_welds[row, [0, 1, 2, 3]] + [0, -1, 0, 0]  # new leader node/particle
#             new_row += 1
#
#         # TODO: if zigzag, layers alternate ascending and descending nodes along n direction (fix "new leader nodes are updated")
#     new_weldDof_weights_vertical = np.vstack((h_welds, v_welds, new_welds))
#     # command to extract results for node 4 and particule 2 : new_weldDof_weights[(new_weldDof_weights[:,2]==4) * (new_weldDof_weights[:,3]==2)]
#     return new_weldDof_weights_vertical

def make_welding_connections_n(meshing, offset_weights, weldDof, v_connections, v_mask):
    '''
    sets weights to connections between particles based on offset

    Parameters
    ----------
    meshing :
    provide all mesh parameters to define L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes, meshType
    offset_weights :

    weldDof : array of size (nNodes*(nLayers-1)*2, 4)
        A NumPy array representing the connection table of weld degrees of freedom.
        Each row represents a connection between two particles, with columns representing:
           - Column 0: Leader Node
           - Column 1: Leader particle
           - Column 2: Follower Node
           - Column 3: Follower particle
    v_connections :
    v_mask :

    Returns
    -------

    '''
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    weights_cantilever, weights_straddle, overhang = offset_weights
    weights_cantilever = weights_cantilever[1]
    weights_2, weights_3, overhang = weights_straddle[0][1], weights_straddle[1][1], overhang[1]  # assuming overhang[0] is t and overhang[1] is n

    if nLayers_v == 1:  # no offset possible for a single layer
        return weldDof

    weldDof_weights_vertical = np.ones((weldDof.shape[0], weldDof.shape[1] + 1), dtype=int)
    weldDof_weights_vertical[:, :-1] = weldDof  # initialize weights to 1 (no offset)
    v_welds = weldDof_weights_vertical[v_mask].astype(float)  # vertical connections
    h_welds = weldDof_weights_vertical[~v_mask]  # horizontal connections
    # new_welds: beads have three vertical neighbours (2 new ones) except first and last beads of layers that only have two vertical neighbours (end particle only has 1 new one)
    new_welds = np.zeros((v_welds.shape[0] * 2 - 4 * (nLayers_v - 1), 5)).astype(float)
    new_welds = np.zeros((v_welds.shape[0] * 2, 5)).astype(float)  # rough over assumption of number of lines, new_welds[:new_row] will delete extra empty lines in the end

    new_row = 0

    for row in range(v_welds.shape[0]):
        node, particle = v_welds[row, [2, 3]].astype(int)  # follower node/particle
        Delta = overhang[node]

        v_con = v_connections.get(node)  # nodes potentially vertically connected to this node as leaders (vertical neighbors)
        main_leader = {v_welds[row, 0].astype(int)}  # initial leader node
        new_leader = v_con.difference(main_leader)  # new leader nodes due to offset

        ## cantilever
        if len(v_con) == 1:  # node only has 1 vertical neighbor (thinwall)
            if particle == 2:
                v_welds[row, 4] = weights_cantilever[node, 0] + weights_cantilever[node, 1]  # weight of initial leader node/particle
                new_welds[new_row, 4] = weights_cantilever[node, 2]  # weight of new leader node/particle
                new_welds[new_row, [0, 1, 2, 3]] = v_welds[row, [0, 1, 2, 3]] + [0, 1, 0, 0]  # new leader node/particle
            elif particle == 3:
                v_welds[row, 4] = weights_cantilever[node, 0] + weights_cantilever[node, 2]  # weight of initial leader node/particle
                new_welds[new_row, 4] = weights_cantilever[node, 1]  # weight of new leader node/particle
                new_welds[new_row, [0, 1, 2, 3]] = v_welds[row, [0, 1, 2, 3]] + [0, -1, 0, 0]  # new leader node/particle
            new_row += 1

        elif len(v_con) == 2 and list(main_leader) > list(new_leader) and Delta > 0:  # if node only has 2 vertical neighbors (first or last bead of the layer) AND node is last of layer AND positive overhang

            if particle == 2:
                v_welds[row, 4] = weights_cantilever[node, 0]  # weight of initial leader node
                new_welds[new_row, 4] = weights_cantilever[node, 1]  # weight of new leader node/particle
                new_welds[new_row + 1, 4] = weights_cantilever[node, 2]  # weight of new leader node/particle
                new_welds[new_row, 1:4] = v_welds[row, 1:4]  # weld particles are updated
                new_welds[new_row + 1, 1:4] = v_welds[row, 1:4] + [1, 0, 0]  # weld particles are updated
                new_welds[new_row, 0] = list(new_leader)[0]  # new leader nodes are updated
                new_welds[new_row + 1, 0] = list(main_leader)[0]
                new_row += 2
            else:  # particle == 3
                v_welds[row, 4] = 1  # remains fully connected to initial particle, no new weld


        elif len(v_con) == 2 and list(main_leader) < list(new_leader) and Delta < 0:  # if node only has 2 vertical neighbors (first or last bead of the layer) AND node is first of layer AND negative overhang

            if particle == 3:
                v_welds[row, 4] = weights_cantilever[node, 0]  # weight of initial leader node
                new_welds[new_row, 4] = weights_cantilever[node, 1]  # weight of new leader node/particle
                new_welds[new_row + 1, 4] = weights_cantilever[node, 2]  # weight of new leader node/particle
                new_welds[new_row, 1:4] = v_welds[row, 1:4] - [1, 0, 0]  # weld particles are updated
                new_welds[new_row + 1, 1:4] = v_welds[row, 1:4]  # weld particles are updated
                new_welds[new_row, 0] = next(iter(main_leader))  # new leader nodes are updated
                new_welds[new_row + 1, 0] = list(new_leader)[0]
                new_row += 2
            else:  # particle == 2
                v_welds[row, 4] = 1  # remains fully connected to initial particle, no new weld

        ## straddle
        else:  # ie if len(v_con)==3 or ( len(v_con) == 2 and (list(main_leader) > list(new_leader) and Delta < 0) or ( len(v_con) == 2 and (list(main_leader) < list(new_leader) and Delta > 0)
            # if len(v_con) == 3:  # if node has 3 vertical neighbors (any bead except first and last of the layer)
            if particle == 2:
                # update welding weight
                v_welds[row, 4] = weights_2[node, 0]  # weight of initial leader node/particle
                new_welds[new_row, 4] = weights_2[node, 1]  # weight of new leader node/particle
                new_welds[new_row + 1, 4] = weights_2[node, 2]  # weight of new leader node/particle
                # update weld nodes
                if Delta > 0:
                    new_welds[new_row, 0] = next(iter(main_leader))  # new leader node is the greater one
                    new_welds[new_row + 1, 0] = np.max(list(new_leader))  # same leader node
                    # update weld particles
                    new_welds[new_row, 2:4] = v_welds[row, 2:4]  # duplicate follower node/particle
                    new_welds[new_row + 1, 2:4] = v_welds[row, 2:4]  # duplicate follower node/particle
                    new_welds[new_row, 1] = v_welds[row, 1] + 1 # update leader particle
                    new_welds[new_row + 1, 1] = v_welds[row, 1] + 1  # update leader particle
                elif Delta < 0:
                    new_welds[new_row, 0] = next(iter(main_leader))  # same leader node
                    new_welds[new_row + 1, 0] = np.min(list(new_leader))  # new leader node is the smaller one
                    # update weld particles
                    new_welds[new_row, 2:4] = v_welds[row, 2:4]  # duplicate follower node/particle
                    new_welds[new_row + 1, 2:4] = v_welds[row, 2:4]  # duplicate follower node/particle
                    new_welds[new_row, 1] = v_welds[row, 1] + 1  # update leader particle
                    new_welds[new_row + 1, 1] = v_welds[row, 1]  # same leader particle
                new_row += 2
            if particle == 3:
                # update welding weight
                v_welds[row, 4] = weights_3[node, 0]  # weight of initial leader node/particle
                new_welds[new_row, 4] = weights_3[node, 1]  # weight of new leader node/particle
                new_welds[new_row + 1, 4] = weights_3[node, 2]  # weight of new leader node/particle
                # update weld nodes
                if Delta > 0:
                    new_welds[new_row, 0] = np.max(list(new_leader))   # same leader node
                    new_welds[new_row + 1, 0] = next(iter(main_leader)) # different leader particle and different leader node
                    # update weld particles
                    new_welds[new_row, 2:4] = v_welds[row, 2:4]  # duplicate follower node/particle
                    new_welds[new_row + 1, 2:4] = v_welds[row, 2:4]  # duplicate follower node/particle
                    new_welds[new_row, 1] = v_welds[row, 1]  # same leader particle
                    new_welds[new_row + 1, 1] = v_welds[row, 1] - 1  # update leader particle
                elif Delta < 0:
                    new_welds[new_row, 0] = next(iter(main_leader))  # new leader node is the smaller one
                    new_welds[new_row + 1, 0] = np.min(list(new_leader))  # different leader particle but same leader node
                    # update weld particles
                    new_welds[new_row, 2:4] = v_welds[row, 2:4]  # duplicate follower node/particle
                    new_welds[new_row + 1, 2:4] = v_welds[row, 2:4]  # duplicate follower node/particle
                    new_welds[new_row, 1] = v_welds[row, 1] + 1  # update leader particle
                    new_welds[new_row + 1, 1] = v_welds[row, 1] + 1 # update leader particle
                new_row += 2

    new_weldDof_weights_vertical = np.vstack((h_welds, v_welds, new_welds[:new_row]))
    # command to extract results for node 4 and particule 2 : # new_weldDof_weights[(new_weldDof_weights[:,2]==4) * (new_weldDof_weights[:,3]==2)]
                                                              # extended_weldDof[(extended_weldDof[:,2]==4) * (extended_weldDof[:,3]==2)]
    return new_weldDof_weights_vertical


def find_leader(node, particle, Sw):
    """
    Find primary leader node and particle to given node and particle

    Parameters
    ----------
    node : node number (int or array)
    particle : particle number (int or array)
    Sw : weld.weldwire_matrix(nNodesTot, weldDof)
        welding matrix defining kinematic relations between follow particles and their primary leader particle

    Returns
    -------
    leader node and particle (int or array)
    """
    # Sw = weld.weldwire_matrix(nNodesTot, weldDof)
    nNodesTot = Sw.shape[0] // 12
    if isinstance(node, np.ndarray):
        node = node.astype(int)
        particle = particle.astype(int)
    elif isinstance(node, float):
        node = int(node)
        particle = int(particle)
    follower_index = node + nNodesTot * particle
    leader_index = Sw.indices[Sw.indptr[follower_index]]
    leader_node = leader_index % nNodesTot
    leader_particle = leader_index // nNodesTot
    return leader_node, leader_particle


def make_primary_leaders(extended_weldDof, Sw):
    primary_weldDof = extended_weldDof
    primary_weldDof[:, 0], primary_weldDof[:, 1] = find_leader(extended_weldDof[:, 0], extended_weldDof[:, 1], Sw)  # shift all follower particles to their primary leader
    return primary_weldDof


def weldwire_matrix_weights(nNodesTot, extended_weldDof):
    """
    Function that builds the wire welding matrices with appropriate weights to every kinematic connection
    nNodesTot = X.shape[0]

    Parameters
    ----------
    nNodesTot : int
        Number of total nodes discretising the structures.
    extended_weldDof : array of size (nNodes*(nLayers-1)*2, 5)
        A NumPy array representing the connection table of weld degrees of freedom including connection weight.
        Each row represents a connection between two particles, with columns representing:
           - Column 0: Leader Node
           - Column 1: Leader particle
           - Column 2: Follower Node
           - Column 3: Follower particle
           - Column 4: Connection weight
        Column [0,1] must be primary leaders, use make_primary_leader to ensure this condition primary_weldDof = shapeOffset.make_primary_leader(extended_weldDof, Sw) )
    Returns
    -------
    Sw : sparse matrix of shape (nNodesTot*nNodeDof, nNodesTot*nNodeDof)
        Inter-wire welding matrix / wire connectivity matrix

    """
    # There are welding conditions to apply
    weldDof_weights = extended_weldDof  # make_welding_connections_n(meshing, offset_weights, weldDof, v_connections, v_mask)            # original weldDof table including follower particles as leader to other follower particles
    # DOF indexes and weights
    wDOFidx = np.hstack((weldDof_weights[:, [0, 2]] + nNodesTot * weldDof_weights[:, [1, 3]], weldDof_weights[:, 4][:, np.newaxis]))
    # Repeat for all 3 coordinates
    wDOFidx = np.tile(wDOFidx, (3, 1)) + np.repeat(4 * nNodesTot * np.arange(3)[:, np.newaxis], wDOFidx.shape[0], axis=0) * [1, 1, 0]

    # Leader particles and follower particles
    wDOFidx_lead = wDOFidx[:, 0]
    wDOFidx_follow = wDOFidx[:, 1]
    wDOFidx_weights = wDOFidx[:, 2]

    # Build welding matrix
    ii_lead = wDOFidx_lead
    jj_lead = ii_lead  # leader particles remain on diagonal
    ii_follow = wDOFidx_follow
    jj_follow = jj_lead  # follower particles are shifted to their respective leader particle's column
    weights_follow = wDOFidx_weights

    Sw_follow = sp.sparse.csr_matrix((weights_follow, (ii_follow, jj_follow)), shape=(12 * nNodesTot, 12 * nNodesTot))
    Sw_lead = sp.sparse.csr_matrix((np.ones(12 * nNodesTot), (np.arange(12 * nNodesTot), np.arange(12 * nNodesTot))), shape=(12 * nNodesTot, 12 * nNodesTot))  # identité CSR
    Sw_lead[ii_follow, ii_follow] = 0  # any particle that is a follower is not a leader

    extended_Sw = Sw_follow + Sw_lead

    return extended_Sw


def update_weldwire_matrix_weights(nNodesTot, extended_weldDof_vertical, Sw):
    """
    Function that updates the wire welding matrix only assigning weights from extended_weldDof_vertical in previous Sw matrix (only vertical connections need to be updated)
    nNodesTot = X.shape[0]

    Parameters
    ----------
    nNodesTot : int
        Number of total nodes discretising the structures.
    extended_weldDof : array of size (nNodes*(nLayers-1)*2, 5)
        A NumPy array representing the connection table of weld degrees of freedom including connection weight.
        Each row represents a connection between two particles, with columns representing:
           - Column 0: Leader Node
           - Column 1: Leader particle
           - Column 2: Follower Node
           - Column 3: Follower particle
           - Column 4: Connection weight
        !!!!!!!!!! Column [0,1] must be primary leaders, use make_primary_leader to ensure this condition primary_weldDof = shapeOffset.make_primary_leader(extended_weldDof, Sw) ) !!!!!!!!!!
    Returns
    -------
    Sw : sparse matrix of shape (nNodesTot*nNodeDof, nNodesTot*nNodeDof)
        Inter-wire welding matrix / wire connectivity matrix

    """

    weldDof_weights = extended_weldDof_vertical  # primary_weldDof_weights[extended_v_mask] where extended_v_mask = welding_type(extended_weldDof)[:,-1].astype(bool)
    # weldDof_weights = make_welding_connections_n(meshing, offset_weights, weldDof, v_connections, v_mask)
    # primary_weldDof_weights = make_primary_leaders(weldDof_weights, Sw)

    # # DOF indexes and weights
    # wDOFidx = np.hstack((weldDof_weights[:, [0, 2]] + nNodesTot * weldDof_weights[:, [1, 3]], weldDof_weights[:, 4][:, np.newaxis]))
    # # Repeat for all 3 coordinates
    # wDOFidx = np.tile(wDOFidx, (3, 1)) + np.repeat(4 * nNodesTot * np.arange(3)[:, np.newaxis], wDOFidx.shape[0], axis=0) * [1, 1, 0]

    # direct broadcasting for   faster approach compared to above lines
    coordinate_offsets = 4 * nNodesTot * np.arange(3)[:, np.newaxis]
    wDOFidx = np.hstack([
    np.tile(weldDof_weights[:, [0, 2]] + nNodesTot * weldDof_weights[:, [1, 3]], (3, 1)),
    np.tile(weldDof_weights[:, 4][:, np.newaxis], (3, 1))])
    wDOFidx[:, :2] += coordinate_offsets.repeat(weldDof_weights.shape[0], axis=0)

    # Leader particles and follower particles
    wDOFidx_lead = wDOFidx[:, 0]
    wDOFidx_follow = wDOFidx[:, 1]
    wDOFidx_weights = wDOFidx[:, 2]

    # Build welding matrix
    ii_lead = wDOFidx_lead
    jj_lead = ii_lead  # leader particles remain on diagonal
    ii_follow = wDOFidx_follow
    jj_follow = jj_lead  # follower particles are shifted to their respective leader particle's column
    weights_follow = wDOFidx_weights

    # Sw_follow = sp.sparse.csr_matrix((weights_follow, (ii_follow, jj_follow)), shape=(12 * nNodesTot, 12 * nNodesTot))
    # Sw_lead = sp.sparse.csr_matrix((np.ones(12 * nNodesTot), (np.arange(12 * nNodesTot), np.arange(12 * nNodesTot))), shape=(12 * nNodesTot, 12 * nNodesTot))  # identité CSR
    # Sw_lead[ii_follow,ii_follow] = 0    # any particle that is a follower is not a leader
    #
    # extended_Sw = Sw_follow + Sw_lead
    Sw_update = Sw.copy()
    Sw_update[ii_follow, jj_follow] = weights_follow
    Sw_update.eliminate_zeros()  # remove zero entries (null weight)
    return Sw_update


def make_extended_weldDof(meshing, offset, weldDof):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing

    if nLayers_h == 1 : #thinwall
        v_connections = {i: {i - nNodes} if i >= nNodes else set() for i in range(nNodes*nLayers_v)}  # ~10 times faster than vertical_connections_n(meshing, weldDof)
        v_mask = np.ones(weldDof.shape[0], dtype=bool)
        offset_weights = welding_weights(meshing, offset)
        extended_weldDof = make_welding_connections_n(meshing, offset_weights, weldDof, v_connections, v_mask) # this makes up 100 % of function computing time, twice faster as for nLayers_h=2
    else :
        v_connections = vertical_connections_n(meshing, weldDof) # this makes up 36% of function computing time
        v_mask = connectionType(weldDof)
        offset_weights = welding_weights(meshing, offset)
        extended_weldDof = make_welding_connections_n(meshing, offset_weights, weldDof, v_connections, v_mask) # this makes up 62% of function computing time

    return extended_weldDof

# def make_extended_weldDof(meshing, offset, weldDof): # time every operation
#     start_total = time.time()
#
#     start = time.time()
#     v_connections = vertical_connections_n(meshing, weldDof)
#     print(f"vertical_connections_n time: {time.time() - start:.6f} seconds")
#
#     start = time.time()
#     v_mask = connectionType(weldDof)
#     print(f"connectionType time: {time.time() - start:.6f} seconds")
#
#     start = time.time()
#     offset_weights = welding_weights(meshing, offset)
#     print(f"welding_weights time: {time.time() - start:.6f} seconds")
#
#     start = time.time()
#     extended_weldDof = make_welding_connections_n(meshing, offset_weights, weldDof, v_connections, v_mask)
#     print(f"make_welding_connections_n time: {time.time() - start:.6f} seconds")
#
#     print(f"Total function time: {time.time() - start_total:.6f} seconds")
#     return extended_weldDof

def offset_weldwire_matrix(meshing, offset, weldDof, Sw):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nNodesTot = nNodes * nLayers_h * nLayers_v

    if nLayers_v == 1:
        return Sw

    extended_weldDof = make_extended_weldDof(meshing, offset, weldDof)   # this makes up 57% of function computing time

    extended_v_mask = welding_type(extended_weldDof)[:, -1].astype(bool)  # adds column with 0 for horizontal connection and 1 for vertical connection
    primary_weldDof = make_primary_leaders(extended_weldDof, Sw)  # shift all follower particles to their primary leader

    Sw_update = update_weldwire_matrix_weights(nNodesTot, primary_weldDof[extended_v_mask], Sw)  # only vertical connections need to be updated     # this makes up 40% of function computing time
    return Sw_update

# def offset_weldwire_matrix(meshing, offset, weldDof, Sw): # time every operation
#     start_total = time.time()
#
#     start = time.time()
#     L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
#     nNodesTot = nNodes * nLayers_h * nLayers_v
#     print(f"Meshing unpacking time: {time.time() - start:.6f} seconds")
#
#     # if nLayers_v == 1:
#     #    return Sw
#
#     start = time.time()
#     extended_weldDof = make_extended_weldDof(meshing, offset, weldDof)
#     print(f"make_extended_weldDof time: {time.time() - start:.6f} seconds")
#
#     start = time.time()
#     extended_v_mask = welding_type(extended_weldDof)[:, -1].astype(bool)
#     print(f"welding_type and mask creation time: {time.time() - start:.6f} seconds")
#
#     start = time.time()
#     primary_weldDof = make_primary_leaders(extended_weldDof, Sw)
#     print(f"make_primary_leaders time: {time.time() - start:.6f} seconds")
#
#     start = time.time()
#     Sw_update = update_weldwire_matrix_weights(nNodesTot, primary_weldDof[extended_v_mask], Sw)
#     print(f"update_weldwire_matrix_weights time: {time.time() - start:.6f} seconds")
#
#     print(f"Total function time: {time.time() - start_total:.6f} seconds")
#     return Sw_update
