""" 
Module for importing thermal simulation results from the fast thermal simulation model to the QuadWire finite element kernel
"""

# %% Import packages
import numpy as np
# %%

def get_data(dataFiles, nElems, nLayers_h, nLayers_v, Tbuild, Tsubstrate=323.15):
    """
    Function for importing the results files of the fast thermal simulation model in the form of a numpy array.
    The temperature of the elements at the time steps when they don't physically exist is set to the build temperature.

    Parameters
    ----------
    dataFiles : list
        Number of time step in the thermal simulation.
    nElems : int
        Number of elements per layer.
        nElems = nNodes - 1 
    nLayers_h : int
        Number of layers of the structure in the horizontal direction.
    nLayers_v : int
        Number of layers of the structure in the build direction.
    Tbuild : float, optional
        Temperature of deposition of the material during the process.
        For a FDM (PLA) process it is = 353.15, for a metal process it is = 2463.060825.
    Tsubstrate : int, optional
        Temperature of the substrate. 
        The default is :  323.15.

    Returns
    -------
    thermalData : array of size (nTimeStep, nElemsTOT)
            Evolution of the elements' temperature throughout the simulation.
        thermalData[tk] = element temperature at time step tk
        thermalData[:, i] = temperature of the ith element throughout the simulation
    """
    

    nLayers = nLayers_h * nLayers_v
    nTimeStep = len(dataFiles)
    thermalData = Tbuild * (np.ones((nTimeStep, nElems * nLayers)))  # - np.diag([1 for k in range(n)]) ) # np.zeros((n,n))
    for i in range(nTimeStep):
        try:
            with open(dataFiles[i], 'r') as fichier:
                # Read each line of the file
                lignes = fichier.readlines()
                # Browse each line and add its contents to the final list
                for j in range(len(lignes)):
                    thermalData[i, j] = float(lignes[j].strip())  # - Tref
            # Close file
            fichier.close()
        except FileNotFoundError:
            print("File couldn't be found.")
        except Exception as e:
            print(f"Error occurred for file {str(e)}")
        thermalData = np.array(thermalData)
    return thermalData


def get_data2tamb(dataFiles, nElems, nLayers_h, nLayers_v, Tbuild, Tsubstrate=323.15, Tamb=293.15):
    """
    Function for importing the results files of the fast thermal simulation model in the form of a numpy array.
    Here an additional time step is added in order to include cool time of the structure to room temperature.
    This is useful for processes where the structure is printed on a heating plate such as FDM of PLA.
    The temperature of the elements at the time steps when they don't physically exist is set to the build temperature.

    Parameters
    ----------
    dataFiles : list
        list of data files for every time step in the thermal simulation. nTimeStep = len(dataFiles)
    nElems : int
        Number of elements per layer.
        nElems = nNodes - 1 
    nLayers : int
        Number of layers in the structure.
    Tbuild : float, optional
        Temperature of deposition of the material during the process.
        For a FDM (PLA) process its = 353.15, for a metal process its = 2463.060825.
    Tsubstrate : int, optional
        Temperature of the substrate. 
        The default is :  323.15.
    Tamb : int, optional
        Temperature of the room.

    Returns
    -------
    thermalData : array of size (nTimeStep, nElemsTot)
        Evolution of the elements' temperature throughout the simulation.
        thermalData[tk] = element temperature at time step tk
        thermalData[:, i] = temperature of the ith element throughout the simulation
    """
    nLayers = nLayers_h * nLayers_v
    nPas = len(dataFiles)
    thermalData = Tbuild * (np.ones((nPas+1, nElems * nLayers)))  # Tbuild * (np.ones((nTimeStep + 1, nElems * nLayers)))  # - np.diag([1 for k in range(n)]) ) # np.zeros((n,n))
    thermalData[-1] = Tamb * np.ones(nElems * nLayers)
    for i in range(nPas):
        try:
            with open(dataFiles[i], 'r') as fichier:
                # Read each line of the file
                lignes = fichier.readlines()
                # Browse each line and add its contents to the final list
                for j in range(len(lignes)):
                    thermalData[i, j] = float(lignes[j].strip())  # - Tref
            #fichier.close()  #remove the explicit fichier.close() because 'with' handles the closing
        except FileNotFoundError:
            print("File couldn't be found.")
        except Exception as e:
            print(f"Error occurred for file {str(e)}")
        thermalData = np.array(thermalData)

    return thermalData

def elem2node(Telem, nElems, nLayers_h, nLayers_v):

    """
    Function used to pass from the temperature at the elements to the temperature at the nodes by linear interpolation.

    Parameters
    ----------
    Telem : array of size (nTimeStep, nElemsTOT)
        Evolution of the elements' temperature throughout the simulation.
    nElems : int
        Number of elements per layer.
        nElems = nNodes - 1 
    nLayers : int
        Number of layers in the structure.

    Returns
    -------
    Tnode : array of size (nTimeStep, nNodesTOT)
        Evolution of the nodes' temperature throughout the simulation.
        Tnode[tk] = node temperature at time step tk
        Tnode[:, i] = temperature of the ith node throughout the simulation
    """
    
    nLayers = nLayers_h * nLayers_v
    nTimeStep = Telem.shape[0]
    Tnode = np.zeros((nTimeStep, (nElems + 1) * nLayers))
    for j in range(nLayers):
        for i in range(1, nElems):
            Tnode[:, j * (nElems + 1) + i] = (Telem[:, j * nElems + (i - 1)] + Telem[:, j * nElems + i]) / 2

        Tnode[:, j * (nElems + 1)] = 2 * Telem[:, j * nElems] - Tnode[:, j * (nElems + 1) + 1]
        Tnode[:, j * (nElems + 1) + nElems] = 2 * Telem[:, j * nElems + i] - Tnode[:, j * (nElems + 1) + nElems - 1]
    return Tnode

def delta_elem_transition_vitreuse(T, Tbuild, Tg):
    """
    Function used to differentiate a temperature matrix with respect to time. 
    nPointsTOT represents the number of calculation points.
    It can either be nNodesTOT if it's the nodal temperature matrix or nElemsTOT if it's the element temperature matrix.
    This function is used for material that undergo a glass transition :
        T > Tg : the material is liquid therefore the thermal eigenstrain is nul : dT = 0
        T < Tg : the material is solid

    Parameters
    ----------
    T : array of size (nTimeStep, nPointsTOT)
        Evolution of the points temperature throughout the simulation.
        T[tk] = points temperature at time step tk
        T[:, i] = temperature of the ith points throughout the simulation.
    Tbuild : float, optional
        Temperature of deposition of the material during the process.
        For a FDM (PLA) process its = 353.15, for a metal process its = 2463.060825.
    Tg : float, optional
        Glass transition of the material. 
        The default value is for PLA 328.15.

    Returns
    -------
    dT : array of size (nTimeStep - 1, nPointsTOT)
        Time differential of the points temperature throughout the simulation.

    """
    # nTimeStep = T.shape[0]
    # nPointsTOT = T.shape[1]
    # dT = np.zeros((nTimeStep, nPointsTOT))
    # dT[0] = np.zeros(nPointsTOT)
    #
    # for i in range(1, nTimeStep):
    #     for j in range(nPointsTOT):
    #         if T[i][j] < Tg:
    #             dT[i][j] = T[i][j] - T[i - 1][j]
    #         else:
    #             dT[i][j] = 0

    # speed up x100 without loop
    dT = np.vstack((np.zeros(T.shape[1]), T[1:] - T[:-1]))
    dT[T>Tg] *= 0

    return dT

def delta_time(T, Tbuild):                                                                             ## pas utilisé nul part
    """
    Function used to differentiate a temperature matrix with respect to time.
    nPointsTOT represents the number of calculation points.
    It can either be nNodesTOT if it's the nodal temperature matrix or nElemsTOT if it's the element temperature matrix.

    Parameters
    ----------
    T : array of size (nTimeStep, nPointsTOT)
        Evolution of the points temperature throughout the simulation.
        T[tk] = points temperature at time step tk
        T[:, i] = temperature of the ith points throughout the simulation.
    Tbuild : float, optional
        Temperature of deposition of the material during the process.
        For a FDM (PLA) process its = 353.15, for a metal process its = 2463.060825.

    Returns
    -------
    dT : array of size (nTimeStep - 1, nPointsTOT)
        Time differential of the points temperature throughout the simulation.
    """
    nTimeStep = T.shape[0] - 1
    nPointsTOT = T.shape[1]
    dT = np.zeros((nTimeStep, nPointsTOT))
    dT[0] = T[0] - Tbuild * np.ones(nPointsTOT)
    for i in range(1, nTimeStep):
        dT[i] = T[i] - T[i - 1]
    return dT


def temperature(tab_path, nElems, nLayers, tau=0, Tbuild=2463.060825, Tsubstrate=323.15, tau_f=0):     ## pas utilisé nul part
    Telem = get_data(tab_path, nElems, nLayers, Tbuild, Tsubstrate)
    Tnode = elem2node(Telem, nElems, nLayers)
    dTnode = delta_time(Tnode, Tbuild)
    return Telem, Tnode, dTnode

