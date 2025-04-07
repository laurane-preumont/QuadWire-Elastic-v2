'''
Fichier pour debugger tranquillement
#%% Mise en données
#%% Exemples pour comment lancer les calculs
#%% Fonctions de debuggage
#%% Mise en oeuvre
#%% brouillon
'''
#%% Mise en données
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from modules import mesh, weld, plot, forces
from shape import shapeOffset, derivateOffset
from shape.shapeOptim_module import projected_gradient_algorithm, offset2param, param2offset, us2param, ScaledCounterDeformationCost, DisplacementMinimizationCost, CounterDeformationCost, analyze_optimization_results, save_optimization_report
''''''
# Integration
elemOrder = 1
quadOrder = elemOrder + 1

# Material behavior
E = 50 #3e3  # Young modulus /!\ in MPa
nu = 0.3  # poisson ratio
alpha = 1.13e-5  # thermal expansion coefficient
optimizedBehavior = False  # use the optimized behavior

# Geometry
L = 10 #0  # length in mm                        #TODO: modif
Hn = 0.2 # width of the section in mm            #TODO: modif
Hb = 0.1 # height of the section in mm          #TODO: modif
beadType = "linear"  # linear, circular, square, quarter_circular, quarter_square, sinus
layerType = "normal"  # default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
meshType = True  # meshRefined
zigzag = False
split_bool = False

# Mesh Parameters
nLayers_h = 1  # number of horizontal layers (beads)
nLayers_v = 3  # number of vertical layers             #TODO: modif
nLayers = nLayers_h * nLayers_v  # number of layers

nNodes = 5 # number of nodes per layers                 #TODO: modif
nNodes_h = nNodes * (nLayers_h+split_bool)
nNodesTot = nNodes * nLayers
nElems = nNodes - 1
nElemsTot = nElems * nLayers

# thermal data
dT = -60
loadType = "uniform"  # "uniform", "linear", "quad" or "random", "poc" ie top layer cools down other layers don't

path = None
# Plot
toPlot = True
clrmap = 'stt'
scfplot = 100

# % Data recap as tuple
meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
loading = dT, loadType
loading0 = 0*dT, loadType
discretization = elemOrder, quadOrder
discretisation = discretization
material = E, nu, alpha, optimizedBehavior
plotting = toPlot, clrmap, scfplot

offset = shapeOffset.stacking_offset(meshing, "linear", 0, -0.3*(nLayers_v-1))  # offset between successive layers along t and n (2, nLayers_v*nNodes_h)

#%% Exemples pour comment lancer les calculs


#%% Fonctions de debuggage

def store_version(meshing, discretisation, material, structural_data,
                 matrices_data, weld_data, behavior_data, thermal_data, results_data):
    """
    Stocke les résultats complets pour comparaison

    Parameters:
    -----------
    meshing : tuple
        L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
    discretisation : tuple
        elemOrder, quadOrder
    material : tuple
        E, nu, alpha, optimizedBehavior
    structural_data : tuple
        X, Elems, U0, offset, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit,
        X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF
    matrices_data : tuple
        xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node,
        T, Tv, Tc, Ta, t, n, b, P
    weld_data : tuple
        weldDof, extended_weldDof, weldDof_unsplit
    behavior_data : tuple
        Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi, Sw0, Sw, Y
    thermal_data : tuple
        nPas, dTelem, Telem, Tg, nNodesTot_unsplit
    results_data : tuple
        U, Eps, Sigma
    """
    results = {
        'input_parameters': {
            'meshing': {
                'L': meshing[0],
                'Hn': meshing[1],
                'Hb': meshing[2],
                'beadType': meshing[3],
                'layerType': meshing[4],
                'nLayers_h': meshing[5],
                'nLayers_v': meshing[6],
                'nNodes': meshing[7]
            },
            'discretisation': {
                'elemOrder': discretisation[0],
                'quadOrder': discretisation[1]
            },
            'material': {
                'E': material[0],
                'nu': material[1],
                'alpha': material[2],
                'optimizedBehavior': material[3]
            }
        },
        'structural_data': {
            'mesh': {
                'X': structural_data[0],
                'Elems': structural_data[1],
                'U0': structural_data[2],
                'offset': structural_data[3],
            },
            'unstructured': {
                'Xunc': structural_data[4],
                'uncElems': structural_data[5],
                'Xunc_unsplit': structural_data[6],
                'uncElems_unsplit': structural_data[7],
                'X_unsplit': structural_data[8],
                'Elems_unsplit': structural_data[9]
            },
            'counts': {
                'nElemsTot': structural_data[10],
                'nUncNodes': structural_data[11],
                'nNodesTot': structural_data[12],
                'nNodeDOF': structural_data[13],
                'nDOF': structural_data[14]
            }
        },
        'matrices': {
            'quadrature': {
                'xiQ': matrices_data[0],
                'XIQ': matrices_data[1],
                'WQ': matrices_data[2],
                'nQP': matrices_data[3]
            },
            'shape_functions': {
                'N': matrices_data[4],
                'Dxi': matrices_data[5],
                'Ds': matrices_data[6]
            },
            'jacobian': {
                'J': matrices_data[7],
                'W': matrices_data[8]
            },
            'mapping': {
                'O': matrices_data[9],
                'qp2elem': matrices_data[10],
                'elemQ': matrices_data[11],
                'elem2node': matrices_data[12]
            },
            'transforms': {
                'T': matrices_data[13],
                'Tv': matrices_data[14],
                'Tc': matrices_data[15],
                'Ta': matrices_data[16]
            },
            'vectors': {
                't': matrices_data[17],
                'n': matrices_data[18],
                'b': matrices_data[19],
                'P': matrices_data[20]
            }
        },
        'weld_data': {
            'dofs': {
                'weldDof': weld_data[0]
            }
        },
        'behavior': {
            'matrices': {
                'Sni': behavior_data[0],
                'Sn': behavior_data[1],
                'Assemble': behavior_data[2],
                'Btot': behavior_data[3],
                'Ctot': behavior_data[4],
                'Rxi': behavior_data[5],
                'Rchi': behavior_data[6]
            },
            'weld': {
                'Sw0': behavior_data[7],
                'Sw': behavior_data[8],
                'Y': behavior_data[9]
            }
        },
        'thermal_data': {
            'time_steps': {
                'nPas': thermal_data[0]
            },
            'temperature': {
                'dTelem': thermal_data[1],
                'Telem': thermal_data[2],
                'Tg': thermal_data[3]
            },
            'mesh': {
                'nNodesTot_unsplit': thermal_data[4]
            }
        },
        'results': {
            'displacements': results_data[0],  # U
            'strains': results_data[1],        # Eps
            'stresses': results_data[2]        # Sigma
        }
    }
    return results


def compare_arrays_with_nan(arr1, arr2, rtol=1e-10, atol=1e-10):
    """
    Compare deux tableaux numpy en tenant compte des valeurs nan.
    """
    if arr1.shape != arr2.shape:
        return False, f"Dimensions différentes: {arr1.shape} vs {arr2.shape}"

    # Créer des masques pour les valeurs nan
    nan1 = np.isnan(arr1)
    nan2 = np.isnan(arr2)

    # Vérifier si les nan sont aux mêmes positions
    if not np.array_equal(nan1, nan2):
        diff_positions = np.sum(nan1 != nan2)
        return False, f"Positions des nan différentes ({diff_positions} positions)"

    # Comparer les valeurs non-nan
    valid_mask = ~nan1  # même chose que ~nan2 car on a vérifié qu'ils sont égaux
    if not np.any(valid_mask):
        return True, None  # Tous les éléments sont nan

    valid1 = arr1[valid_mask]
    valid2 = arr2[valid_mask]

    if not np.allclose(valid1, valid2, rtol=rtol, atol=atol):
        abs_diff = np.abs(valid1 - valid2)
        return False, {
            'max_diff': float(np.max(abs_diff)),
            'mean_diff': float(np.mean(abs_diff)),
            'rel_diff': float(np.max(abs_diff / (np.abs(valid1) + 1e-10)))
        }

    return True, None

def compare_dicts_with_sparse(dict1, dict2, rtol=1e-10, atol=1e-10, prefix=''):
    """
    Compare deux dictionnaires contenant des matrices creuses et des tableaux numpy avec nan.
    """
    def compare_sparse_matrices(mat1, mat2):
        if not (sp.sparse.issparse(mat1) and sp.sparse.issparse(mat2)):
            return False, "Types différents"
        if mat1.shape != mat2.shape:
            return False, f"Dimensions différentes: {mat1.shape} vs {mat2.shape}"
        if mat1.nnz != mat2.nnz:
            return False, f"Nombre d'éléments non-nuls différent: {mat1.nnz} vs {mat2.nnz}"

        # Convertir au même format pour la comparaison
        mat1_csr = mat1.tocsr()
        mat2_csr = mat2.tocsr()

        # Comparer les indices et les données
        indices_equal = (
            np.array_equal(mat1_csr.indices, mat2_csr.indices) and
            np.array_equal(mat1_csr.indptr, mat2_csr.indptr)
        )
        if not indices_equal:
            return False, "Structure des matrices creuses différente"

        if not np.allclose(mat1_csr.data, mat2_csr.data, rtol=rtol, atol=atol):
            abs_diff = np.abs(mat1_csr.data - mat2_csr.data)
            return False, {
                'max_diff': float(np.max(abs_diff)),
                'mean_diff': float(np.mean(abs_diff)),
                'rel_diff': float(np.max(abs_diff / (np.abs(mat1_csr.data) + 1e-10)))
            }
        return True, None

    differences = {}

    for key in set(dict1.keys()) | set(dict2.keys()):
        current_path = f"{prefix}/{key}" if prefix else key

        # Vérifier si la clé existe dans les deux dictionnaires
        if key not in dict1:
            differences[key] = "Absent dans dict1"
            continue
        if key not in dict2:
            differences[key] = "Absent dans dict2"
            continue

        val1, val2 = dict1[key], dict2[key]

        # Comparer selon le type
        if isinstance(val1, dict) and isinstance(val2, dict):
            sub_diff = compare_dicts_with_sparse(val1, val2, rtol, atol, current_path)
            if sub_diff:
                differences[key] = sub_diff

        elif sp.sparse.issparse(val1) or sp.sparse.issparse(val2):
            is_equal, msg = compare_sparse_matrices(val1, val2)
            if not is_equal:
                differences[key] = msg

        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            is_equal, msg = compare_arrays_with_nan(val1, val2, rtol, atol)
            if not is_equal:
                differences[key] = msg

        elif val1 != val2:
            if np.isnan(val1) and np.isnan(val2):
                continue  # Ignorer si les deux sont nan
            differences[key] = f"Valeurs différentes: {val1} vs {val2}"

    return differences if differences else None

#%% Mise en oeuvre

# [faire tourner additive mixte dans la console puis lancer les lignes suivantes pour stocker les données générées]
meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
discretisation =  elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
structural_data = meshing, X, Elems, U0, offset, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit, X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF, xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P, Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi
matrices_data = xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P
weld_data = weldDof
behavior_data = Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi, Sw0, Sw, Y
thermal_data = nPas, dTelem, Telem, Tg, nNodesTot_unsplit
results_data = U, Eps, Sigma

results_v0 = store_version(        # additive
    meshing, discretisation, material, structural_data,
    matrices_data, weld_data, behavior_data, thermal_data, results_data
)

# [faire tourner additive classic dans la console puis lancer les lignes suivantes pour stocker les données générées]
meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
discretisation =  elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
structural_data = meshing, X, Elems, U0, offset, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit, X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF, xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P, Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi
matrices_data = xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P
weld_data = weldDof
behavior_data = Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi, Sw0, Sw, Y
thermal_data = nPas, dTelem, Telem, Tg, nNodesTot_unsplit
results_data = U, Eps, Sigma

results_v1 = store_version(        # additive_mixte
    meshing, discretisation, material, structural_data,
    matrices_data, weld_data, behavior_data, thermal_data, results_data
)

# [faire tourner additive normal (qwa.additive) dans la console puis lancer les lignes suivantes pour stocker les données générées]
meshing = L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag
discretisation =  elemOrder, quadOrder
material = E, nu, alpha, optimizedBehavior
structural_data = meshing, X, Elems, U0, offset, Xunc, uncElems, Xunc_unsplit, uncElems_unsplit, X_unsplit, Elems_unsplit, nElemsTot, nUncNodes, nNodesTot, nNodeDOF, nDOF, xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P, Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi
matrices_data = xiQ, XIQ, WQ, nQP, N, Dxi, Ds, J, W, O, qp2elem, elemQ, elem2node, T, Tv, Tc, Ta, t, n, b, P
weld_data = weldDof
behavior_data = Sni, Sn, Assemble, Btot, Ctot, Rxi, Rchi, Sw0, Sw, Y
thermal_data = nPas, dTelem, Telem, Tg, nNodesTot_unsplit
results_data = U, Eps, Sigma

results_v2 = store_version(        # additive
    meshing, discretisation, material, structural_data,
    matrices_data, weld_data, behavior_data, thermal_data, results_data
)

compare_dicts_with_sparse(results_v0, results_v2)      # différence entre additive mixte et additive
compare_dicts_with_sparse(results_v1, results_v2)      # différence entre additive classic et additive
compare_dicts_with_sparse(results_v0, results_v1)      # différence entre additive mixte et additive classic

#%% brouillon