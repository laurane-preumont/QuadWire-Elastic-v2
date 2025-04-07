""" Module optimizing shape offset to minimize cost function"""

# %% import packages

import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from modules import mesh, fem, weld, behavior, plot
from shape import shapeOffset, derivateOffset

# %% Fonctions techniques
#from derivateOffset import shape_structure, derive_yKy, derive_yfbc, offset_epsilon


def param2overhang(param):
    nLayers_v = len(param)
    A = np.eye(nLayers_v)
    # A[0,0]=0    # pour que A@A.T soit inversible
    A[1:, :-1] -= np.eye(nLayers_v - 1)
    overhang = A @ param
    overhang = np.hstack((0, param[1:] - param[:-1]))
    return overhang


def offset2param(meshing, offset):
    '''offset includes values for every node while the gradient is only computed by layer (same value for every node of each layer)'''
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing

    param = offset[1][::nNodes*nLayers_h]
    return param


def param2offset(meshing, param):
    '''rolls back layerwise parameters into offset shape'''
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nNodes_h = nNodes * nLayers_h

    offset = np.repeat(np.stack((0 * param, param)), nNodes_h, axis=1)
    return offset


def offset2uslike(meshing, offset, Y, Assemble):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    nNodes_h = nNodes * nLayers_h
    X, Elems, U0 = mesh.generate_trajectory(meshing)

    offset_tnb = np.vstack((offset, np.zeros((1, nNodes_h * nLayers_v)))).T  # same size as X
    offset_tnb_unc = offset_tnb[Elems.flatten(), :]  # same size as Xunc
    offset_tnb_unc_4 = np.repeat(offset_tnb_unc, 4, axis=0).flatten()  # same size as f
    offset_tnb_unc_4_assemble_bound = Y.T @ Assemble.T @ offset_tnb_unc_4  # same size as us / yfbc
    return offset_tnb_unc_4_assemble_bound

# %% Ajout contrainte

def G(param, overhang_max):  # constraint G<0
    overhang = param2overhang(param)
    overhang_1 = np.abs(overhang)[1:]
    return overhang_1 - overhang_max


def Gprime(param):  # gradient of constraint with respect to delta
    nLayers_v = len(param)
    overhang = param2overhang(param)
    overhang_1 = np.abs(overhang)[1:]

    mask_pos = (overhang_1 > 0)
    mask_neg = ~mask_pos

    # Diagonal values : derive overhang = delta_{i+1} - delta_i with respect to delta_{i+1}
    diag_values = mask_pos * 1 + mask_neg * -1
    # Sub-diagonal values : derive overhang = delta_{i+1} - delta_i with respect to delta_i
    sub_diag_values = mask_pos * -1 + mask_neg * 1

    # derivative of g(delta)=|overhang|-1 with respect to delta (square matrix of size n)
    G_prime = np.eye(nLayers_v)
    np.fill_diagonal(G_prime, diag_values)
    np.fill_diagonal(G_prime[1:], sub_diag_values)
    G_prime = G_prime[1:]  # (overhang 0 is not constrained)
    return G_prime

def project_onto_constraints_ortho(param, overhang_max):
    """Project parameters orthogonally onto the feasible region defined by g(param) < 0"""
    nLayers_v = len(param)
    A = np.eye(nLayers_v)
    A[1:, :-1] -= np.eye(nLayers_v - 1)
    A = A[1:]
    overhang_1 = A @ param  # overhang 1 to n (overhang 0 is not constrained)
    A = ((A @ param > 0) * 2 - 1) * np.eye(nLayers_v - 1) @ A  # A@param = |overhang_1| valeur absolue

    b = overhang_max * np.ones(nLayers_v - 1)
    G = A @ param - b
    z = sp.linalg.solve(A @ A.T, A @ param - b)
    param_proj = param - A.T @ z
    return param_proj

# %% Fonction coût

def J_cost(us):
    Jval = np.sum(us ** 2)
    return Jval

# %% Dérivée de J par rapport à delta_i

def Jprime_i_diffFinies(meshing, offset, loading, discretization, material, epsilon, i):
    offset_epsilon_i = derivateOffset.offset_epsilon(meshing, offset, epsilon, i)
    us = derivateOffset.shape_structure(meshing, offset, loading, discretization, material)[1]
    us_epsilon_i = derivateOffset.shape_structure(meshing, offset_epsilon_i, loading, discretization, material)[1]

    # fonction objectif : somme sur toute la structure des carrés des déplacements
    J = J_cost(us)  # * Hb
    J_epsilon = J_cost(us_epsilon_i)  # * Hb

    return (J_epsilon - J) / epsilon


def Jprime_diffFinies(meshing, offset, loading, discretization, material, epsilon=1e-6):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    J_prime_diffFinies = np.zeros(nLayers_v)
    for i in range(nLayers_v):
        J_prime_diffFinies[i] = Jprime_i_diffFinies(meshing, offset, loading, discretization, material, epsilon, i)
    return J_prime_diffFinies


def Jprime(us, p, meshing, offset, loading, discretization, material):
    L, Hn, Hb, nLayers_h, nLayers_v, nNodes, beadType, layerType, meshType, zigzag = meshing
    J_prime = np.zeros(nLayers_v)

    for i in range(nLayers_v):
        Aprime = derivateOffset.derive_yKy(meshing, offset, loading, discretization, material, i)
        fprime = derivateOffset.derive_yfbc(meshing, offset, loading, discretization, material, i)
        J_prime[i] = - np.sum(np.multiply(p[:, np.newaxis], (Aprime @ us[:, np.newaxis] - fprime)))
    return J_prime


def compute_adjoint_state(meshing, A, us):
    ''' attention : la formule de l'état adjoint dépend de la fonction objectif '''
    # Résolution du problème adjoint : transpose(A) * p = 2 * u
    p = sp.sparse.linalg.spsolve(A.T, 2 * us)
    return p


def compute_next_param_gradient(param, mu, J_prime):
    return param - mu * J_prime


def update_step_size(mu, L_new, L_old, alpha_plus, alpha_minus):
    if L_new <= L_old:
        print('mieux')
        return mu * alpha_plus
    else:
        print('pire')
        return mu * alpha_minus


# Algorithme principal
from enum import Enum

class TerminationReason(Enum):
    MAX_ITERATIONS = "Maximum iterations reached"
    COST_TOLERANCE = "Cost function tolerance reached"
    FLAT_GRADIENT = "Gradient became flat (converged)"

def projected_gradient_algorithm(meshing, initial_offset, loading, discretization, material, max_iterations, tolerance):
    # Paramètres de l'algorithme
    overhang_max = 1
    alpha_plus = 1.5
    alpha_minus = 0.8
    gradient_tolerance = 1e-12 

    # Initialize parameter vectors and lists to store values
    offset = initial_offset
    param = offset2param(meshing, offset)

    # Store values
    param_values = []
    mu_values = []
    J_values = []
    Jprime_values = []
    G_values = []

    # Start the clock
    tic = time.time()

    # Initial computations
    U, us, yKy, Assemble, Y, yfbc, X, Elems = derivateOffset.shape_structure(
        meshing, offset, loading, discretization, material
    )
    Jval = J_cost(us)
    p = compute_adjoint_state(meshing, yKy, us)
    J_prime = Jprime(us, p, meshing, offset, loading, discretization, material)
    Gval = G(param, overhang_max) < 0
    mu = 0.1 * 1 / np.sqrt(np.sum(J_prime ** 2))

    # Initialize termination reason
    termination_reason = None

    k = 0
    while k < max_iterations:
        print('k=', k)
        J_old = Jval

        # Store values
        param_values.append(param)
        mu_values.append(mu)
        J_values.append(Jval)
        Jprime_values.append(J_prime)
        G_values.append(Gval)

        # Check termination conditions
        if Jval <= tolerance:
            termination_reason = TerminationReason.COST_TOLERANCE
            break
        if np.linalg.norm(J_prime) <= gradient_tolerance:
            termination_reason = TerminationReason.FLAT_GRADIENT
            break

        # Update parameters
        param_new = compute_next_param_gradient(param, mu, J_prime)
        Gval = G(param_new, overhang_max) < 0
        if np.all(Gval):
            print(' ')
        if not np.all(Gval):
            print('contrainte active')
            param_new = project_onto_constraints_ortho(param_new, overhang_max)
        offset_new = param2offset(meshing, param_new)

        # Update mechanical problem for new offset
        U_new, us_new, yKy_new, Assemble_new, Y_new, yfbc_new, X, Elems = derivateOffset.shape_structure(
            meshing, offset_new, loading, discretization, material
        )

        # Update
        J_new = J_cost(us_new)
        if J_new <= J_old:
            # Update for next iteration
            Jval = J_new
            param, offset = param_new, offset_new
            us, yKy, yfbc = us_new, yKy_new, yfbc_new
            p = compute_adjoint_state(meshing, yKy, us)
            J_prime = Jprime(us_new, p, meshing, offset, loading, discretization, material)

        # Update step size
        mu = update_step_size(mu, J_new, J_old, alpha_plus, alpha_minus)

        k += 1

    # If we exit the loop without setting a termination reason, it must be max iterations
    if termination_reason is None:
        termination_reason = TerminationReason.MAX_ITERATIONS

    # Stop the clock
    toc = time.time() - tic
    print(f'done : J(p)= {J_values[-1]}')
    print(f'Elapsed time: {toc}')
    print(f'Termination reason: {termination_reason.value}')

    return {
        'param_values': param_values,
        'J_values': J_values,
        'Jprime_values': Jprime_values,
        'mu_values': mu_values,
        'G_values': G_values,
        'termination_reason': termination_reason,
        'iterations': k,
        'final_cost': J_values[-1],
        'computation_time': toc
    }

# result = projected_gradient_algorithm(meshing, initial_offset, loading,
#                                    discretization, material, max_iterations, tolerance)
# print(f"Algorithm terminated because: {result['termination_reason'].value}")
# print(f"Took {result['iterations']} iterations")
# print(f"Final cost: {result['final_cost']}")


# %% Descente de gradient (sans contrainte)
#
# def gradient_descent(meshing, initial_offset, loading, discretization, material, tolJ=1e-7, tolJprime=1e-9, kmax=50):
#     # Initialize parameter vectors and lists to store values
#     param0 = offset2param(meshing, initial_offset)
#     J_0 = costFunction(meshing, initial_offset, loading, discretization, material)
#     J_prime0 = Jprime(meshing, initial_offset, loading, discretization, material)
#
#     J_values = [J_0]
#     Jprime_values = [np.linalg.norm(J_prime0)]
#     param_values = [param0]
#     # Calculate an initial step size based on the magnitude of the gradient
#     mu = 0.1 * 1 / np.sqrt(np.sum(J_prime0 ** 2))
#     mu_values = [mu]
#
#     # Main optimization loop
#     # %start the clock
#     tic = time.time()
#     k = 0
#     param = param0
#     J = J_0
#     J_prime = J_prime0
#     while J_values[-1] > tolJ and np.abs(Jprime_values[-1]) > tolJprime and k < kmax:
#         # Store the current parameter vector
#         param_old = param
#         # Update the parameter vector using the step size and gradient
#         param = param - mu * J_prime
#         offset = param2offset(meshing, param)
#         # Calculate the cost function with the updated parameter vector
#         J_new = costFunction(meshing, offset, loading, discretization, material)
#
#         # Check conditions for adjusting the step size or updating J
#         if np.linalg.norm(J_new) > (1 + 1e-8) * np.linalg.norm(J):
#             print('pire')
#             # Reduce the step size and reset offset to the previous value
#             mu = mu / 2
#             param = param_old
#         elif np.linalg.norm(J_new) < np.linalg.norm(J):
#             print('mieux')
#             # Increase the step size and update J
#             mu = 1.5 * mu
#             J = J_new
#         else:
#             print('bof')
#             # Update J_0
#             J = J_new
#
#         # Recalculate the gradient with the updated parameter vector
#         offset = param2offset(meshing, param)
#         J_prime = Jprime(meshing, offset, loading, discretization, material)
#
#         # Store values for analysis and debugging
#         J_values.append(J_0)
#         Jprime_values.append(np.linalg.norm(J_prime))
#         mu_values.append(mu)
#         param_values.append(param)
#
#         print(k, J_values[-1], Jprime_values[-1])
#         k += 1
#
#     # % stop the clock
#     toc = time.time() - tic
#     print('done : J(p)=', J_values[-1])
#     print('Elapsed time:', toc)
#
#     return param_values, J_values, Jprime_values, mu_values
#


# def uzawa_algorithm(meshing, initial_offset, loading, discretization, material, initial_lambda_g, max_iterations, tolerance):
#     def compute_next_param_uzawa(param, mu, J_prime, G_prime, lambda_g):
#        return param - mu * (J_prime + lambda_g @ G_prime)
#     # Paramètres de l'algorithme
#     mu_g = 0.5  # petit parametre : peu d'oscillation mais beaucoup d'itérations
#     alpha_plus = 1.2
#     alpha_minus = 0.8
#
#     # Initialize parameter vectors and lists to store values
#     offset = initial_offset
#     param = offset2param(meshing, offset)
#     lambda_g = initial_lambda_g
#
#     # Store values
#     param_values = []
#     mu_values = []
#     J_values = []
#     Jprime_values = []
#     L_values = []
#     lambda_g_values = []
#     G_values = []
#
#     # Main minimization loop
#     # %start the clock
#     tic = time.time()
#
#     U, us, yKy, Assemble, Y, yfbc, X, Elems = derivateOffset.shape_structure(meshing, offset,loading, discretization, material)
#     # Compute initial gradient for step size calculation
#     p = compute_adjoint_state(meshing, yKy, us)
#     J_prime0 = Jprime(us, p, meshing, offset,loading, discretization, material)
#     mu = 0.1 * 1 / np.sqrt(np.sum(J_prime0 ** 2))
#
#     k = 0
#     while k < max_iterations:
#         print('k=', k)
#
#         # Compute adjoint state
#         p = compute_adjoint_state(meshing, yKy, us)
#
#         # Compute function values and gradients
#         Jval = J(meshing, us)
#         Gval = G(param)
#         J_prime = Jprime(us, p, meshing, offset,loading, discretization, material)
#         G_prime = Gprime(param)
#
#         # compute Lagrangian
#         L_old = Jval + lambda_g @ Gval.T
#
#         # store values
#         param_values.append(param)
#         mu_values.append(mu)
#         J_values.append(Jval)
#         Jprime_values.append(J_prime)
#         L_values.append(L_old)
#         lambda_g_values.append(lambda_g)
#         G_values.append(Gval)
#
#         # Update parameters
#         param_new = compute_next_param_uzawa(param, mu, J_prime, G_prime, lambda_g)
#         # param_new = project_onto_constraints(param_new, overhang_max)
#         offset_new = param2offset(meshing, param_new)
#         lambda_g_new = np.maximum(0, lambda_g + mu_g * G(param_new))
#
#         # update mechanical problem for new offset
#         U_new, us_new, yKy_new, Assemble_new, Y_new, yfbc_new, X, Elems = derivateOffset.shape_structure(meshing, offset_new,loading, discretization, material)
#         # Compute new Lagrangian
#         L_new = J(meshing, us_new) + lambda_g_new @ G(param_new)
#         if L_new <= L_old:
#             us, yKy, yfbc = us_new, yKy_new, yfbc_new
#         # Update step size
#         mu = update_step_size(mu, L_new, L_old, alpha_plus, alpha_minus)
#
#         # Check for convergence
#         tol = np.linalg.norm(param_new - param)
#         print('tol=', tol)
#         if tol < tolerance:
#             print('tolerance reached')
#             break
#
#         # Update for next iteration
#         param, offset = param_new, offset_new
#         lambda_g = lambda_g_new
#         k += 1
#
#     # % stop the clock
#     toc = time.time() - tic
#     print('done : J(p)=', J_values[-1])
#     print('Elapsed time:', toc)
#
#     return param, offset, lambda_g, param_values, J_values, Jprime_values, mu_values, L_values, lambda_g_values, G_values
#
# def projected_gradient_algorithm_intuit(meshing, initial_offset,loading, discretization, material, max_iterations, tolerance):
#
#     # Paramètres de l'algorithme
#     alpha_plus = 1.2
#     alpha_minus = 0.5
#
#     # Initialize parameter vectors and lists to store values
#     offset = initial_offset
#     param = offset2param(meshing, offset)
#
#     # Store values
#     param_values = []
#     mu_values = []
#     J_values = []
#     Jprime_values = []
#     G_values = []
#
#     # Main minimization loop
#     # %start the clock
#     tic = time.time()
#
#     U, us, yKy, Assemble, Y, yfbc, X, Elems = derivateOffset.shape_structure(meshing, offset,loading, discretization, material)
#     # Compute initial gradient for step size calculation
#     Jval = J(meshing, us)
#     p = compute_adjoint_state(meshing, yKy, us)
#     J_prime = Jprime(us, p, meshing, offset,loading, discretization, material)
#     Gval = G(param)
#     mu = 1 / np.sqrt(np.sum(J_prime ** 2))
#
#     k = 0
#     while k < max_iterations and Jval > tolerance :
#         print('k=', k)
#         J_old = Jval
#
#         # store values
#         param_values.append(param)
#         mu_values.append(mu)
#         J_values.append(Jval)
#         Jprime_values.append(J_prime)
#         G_values.append(Gval)
#
#         # Update parameters
#         param_new = compute_next_param_gradient(param, mu, J_prime)
#         Gval = G(param_new)<0
#         if np.all(Gval) :
#             print(' ')
#         if not np.all(Gval) :
#             print('contrainte active')
#             param_new = project_onto_constraints_intuit(param_new, overhang_max)
#         offset_new = param2offset(meshing, param_new)
#
#         # update mechanical problem for new offset
#         U_new, us_new, yKy_new, Assemble_new, Y_new, yfbc_new, X, Elems = derivateOffset.shape_structure(meshing, offset_new,loading, discretization, material)
#         # Update
#         J_new = J(meshing, us_new)
#         if J_new <= J_old:
#             # Update for next iteration
#             Jval = J_new
#             param, offset = param_new, offset_new
#             us, yKy, yfbc = us_new, yKy_new, yfbc_new
#             p = compute_adjoint_state(meshing, yKy, us)
#             J_prime = Jprime(us_new, p, meshing, offset_new,loading, discretization, material)
#
#         # Update step size
#         mu = update_step_size(mu, J_new, J_old, alpha_plus, alpha_minus)
#
#         k += 1
#
#     # % stop the clock
#     toc = time.time() - tic
#     print('done : J(p)=', J_values[-1])
#     print('Elapsed time:', toc)
#
#     return param_values, J_values, Jprime_values, mu_values, G_values
#


# def project_onto_constraints_intuit(param, overhang_max):
#         """Project parameters orthogonally onto the feasible region defined by g(param) < 0"""
#         # projection intuitive mais non orthogonale :
#         overhang = param2overhang(param)
#         overhang = np.minimum(overhang, overhang_max)
#         overhang = np.maximum(overhang, -overhang_max)
#         param_proj = np.cumsum(overhang)
#         return param_proj
