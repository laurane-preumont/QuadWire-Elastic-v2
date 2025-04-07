"""
Testing shape optimization algorithms
"""
# %%
# clear workspace

from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic('reset', '-sf')

# %% Imports
import time
import numpy as np
import matplotlib.pyplot as plt
import json

from modules import mesh, weld, plot, forces
from shape import shapeOffset, derivateOffset
from shape.shapeOptim_module import projected_gradient_algorithm, offset2param, param2offset, us2param, DisplacementMinimizationCost, CounterDeformationCost, analyze_optimization_results, \
    save_optimization_report, plot_param_states, plot_cost_convergence, plot_parameter_evolution, plotparam

''''''
# Integration
elemOrder = 1
quadOrder = elemOrder + 1

# Material behavior
E = 3e3  # Young modulus /!\ in MPa
nu = 0.3  # poisson ratio
alpha = 1.13e-5  # thermal expansion coefficient
optimizedBehavior = False  # use the optimized behavior

# Geometry
L = 100 #0  # length in mm                        #TODO: modif
Hn = 1 # width of the section in mm            #TODO: modif
Hb = 1 # height of the section in mm          #TODO: modif
beadType = "linear"  # linear, circular, square, quarter_circular, quarter_square, sinus
layerType = "normal"  # default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
meshType = False  # meshRefined
zigzag = False
split_bool = False

# Mesh Parameters
nLayers_h = 1  # number of horizontal layers (beads)
nLayers_v = 15  # number of vertical layers             #TODO: modif
nLayers = nLayers_h * nLayers_v  # number of layers

nNodes = 25 # number of nodes per layers                 #TODO: modif
nNodes_h = nNodes * (nLayers_h+split_bool)
nNodesTot = nNodes * nLayers
nElems = nNodes - 1
nElemsTot = nElems * nLayers

# thermal data
dT = -600 #-0.05/alpha # deformation imposée de 5% pour inclure la polymérisation
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
material = E, nu, alpha, optimizedBehavior
plotting = toPlot, clrmap, scfplot

offset = shapeOffset.stacking_offset(meshing, "linear", 0, -0.5*(nLayers_v-1))  # offset between successive layers along t and n (2, nLayers_v*nNodes_h)


# %% Descending gradient algorithm
initial_offset = offset
initial_param = offset2param(meshing, offset)
param = initial_param
# Initial problem setup
initialization = derivateOffset.fixed_structure(meshing, loading, discretization, material, additive=False)
_, us, _, _, Y, *_ = derivateOffset.shape_structure(initialization, meshing, initial_offset, loading, discretization, material, additive=False)
initial_deformation = us2param(meshing, us, Y)

# %%
cost_function = CounterDeformationCost()

results = projected_gradient_algorithm(
    meshing=meshing,
    offset=initial_offset,
    loading=loading,
    discretization=discretization,
    material=material,
    cost_function=cost_function,
    additive=False,
    max_iterations=30,
    tolerance=1e-6,
    rel_tolerance=1e-8#,
    #previous_state=results['current_state']
)
analyze_optimization_results(results)

plotparam(results['param_values'], 'Layerwise offset', gap=1, difference=True)

# check final result :
final_param = results['param_values'][-1]
final_offset = param2offset(meshing, final_param)
U, us, yKy, Assemble, Y, yfbc, X, Elems = derivateOffset.shape_structure(initialization, meshing, final_offset, loading, discretization, material, additive=False)
final_deformation = us2param(meshing, us, Y)

moyenne_finale, moyenne_initiale = np.mean(final_deformation + final_param - initial_param), np.mean(initial_deformation)
print('moyenne finale : {:.2e}'.format(moyenne_finale),
      '\n moyenne initiale : {:.2e}'.format(moyenne_initiale),
      '\n amélioration : {:.2e}'.format(moyenne_initiale/moyenne_finale))

# %%

cost_function = DisplacementMinimizationCost() # CounterDeformationCost() #

results = projected_gradient_algorithm(
        meshing=meshing,
        offset=initial_offset,
        loading=loading,
        discretization=discretization,
        material=material,
        cost_function=cost_function,
        additive=False,
        max_iterations=50,
        tolerance=1e-8,
        rel_tolerance=1e-8
    )
analyze_optimization_results(results)
# plot data
plot.plotparam(results['param_values'], 'Layerwise offset', gap=1, difference=False)



# check final result :
final_param = results['param_values'][-1]
final_offset = param2offset(meshing, final_param)
U, us, yKy, Assemble, Y, yfbc, X, Elems = derivateOffset.shape_structure(initialization, meshing, final_offset, loading, discretization, material, additive=False)
final_deformation = us2param(meshing, us, Y)

moyenne_finale, moyenne_initiale = np.mean(final_deformation), np.mean(initial_deformation)
print('moyenne finale : {:.2e}'.format(moyenne_finale),
      '\n moyenne initiale : {:.2e}'.format(moyenne_initiale),
      '\n amélioration : {:.0e}'.format(moyenne_initiale/moyenne_finale))

#%%
# continue
if results['termination_reason'].value == 'Maximum iterations reached' :
    new_results = projected_gradient_algorithm(
        meshing=meshing,
        offset=initial_offset,
        loading=loading,
        discretization=discretization,
        material=material,
        cost_function=cost_function,
        additive=False,
        max_iterations=50,
        tolerance=1e-6,
        rel_tolerance=1e-8,
        previous_state=results['current_state']
    )
    analyze_optimization_results(new_results)
    # plot data
    plotparam(new_results['param_values'], 'Layerwise offset', gap=20, difference=True)

# %% check final state
final_param = results['param_values'][-1]
final_offset = param2offset(meshing, final_param)
U, us, yKy, Assemble, Y, yfbc, X, Elems = derivateOffset.shape_structure(initialization, meshing, final_offset, loading, discretization, material, additive=False)
final_deformation = us2param(meshing, us, Y)

# %%
# Save report
save_optimization_report(results, 'optimization_results.json', include_plots=True)
with open('optimization_results.json', 'r') as f:
        data = json.load(f)
plot_data = data['plotting_data']
iterations = np.array(plot_data['iterations'])
J_values = np.array(plot_data['cost_values'])
Jprime_values = results['Jprime_values']
param_array = np.array(plot_data['parameter_values'])

# Save evolution of states
fig_evolution = plotparam(results['param_values'], '', gap=20, difference=True)
fig_evolution.savefig('result_evolution.pdf', bbox_inches='tight', dpi=300)

# Save result - initial and final states
fig_states = plot_param_states(initial_param, final_param, initial_deformation, final_deformation, title='', scale=1, difference=True)
fig_states.savefig('result.pdf', bbox_inches='tight', dpi=300)

# Save cost convergence plot
fig_cost = plot_cost_convergence(iterations, J_values, Jprime_values)
fig_cost.savefig('J_cost.pdf', bbox_inches='tight', dpi=300)
#plt.close(fig_cost)

# Save parameter evolution plot
fig_param = plot_parameter_evolution(iterations, param_array, difference=True)
fig_param.savefig('param.pdf', bbox_inches='tight', dpi=300)
#plt.close(fig_param)


