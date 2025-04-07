# QuadWire

QuadWire is a Python-based finite element modeling framework for simulating and optimizing additive manufacturing processes. It implements a specialized approach for analyzing the mechanical behavior of additively manufactured structures, with a particular focus on mechanical interactions during the printing process.

The codebase also includes shape optimization capabilities using gradient descent with adjoint methods, featuring bead offset manipulation to minimize two different cost functions regarding final displacements (minimum displacement and counter-deformation).

---

## Code Structure

The project is organized into several modules:

├── qw_structure.py        # Complete structure loading simulation
├── qw_additive.py         # Element-wise additive loading simulation
├── qw_additive_mixte.py   # Layer-wise + element additive simulation

├── modules/               # Computation modules
│   ├── fem.py             # FEM for Lagrangian elements
│   ├── mesh.py            # 3D mesh generation
│   ├── behavior.py        # Material behavior
│   ├── forces.py          # QuadWire forces
│   ├── weld.py            # Welding conditions
│   ├── thermaldata.py     # Thermal analysis
│   ├── generate_thermal.py# generate analytical thermal data to run test simulations
│   ├── plot.py            # all display functions
│   ├── solver3D.py        # 3D fem framework (Cauchy3DTM)

├── shape/                 # Shape optimization
│   ├── shapeOffset.py     # Stacking offset
│   ├── derivateOffset.py  # Sensitivity analysis
│   ├── splitBeads.py      # Overhang bead kinematics
│   ├── shapeOptim_module.py # Gradient-based optimization

└── tests/                 # draft files
    ├── ...             # test files are used to check computations


---

## 1. Main Files

### Simulation Engines

- **`qw_structure.py`**: Simulates mechanical loading on the full structure in a single step.
- **`qw_additive.py`**: Simulates layer-by-layer additive manufacturing with element-by-element activation.
- **`qw_additive_mixte.py`**: Experimental mixed simulation combining layer birth and element activation. *(Work in progress — debugging layer birth load continuity)*

Main features:
- Mesh and material initialization
- FEM mesh generation and welding conditions
- Thermal loading (weak coupling)
- Iterative solving for additive simulations
- Post-processing and visualization

> ⚠️ `qw_structure.py` applies static load on the final geometry; `qw_additive.py` activates elements progressively; `qw_additive_mixte.py` handles dynamic mesh growth per layer but requires debugging to ensure continuity of loads and constraints.

---

## 2. Modules (`modules/`)

### Finite Element Analysis

- **`fem.py`**: Core FEM functions  
  - Shape functions, derivatives  
  - Quadrature and integration matrices  
  - Local/global transformations

- **`mesh.py`**: Mesh generator  
  - Supports 3D geometries  
  - Adaptive refinement  
  - Node utilities

### Material Physics

- **`behavior.py`**: Material laws  
  - Thermal strain  
  - Energy density  
  - Plasticity criteria  
  - Multi-scale homogenization

- **`forces.py`**: Force computations  
  - Internal forces  
  - 3D stress tensor  
  - Draft delamination models

### Manufacturing Process

- **`weld.py`**: Welding logic  
  - Connection matrices  
  - Kinematic constraints

- **`thermaldata.py`**: Temperature effects  
  - Weak thermal coupling  
  - Glass transition handling

---

## 3. Shape Optimization (`shape/`)

- **`shapeOffset.py`**: Manages bead deposition offset
- **`derivateOffset.py`**: Computes shape sensitivity
- **`shapeOptim_module.py`**: Gradient-based optimizer using adjoint methods
- **`splitBeads.py`**: Models overhangs and enriched bead division

Bead offset optimization requires extended kinematic handling to ensure proper welding between adjacent and overhang beads.

---

## Simulation Approaches

### Structure-Based Simulation

`qw_structure.py`: Simulates full structure loading in one step.

- Efficient for final static analyses
- Acts as a baseline for comparison

### Additive Simulation

`qw_additive.py`: Activates elements incrementally to simulate layer-by-layer deposition.

- Captures the printing sequence
- Models internal stress evolution

### Mixed Additive Simulation (Experimental)

`qw_additive_mixte.py`: Combines layer birth and element activation.

- Dynamic layer growth with nested activation
- Optimized mesh scaling
- Currently needs to be debugged

---

## Shape Optimization

### Bead Offset Optimization

Optimizes deposition path offsets to enhance mechanical performance:

- Handles complex interactions between beads
- Supports modeling of overhang behaviors

### Gradient Descent Optimization

Implements optimization using adjoint methods:

- Two customizable cost functions
- Efficient gradient computation
- Iterative convergence with stopping criteria

---

## Testing Framework

Located in the `tests/` directory.

- Draft test cases for some features
- Can be used as minimal working examples

---

## Dependencies

Make sure to install:

- `NumPy`: Numerical operations
- `SciPy`: Sparse matrices, optimization routines
- `Matplotlib`: Visualization
                                                  
## Credits
This code is a working version forked from QuadWire-Elastic initial version published along the seminal paper and first QuadWire article
``QuadWire: an extended one dimensional model for efficient mechanical simulations of bead-based additive manufacturing processes'' (hal-04609753)(https://doi.org/10.1016/j.cma.2024.117010)
published on Zenodo as 10.5281/zenodo.10822308 from rafaelviano/QuadWire-Elastic
This new v2 version includes analytical thermal generation,
mesh improvements (massive 3D geometries (instead of carpets and thinwalls), zigzag trajectories),
bead offset control and extended kinematic conditions (straddle beads and overhang beads),
shape optimization capabilities (shape/shapeOptim_module).