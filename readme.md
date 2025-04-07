# QuadWire

QuadWire is a Python-based finite element modeling framework for simulating and optimizing additive manufacturing processes. The code implements a specialized approach for analyzing the mechanical behavior of additively manufactured structures, with particular focus on thermal and mechanical interactions during the printing process.

## Project Structure

### 1. Core Model (`qw_additive.py`)
Main simulation engine handling:
- Initialization of mesh and material parameters
- FEM mesh generation and welding conditions
- Thermal loading computation
- Iterative mechanical solving
- Results post-processing and visualization

### 2. Computation Modules (`modules/`)

#### Finite Element Analysis
- `fem.py`: Core FEM components
  - Shape functions and derivatives
  - Element quadrature
  - Integration matrices
  - Local/global coordinate transformations

- `mesh.py`: Mesh generation 
  - Supports linear, circular, square geometries
  - Adaptive mesh refinement
  - Node manipulation utilities

#### Material Physics
- `behavior.py`: Material behavior
  - Multi-scale homogenization
  - Thermal strain computation
  - Energy density calculations
  - Plasticity criteria

- `forces.py`: Force analysis
  - Internal forces computation
  - 3D stress tensor reconstruction
  - Delamination analysis

#### Manufacturing Process
- `weld.py`: Interface management
  - Node connection matrices
  - Kinematic welding conditions

- `thermaldata.py`: Thermal analysis
  - Fast thermal simulation integration
  - Glass transition handling

### 3. Shape Optimization Tools (`shape/`)

- `shapeOffset.py`: Stacking offset management
- `shapeOptim.py`: Gradient-based optimization
- `splitBeads.py`: Advanced bead division modeling
- `derivateOffset.py`: Shape sensitivity analysis

## Dependencies

Required Python packages:
- NumPy: Matrix operations
- SciPy: Sparse matrices, optimization
- Matplotlib: Visualization

