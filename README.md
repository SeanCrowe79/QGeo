# QGeo

This project implements a numerical method due to Nielsen for finding geodesics on the special unitary group between the identity operator and a target unitary. This project is described in detail here: [arXiv:2504.16157](https://arxiv.org/abs/2504.16157)

## Quick Start

To install this package clone the repo with:
```bash
git clone https://github.com/SeanCrowe79/QGeo.git
```
Navigate to the project directory and run:
```bash
pip install -e .
```
This installs the module into your Conda environment. It is recommended to start with a fresh environment. From there the example notebook can be used to generate geodesic control functions for your desired quantum circuit.

## Usage

The main functionality is provided by the `GeoComplexity` and `GateSumComplexity` functions. You can use these to calculate the complexity of a given Qiskit `QuantumCircuit`.

The `GeoComplexity` function calculates the geodesic quantum complexity, which is a measure of the difficulty of preparing a given unitary matrix specified by a metric on the unitary group. It looks for curves of minimal length (geodesics) that connect the identity operator to the target unitary matrix.

The `GateSumComplexity` function calculates the gate sum complexity, which is the sum of the complexities of the individual gates in the circuit. This is useful for benchmarking the geodesic compilation of full circuits.

### Example: Calculating Geodesic Complexity

```python
from QGeo import GeoComplexity
from qiskit import QuantumCircuit

# Create a simple quantum circuit
qc = QuantumCircuit(1)
qc.rz(3.14, 0)

# Calculate the geodesic complexity
solution = GeoComplexity(qc)

print(f"Geodesic Complexity: {solution.geocomplex}")
```

### Example: Calculating Gate Sum Complexity

```python
from QGeo import GateSumComplexity
from qiskit import QuantumCircuit

# Create a quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Calculate the gate sum complexity
complexity = GateSumComplexity(qc)

print(f"Gate Sum Complexity: {complexity}")
```

### Example: Random IBM Circuit

You can also generate random circuits using the IBM gate set and analyze them. The `random_IBM_circuit` function generates a circuit using gates drawn from the set `{X, SX, Rz(θ), CZ}` with a uniform probability distribution:

```python
from QGeo import random_IBM_circuit, GeoComplexity

# Generate a random circuit with 3 qubits and 100 gates
qc = random_IBM_circuit(num_qubits=3, num_gates=100, seed=42)

# Calculate complexity
solution = GeoComplexity(qc)
print(f"Random Circuit Complexity: {solution.geocomplex}")
```

### Example: Fermion Chain Evolution

QGeo can also be used to analyze and optimize circuits with a block-type structure, such as those used for the quantum simulation of a fermionic chain. For example, the time evolution operator for a non-interacting fermionic chain can be approximated using Trotter-Suzuki decimation and written in terms of $R_{xx}$ and $R_{yy}$ gates. 

By compiling these repeating blocks into geodesic control functions using QGeo, significant cost reductions can be achieved. In the paper, it was shown that a Trotter block with a gate-sum complexity of ~132.4 could be reduced to a geodesic complexity of just 0.77, a 99.4% reduction in cost. This allows for the time evolution of fermionic chains for many more Trotter steps than would otherwise be possible.

```python
from QGeo import GeoComplexity, GateSumComplexity
from qiskit import QuantumCircuit

# Example of a Trotterized block for Fermion Chain Evolution
phi = 0.5
qc_trot = QuantumCircuit(4)

# Block 1
qc_trot.rxx(phi, 0, 1)
qc_trot.rxx(phi, 2, 3)
qc_trot.ryy(phi, 0, 1)
qc_trot.ryy(phi, 2, 3)

# Block 2
qc_trot.rxx(phi, 1, 2)
qc_trot.ryy(phi, 1, 2)

# Calculate complexities
geo_sol = GeoComplexity(qc_trot)
gate_sum = GateSumComplexity(qc_trot)

print(f"Geodesic Complexity: {geo_sol.geocomplex}")
print(f"Gate Sum Complexity: {gate_sum}")
```

## Output Data

The `GeoComplexity` function returns a `solution_object` containing various useful properties:
- `geocomplex`: The final calculated geodesic complexity.
- `complexHist`: History of complexity values during the calculation.
- `q_vals`: The penalty factor values used during the calculation.
- `time`: Time steps for the evolution.
- `H_coeff_data_p`: Pauli coefficients for the P space over time.
- `H_coeff_data_Q`: Pauli coefficients for the Q space over time.
- `U_norm_data`: Residual norm $||U(t) - U_{\text{target}}||$ over time.

For more detailed examples, including plotting and convergence testing, please refer to the `notebook/TestNotebook.ipynb` included in the repository.
