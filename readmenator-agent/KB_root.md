# Subsystem: root

## app.py
- Layer: utility
- Doc: _*_ coding: utf8 _*_
- Language: py
- Symbols:
  - `get_e8_lattice` (function, line 38) `def get_e8_lattice()`
  - `RigidSAE` (class, line 56) `class RigidSAE(Module)`
  - `SplineComplexityManager` (class, line 112) `class SplineComplexityManager`
  - `FastGCNLayer` (class, line 134) `class FastGCNLayer(Module)`
  - `SwanEllipticGNN_v51` (class, line 152) `class SwanEllipticGNN_v51(Module)`
  - `load_elliptic_data` (method, line 229) `def load_elliptic_data()`
  - `train_and_evaluate_v51` (method, line 260) `def train_and_evaluate_v51(X_raw, y, edge_index, train_idx, val_idx, epochs, name)`
  - `temporal_cross_validate` (method, line 372) `def temporal_cross_validate(X_raw, y, edge_index, timestep)`
  - `__init__` (method, line 64) `def __init__(self, d_model, d_sae)`
  - `forward` (method, line 73) `def forward(self, h)`
  - `compute_psi_metrics` (method, line 80) `def compute_psi_metrics(self, z)`
  - `get_sparsity_loss` (method, line 107) `def get_sparsity_loss(self, z)`
  - `__init__` (method, line 118) `def __init__(self, threshold)`
  - `compute_complexity` (method, line 121) `def compute_complexity(self, pre_acts_list)`
  - `__init__` (method, line 135) `def __init__(self, in_features, out_features, edge_index, num_nodes)`
  - `forward` (method, line 148) `def forward(self, x)`
  - `__init__` (method, line 153) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
  - `evolve_topology` (method, line 185) `def evolve_topology(self, gap)`
  - `forward` (method, line 198) `def forward(self, x, edge_index)`

## install.sh
- Layer: utility
- Language: sh
