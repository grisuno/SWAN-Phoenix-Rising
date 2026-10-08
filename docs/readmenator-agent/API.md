# API

## app.py
- `get_e8_lattice` (function) `app.py:38` `def get_e8_lattice()`
- `RigidSAE.__init__` (method) `app.py:64` `def __init__(self, d_model, d_sae)`
- `RigidSAE.forward` (method) `app.py:73` `def forward(self, h)`
- `RigidSAE.compute_psi_metrics` (method) `app.py:80` `def compute_psi_metrics(self, z)` -- Implementación de las Ecuaciones (6), (7) y (8) de la teoría.
- `RigidSAE.get_sparsity_loss` (method) `app.py:107` `def get_sparsity_loss(self, z)`
- `SplineComplexityManager.__init__` (method) `app.py:118` `def __init__(self, threshold)`
- `SplineComplexityManager.compute_complexity` (method) `app.py:121` `def compute_complexity(self, pre_acts_list)`
- `FastGCNLayer.__init__` (method) `app.py:135` `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- `FastGCNLayer.forward` (method) `app.py:148` `def forward(self, x)`
- `SwanEllipticGNN_v51.__init__` (method) `app.py:153` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `SwanEllipticGNN_v51.evolve_topology` (method) `app.py:185` `def evolve_topology(self, gap)`
- `SwanEllipticGNN_v51.forward` (method) `app.py:198` `def forward(self, x, edge_index)`
- `SwanEllipticGNN_v51.load_elliptic_data` (method) `app.py:229` `def load_elliptic_data()`
- `SwanEllipticGNN_v51.train_and_evaluate_v51` (method) `app.py:260` `def train_and_evaluate_v51(X_raw, y, edge_index, train_idx, val_idx, epochs, name)`
- `SwanEllipticGNN_v51.temporal_cross_validate` (method) `app.py:372` `def temporal_cross_validate(X_raw, y, edge_index, timestep)`
