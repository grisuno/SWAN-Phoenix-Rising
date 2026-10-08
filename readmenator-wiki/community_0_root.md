# root

*Community 0 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `FastGCNLayer`, `RigidSAE`, `SplineComplexityManager`, `SwanEllipticGNN_v51`, `__init__`, `compute_complexity`, `compute_psi_metrics`, `evolve_topology`. Core file: `app.py` (19 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 19 | yes |
| `install.sh` | sh | utility | 0 | no |

## Key Symbols

- `get_e8_lattice` (function, `app.py:38`) `def get_e8_lattice()`
- `RigidSAE` (class, `app.py:56`) `class RigidSAE(Module)` - Autoencoder Disperso (SAE) ajustado al estándar teórico:
- `__init__` (method, `app.py:64`) `def __init__(self, d_model, d_sae)`
- `forward` (method, `app.py:73`) `def forward(self, h)`
- `compute_psi_metrics` (method, `app.py:80`) `def compute_psi_metrics(self, z)` - Implementación de las Ecuaciones (6), (7) y (8) de la teoría.
- `get_sparsity_loss` (method, `app.py:107`) `def get_sparsity_loss(self, z)`
- `SplineComplexityManager` (class, `app.py:112`) `class SplineComplexityManager` - Medida de Complejidad Local (LC).
- `__init__` (method, `app.py:118`) `def __init__(self, threshold)`
- `compute_complexity` (method, `app.py:121`) `def compute_complexity(self, pre_acts_list)`
- `FastGCNLayer` (class, `app.py:134`) `class FastGCNLayer(Module)`
- `__init__` (method, `app.py:135`) `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- `forward` (method, `app.py:148`) `def forward(self, x)`
- `SwanEllipticGNN_v51` (class, `app.py:152`) `class SwanEllipticGNN_v51(Module)`
- `__init__` (method, `app.py:153`) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `evolve_topology` (method, `app.py:185`) `def evolve_topology(self, gap)`
- `forward` (method, `app.py:198`) `def forward(self, x, edge_index)`
- `load_elliptic_data` (method, `app.py:229`) `def load_elliptic_data()`
- `train_and_evaluate_v51` (method, `app.py:260`) `def train_and_evaluate_v51(X_raw, y, edge_index, train_idx, val_idx, epochs, nam`
- `temporal_cross_validate` (method, `app.py:372`) `def temporal_cross_validate(X_raw, y, edge_index, timestep)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `install.sh`
