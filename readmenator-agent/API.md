# API

## app.py

### get_e8_lattice `def get_e8_lattice()`
- Defined: `app.py:38`

### load_elliptic_data `def load_elliptic_data()`
- Defined: `app.py:229`

### train_and_evaluate_v51 `def train_and_evaluate_v51(X_raw, y, edge_index, train_idx, val_idx, epochs, name)`
- Defined: `app.py:260`

### temporal_cross_validate `def temporal_cross_validate(X_raw, y, edge_index, timestep)`
- Defined: `app.py:372`

### __init__ `def __init__(self, d_model, d_sae)`
- Defined: `app.py:64`

### forward `def forward(self, h)`
- Defined: `app.py:73`

### compute_psi_metrics `def compute_psi_metrics(self, z)`
- Defined: `app.py:80`
- Doc: Implementación de las Ecuaciones (6), (7) y (8) de la teoría.

### get_sparsity_loss `def get_sparsity_loss(self, z)`
- Defined: `app.py:107`

### __init__ `def __init__(self, threshold)`
- Defined: `app.py:118`

### compute_complexity `def compute_complexity(self, pre_acts_list)`
- Defined: `app.py:121`

### __init__ `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- Defined: `app.py:135`

### forward `def forward(self, x)`
- Defined: `app.py:148`

### __init__ `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- Defined: `app.py:153`

### evolve_topology `def evolve_topology(self, gap)`
- Defined: `app.py:185`

### forward `def forward(self, x, edge_index)`
- Defined: `app.py:198`
