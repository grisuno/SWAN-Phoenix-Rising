#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: xx/xx/xxxx
Licencia: GPL v3

Descripción:  
"""
import os
import glob
import torch
import zipfile
import kagglehub
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops, degree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import time
import warnings
import itertools
import math
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# I. GEOMETRÍA E8
# =============================================================================

def get_e8_lattice():
    verts = []
    for i in range(8):
        for j in range(i+1, 8):
            for s1, s2 in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                v = np.zeros(8); v[i], v[j] = s1, s2
                verts.append(v)
    for signs in itertools.product([-1,1], repeat=8):
        if sum(1 for s in signs if s==-1)%2==0:
            verts.append(0.5 * np.array(signs))
    return torch.tensor(np.array(verts), dtype=torch.float32).to(DEVICE)

E8_VERTICES = get_e8_lattice()

# =============================================================================
# II. MARCO TEÓRICO RIGUROSO CORREGIDO
# =============================================================================

class RigidSAE(nn.Module):
    """
    Autoencoder Disperso (SAE) ajustado al estándar teórico:
    - Pesos Atados (Tied Weights)
    - Sin sesgo en decodificador
    - Medición de Psi y F efectivas mediante Entropía de Shannon.
    - CORRECCIÓN: Incluye método get_sparsity_loss para el mecanismo Phoenix.
    """
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        # Solo codificador. El decodificador usa la transpuesta.
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))

    def forward(self, h: torch.Tensor):
        # Ecuación (3) Codificación: z = ReLU(Wx + b)
        z = F.relu(h @ self.W_enc + self.b_enc)
        # Ecuación (4) Reconstrucción (Tied): x' = z @ W_enc.T
        x_recon = z @ self.W_enc.t()
        return x_recon, z

    def compute_psi_metrics(self, z: torch.Tensor):
        """
        Implementación de las Ecuaciones (6), (7) y (8) de la teoría.
        Calcula p_i, H(p), F y Psi.
        """
        with torch.no_grad():
            # Ecuación (6): Probabilidad de características (p_i)
            # p_i = suma(activacion_i) / presupuesto_total
            f_act = z.abs().sum(dim=0) # Suma sobre el batch (S)
            total_budget = f_act.sum() + 1e-12
            p_i = f_act / total_budget
            
            # Filtramos p_i válidos para logaritmos
            p_i_safe = p_i[p_i > 1e-10]
            
            # Entropía de Shannon H(p)
            h_p = -torch.sum(p_i_safe * torch.log(p_i_safe))
            
            # Ecuación (7): Características Efectivas (F)
            f_eff = torch.exp(h_p)
            
            # Ecuación (8): Medida de Superposición (Psi)
            psi = f_eff / self.d_model
            
        return {"f_eff": f_eff.item(), "psi": psi.item()}

    # --- CORRECCIÓN DEL ERROR ATTRIBUTEERROR ---
    def get_sparsity_loss(self, z):
        # Penalización L1 (Ecuación 5 parte 2)
        # Necesaria para el cálculo de la pérdida en el training loop
        return z.norm(p=1, dim=1).mean()

class SplineComplexityManager:
    """
    Medida de Complejidad Local (LC).
    Cuenta la intersección de hiperplanos (Ecuación 6) en una región local.
    Aproximación: Neuronas con pre-activación cercana a 0.
    """
    def __init__(self, threshold: float = 0.08):
        self.threshold = threshold

    def compute_complexity(self, pre_acts_list):
        intersections = 0
        for h in pre_acts_list:
            # Una neurona intersecta la región si está cerca de la no-linealidad
            # Esto aproxima N = |{i: Hi ∩ Φ != ∅}|
            intersect = (h.abs() < self.threshold).float().sum(dim=1).mean().item()
            intersections += intersect
        return intersections / len(pre_acts_list)

# =============================================================================
# III. ARQUITECTURA SWAN ELLIPTIC v5.1 (PHOENIX RISE)
# =============================================================================

class FastGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, edge_index, num_nodes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
        self.register_buffer('adj', adj)
        
    def forward(self, x):
        x = torch.sparse.mm(self.adj, x)
        return F.linear(x, self.weight)

class SwanEllipticGNN_v51(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. EMBEDDING TOPOLÓGICO E8
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.register_buffer("topo_mask", torch.ones_like(self.embed.weight))
        self.e8_proj = nn.Parameter(torch.randn(input_dim, 8))
        nn.init.orthogonal_(self.e8_proj)
        
        # 2. BACKBONE GNN
        self.gcn = FastGCNLayer(hidden_dim, hidden_dim, edge_index, num_nodes)
        self.gat = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        
        # 3. SAE RIGUROSO (Tied Weights, No Decoder Bias)
        self.sae = RigidSAE(hidden_dim, hidden_dim * 4)
        
        # 4. COMPLEJIDAD LOCAL (SPLINE)
        self.spline_manager = SplineComplexityManager()
        
        # 5. CABEZAL
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = dropout

    def evolve_topology(self, gap):
        w = self.embed.weight.data
        thresh = 0.05 if gap > 15.0 else 0.01 
        self.topo_mask[:] = (torch.abs(w) > thresh).float()
        dead = (self.topo_mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        if len(dead) > 0:
            idx = torch.randint(0, 240, (len(dead),), device=DEVICE)
            new_dna = E8_VERTICES[idx] @ self.e8_proj.T
            self.embed.weight.data[dead] = new_dna * 0.1
            self.topo_mask[dead] = 1.0
            return len(dead)
        return 0

    def forward(self, x, edge_index):
        # --- Embedding ---
        mask = self.topo_mask
        if self.training:
            mask = mask * (torch.rand_like(mask) > 0.1).float()
        
        z_embed = F.linear(x, self.embed.weight * mask, self.embed.bias)
        h_embed = torch.tanh(z_embed)
        
        # --- GNN ---
        res = h_embed
        x_gcn = self.gcn(h_embed)
        x_gat = self.gat(h_embed, edge_index)
        
        x_fused = self.fusion(torch.cat([x_gcn, x_gat], dim=-1))
        z_fused = x_fused # Pre-activación para LC
        
        x_out = F.relu(self.norm(x_fused + res))
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        
        # --- SAE ---
        h_recon, z_sae = self.sae(x_out)
        
        logits = torch.sigmoid(self.readout(x_out))
        
        return logits, h_recon, z_sae, z_embed, z_fused, x_out

# =============================================================================
# IV. DATOS Y PIPELINE
# =============================================================================

def load_elliptic_data():
    print("📥 Descargando Dataset...")
    path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
    zip_files = glob.glob(os.path.join(path, "*.zip"))
    if zip_files:
        with zipfile.ZipFile(zip_files[0], "r") as z:
            z.extractall(path)
    base_dir = os.path.dirname(glob.glob(os.path.join(path, "**", "elliptic_txs_features.csv"), recursive=True)[0])
    df_nodes = pd.read_csv(os.path.join(base_dir, "elliptic_txs_features.csv"), header=None)
    df_classes = pd.read_csv(os.path.join(base_dir, "elliptic_txs_classes.csv"))
    df_edges = pd.read_csv(os.path.join(base_dir, "elliptic_txs_edgelist.csv"))

    df_nodes.columns = ["txId", "timestep"] + [f"feat_{i}" for i in range(1, 166)]
    df = df_nodes.merge(df_classes, on="txId", how="left")
    df = df[df["class"].isin(["1", "2"])]
    df["class"] = df["class"].map({"1": 1, "2": 0}).astype(int)

    timestep = df["timestep"].values
    X_raw = df.iloc[:, 2:-1].values 
    y = df["class"].values

    tx_to_idx = {tx: i for i, tx in enumerate(df["txId"])}
    edge_df = df_edges[df_edges["txId1"].isin(tx_to_idx) & df_edges["txId2"].isin(tx_to_idx)]
    edge_index = torch.tensor([[tx_to_idx[tx] for tx in edge_df["txId1"]],
                              [tx_to_idx[tx] for tx in edge_df["txId2"]]], dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(X_raw))
    edge_index = edge_index.to(DEVICE)
    y = torch.tensor(y, dtype=torch.long).to(DEVICE)
    print(f"✅ Datos cargados: {len(X_raw)} nodos.")
    return X_raw, y, edge_index, timestep

def train_and_evaluate_v51(X_raw, y, edge_index, train_idx, val_idx, epochs=1500, name="Model"):
    # --- ESCALADO TEMPORAL CORRECTO ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_raw[train_idx.cpu().numpy()])
    X_all_scaled = scaler.transform(X_raw)
    X = torch.tensor(X_all_scaled, dtype=torch.float32).to(DEVICE)

    # --- SETUP MODELO ---
    num_nodes = X.shape[0]
    model = SwanEllipticGNN_v51(X.shape[1], 128, edge_index, num_nodes).to(DEVICE)
    
    y_train_subset = y[train_idx]
    num_pos = y_train_subset.sum().item()
    num_neg = len(y_train_subset) - num_pos
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(DEVICE))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.15)
    
    best_auprc = 0
    start_time = time.time()
    
    print(f"    ▶ Ciclo {name} | Pos_W: {pos_weight_val:.2f}")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # --- FORWARD ---
        logits, h_recon, z_sae, z_embed, z_fused, x_out = model(X, edge_index)
        
        # --- MÉTRICAS DE ORO ---
        metrics_psi = model.sae.compute_psi_metrics(z_sae[train_idx])
        psi_val = metrics_psi['psi']
        f_eff_val = metrics_psi['f_eff']
        
        lc_n = model.spline_manager.compute_complexity([z_embed[train_idx], z_fused[train_idx]])
        
        # --- PÉRDIDA OMNIMODA v5.1 (PHOENIX MECHANISM) ---
        
        cls_loss = criterion(logits[train_idx].squeeze(), y[train_idx].float())
        
        # Reconstrucción Aumentada (Crucial para mantener Psi alto)
        loss_sae = 0.1 * F.mse_loss(h_recon[train_idx], x_out[train_idx])
        
        # MECANISMO PHOENIX: L1 ADAPTATIVO
        # Si Psi es muy bajo (< 0.5), RELAJAMOS la penalización de dispersión para resucitar neuronas
        base_l1_coeff = 1e-5 # Reducido de 1e-4
        if psi_val < 0.5:
            # Modo Resurrección: Permitir activación
            current_l1_coeff = 1e-6
        else:
            # Modo Normal: Esparsificar suavemente
            current_l1_coeff = base_l1_coeff
            
        # USO DEL MÉTODO CORREGIDO get_sparsity_loss
        loss_l1 = current_l1_coeff * model.sae.get_sparsity_loss(z_sae[train_idx])
        
        # Penalización de Psi (Consolidación Algorítmica)
        loss_psi = 0.5 * (psi_val - 1.2)**2
        
        # Penalización de Complejidad Local (Grokking)
        loss_lc = 0.5 * (lc_n / 10.0)**2 
        
        # Ortogonalidad
        w_embed = model.embed.weight
        loss_ortho = 1e-4 * torch.norm(w_embed @ w_embed.t() - torch.eye(w_embed.shape[0]).to(DEVICE))
        
        total_loss = cls_loss + loss_sae + loss_l1 + loss_psi + loss_lc + loss_ortho
        
        # --- BACKWARD ---
        total_loss.backward()
        with torch.no_grad():
            model.embed.weight.grad *= model.topo_mask
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # --- VALIDACIÓN ---
        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_logits, _, _, z_e_val, z_f_val, _ = model(X, edge_index)
                
                val_preds = val_logits[val_idx].cpu().numpy()
                val_y = y[val_idx].cpu().numpy()
                
                auprc = average_precision_score(val_y, val_preds)
                
                lc_val = model.spline_manager.compute_complexity([z_e_val[val_idx], z_f_val[val_idx]])
                
                elapsed = time.time() - start_time
                # Imprimimos Psi y F para verificar la teoría
                mode_str = "[PHX]" if psi_val < 0.5 else "[STD]"
                print(f"      [Ep {epoch+1:3d}] AUPRC: {auprc:.4f} | ψ: {psi_val:.3f} | F: {f_eff_val:.1f} | LC: {lc_val:.3f} | {mode_str} | Time: {elapsed:.1f}s")
                
                if auprc > best_auprc:
                    best_auprc = auprc

        # Evolución Topológica
        if epoch > 0 and epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                t_logits, _, _, _, _, _ = model(X, edge_index)
                train_preds = (t_logits[train_idx] > 0.5).float().cpu().numpy()
                val_preds_check = (val_logits[val_idx] > 0.5).float().cpu().numpy()
                t_acc = (train_preds.squeeze() == y[train_idx].cpu().numpy()).mean() * 100
                ood_acc = (val_preds_check.squeeze() == val_y).mean() * 100
                model.evolve_topology(t_acc - ood_acc)
                model.train()

    return best_auprc

def temporal_cross_validate(X_raw, y, edge_index, timestep):
    unique_ts = np.sort(np.unique(timestep))
    auprc_scores = []
    print(f"  Iniciando CV Temporal con Mecanismo Phoenix (L1 Adaptativo)...")
    
    for t in unique_ts[3:-1]:
        train_idx = np.where(timestep < t)[0]
        val_idx   = np.where(timestep == t)[0]

        if len(val_idx) < 50 or len(train_idx) < 200:
            continue

        train_idx = torch.tensor(train_idx)
        val_idx   = torch.tensor(val_idx)

        auprc = train_and_evaluate_v51(
            X_raw, y, edge_index, train_idx, val_idx,
            epochs=1500, name=f"T={t:02d}"
        )
        
        auprc_scores.append(auprc)
        print(f"    ✅ Final AUPRC: {auprc:.4f}\n")

    return np.mean(auprc_scores)

if __name__ == "__main__":
    print("🚀 SWAN v5.1: Phoenix Rise (Adaptive L1 to fix Psi Death) - FIXED")
    print("=" * 100)
    
    X_raw, y, edge_index, timestep = load_elliptic_data()
    
    start_global = time.time()
    mean_auprc = temporal_cross_validate(X_raw, y, edge_index, timestep)
    total_time = time.time() - start_global
    
    print("\n" + "=" * 100)
    print("🏆 RESULTADO FINAL")
    print("=" * 100)
    print(f"Mean AUPRC: {mean_auprc:.4f}")
    print(f"Tiempo: {total_time/60:.1f} min")
    print("\n✅ Se ha implementado el mecanismo Phoenix para evitar el colapso de Superposición en Concept Drift.")
