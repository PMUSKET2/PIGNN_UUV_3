"""
Physics-Informed Heterogeneous Graph Neural Network (PIGNN) v2 for BlueROV2.

Changes from v1:
    - Unit-circle projection: after computing (cos ψ̂, sin ψ̂), normalize
      back onto S¹ to keep the heading representation valid. This ensures
      the physics residual loss computes derivatives at valid states during
      multi-step rollout (addresses reviewer feedback on AD in non-Euclidean
      domains).
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch

from models.graph_builder import (
    build_graph, build_graph_batch, allocate_thrusts, NUM_THRUSTERS,
)
from src.parameters import THRUSTER_CONFIG


# ---------------------------------------------------------------------------
# Adaptive Softplus
# ---------------------------------------------------------------------------
class AdaptiveSoftplus(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.sp   = nn.Softplus()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return torch.reciprocal(self.beta) * self.sp(self.beta * x)


# ---------------------------------------------------------------------------
# MLP helper
# ---------------------------------------------------------------------------
def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2):
    layers = []
    layers.append(nn.Linear(in_dim, hidden))
    layers.append(AdaptiveSoftplus())
    layers.append(nn.LayerNorm(hidden))
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(AdaptiveSoftplus())
        layers.append(nn.LayerNorm(hidden))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Per-edge-type message functions
# ---------------------------------------------------------------------------
class ThrusterToHullConv(nn.Module):
    def __init__(self, thruster_dim, edge_dim, hull_dim, out_channels, hidden=32):
        super().__init__()
        self.mlp = _mlp(thruster_dim + edge_dim + hull_dim, hidden, out_channels)

    def forward(self, x_src, x_dst, edge_attr, edge_index):
        src, dst = edge_index
        msg_in = torch.cat([x_src[src], edge_attr, x_dst[dst]], dim=-1)
        return self.mlp(msg_in)


class HydroToHullConv(nn.Module):
    def __init__(self, hydro_dim, edge_dim, hull_dim, out_channels, hidden=32):
        super().__init__()
        self.mlp = _mlp(hydro_dim + edge_dim + hull_dim, hidden, out_channels)

    def forward(self, x_src, x_dst, edge_attr, edge_index):
        src, dst = edge_index
        msg_in = torch.cat([x_src[src], edge_attr, x_dst[dst]], dim=-1)
        return self.mlp(msg_in)


class BuoyToHullConv(nn.Module):
    def __init__(self, buoy_dim, edge_dim, hull_dim, out_channels, hidden=32):
        super().__init__()
        self.mlp = _mlp(buoy_dim + edge_dim + hull_dim, hidden, out_channels)

    def forward(self, x_src, x_dst, edge_attr, edge_index):
        src, dst = edge_index
        msg_in = torch.cat([x_src[src], edge_attr, x_dst[dst]], dim=-1)
        return self.mlp(msg_in)


# ---------------------------------------------------------------------------
# Heterogeneous message-passing layer
# ---------------------------------------------------------------------------
class PIGNNLayer(nn.Module):
    def __init__(self, node_dims, edge_dims, hidden=32, msg_dim=32):
        super().__init__()
        self.thruster_conv = ThrusterToHullConv(
            node_dims["thruster"], edge_dims[("thruster", "forces", "hull")],
            node_dims["hull"], msg_dim, hidden,
        )
        self.hydro_conv = HydroToHullConv(
            node_dims["hydrodynamic"], edge_dims[("hydrodynamic", "drag", "hull")],
            node_dims["hull"], msg_dim, hidden,
        )
        self.buoy_conv = BuoyToHullConv(
            node_dims["buoyancy"], edge_dims[("buoyancy", "restoring", "hull")],
            node_dims["hull"], msg_dim, hidden,
        )
        self.hull_update = _mlp(
            node_dims["hull"] + 3 * msg_dim, hidden, node_dims["hull"], n_layers=2,
        )

    def forward(self, data: HeteroData) -> HeteroData:
        ei_t = data["thruster", "forces", "hull"].edge_index
        ea_t = data["thruster", "forces", "hull"].edge_attr
        msg_t = self.thruster_conv(data["thruster"].x, data["hull"].x, ea_t, ei_t)
        num_hull = data["hull"].x.size(0)
        agg_t = torch.zeros(num_hull, msg_t.size(-1),
                            device=msg_t.device, dtype=msg_t.dtype)
        agg_t.scatter_add_(0, ei_t[1].unsqueeze(-1).expand_as(msg_t), msg_t)

        ei_h = data["hydrodynamic", "drag", "hull"].edge_index
        ea_h = data["hydrodynamic", "drag", "hull"].edge_attr
        agg_h = self.hydro_conv(data["hydrodynamic"].x, data["hull"].x, ea_h, ei_h)

        ei_b = data["buoyancy", "restoring", "hull"].edge_index
        ea_b = data["buoyancy", "restoring", "hull"].edge_attr
        agg_b = self.buoy_conv(data["buoyancy"].x, data["hull"].x, ea_b, ei_b)

        hull_in = torch.cat([data["hull"].x, agg_t, agg_h, agg_b], dim=-1)
        data["hull"].x = self.hull_update(hull_in)
        return data


# ---------------------------------------------------------------------------
# Full PIGNN model (v2 — with unit-circle projection)
# ---------------------------------------------------------------------------
class PIGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network for BlueROV2 dynamics.

    v2 changes:
        - Unit-circle projection on (cos ψ̂, sin ψ̂) after residual addition.
          Ensures the heading stays on S¹ during autoregressive rollout,
          making the physics residual loss valid at every rollout step.

    Interface:
        Input:  Z = [state(9) | control(4) | time(1)]  shape (B, 14)
        Output: x̂(t+dt)                                 shape (B, 9)
    """

    def __init__(
        self,
        N_in:  int = 14,
        N_out: int = 9,
        hidden:     int = 32,
        msg_dim:    int = 32,
        n_mp_layers: int = 2,
    ):
        super().__init__()
        self.N_in  = N_in
        self.N_out = N_out

        hull_enc_dim = hidden
        self.node_dims = {
            "hull": hull_enc_dim, "thruster": 8,
            "hydrodynamic": 4,    "buoyancy": 6,
        }
        self.edge_dims = {
            ("thruster",     "forces",    "hull"): 7,
            ("hydrodynamic", "drag",      "hull"): 8,
            ("buoyancy",     "restoring", "hull"): 4,
        }

        self.hull_encoder = _mlp(N_in, hidden, hull_enc_dim)

        self.mp_layers = nn.ModuleList([
            PIGNNLayer(self.node_dims, self.edge_dims, hidden, msg_dim)
            for _ in range(n_mp_layers)
        ])

        self.readout = _mlp(hull_enc_dim, hidden, N_out, n_layers=3)
        self._init_weights()

    def _init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if Z.dim() == 1:
            Z = Z.unsqueeze(0)

        B = Z.shape[0]
        device = Z.device

        # --- Build batch of graphs (vectorized) ---
        states = Z[:, :9].detach()
        taus   = Z[:, 9:13].detach()
        graphs = build_graph_batch(states, taus)

        # Encode hull nodes
        hull_enc = self.hull_encoder(Z)
        for b in range(B):
            graphs[b]["hull"].x = hull_enc[b].unsqueeze(0)

        batch = Batch.from_data_list(graphs)

        # Message passing
        for mp_layer in self.mp_layers:
            batch = mp_layer(batch)

        # Readout
        hull_emb = batch["hull"].x
        delta    = self.readout(hull_emb)

        # Residual connection
        state_in = Z[:, :self.N_out]

        cos_psi_hat = delta[:, 3] + state_in[:, 3]
        sin_psi_hat = delta[:, 4] + state_in[:, 4]

        # === Unit-circle projection (v2) ===
        # Project (cos ψ̂, sin ψ̂) back onto S¹ so that the heading
        # representation remains valid during autoregressive rollout.
        # This ensures the physics residual loss computes derivatives
        # at states that correspond to actual heading angles.
        psi_norm = torch.sqrt(cos_psi_hat ** 2 + sin_psi_hat ** 2 + 1e-8)
        cos_psi_hat = cos_psi_hat / psi_norm
        sin_psi_hat = sin_psi_hat / psi_norm

        # Body→world rotation for position increments
        x_world = cos_psi_hat * delta[:, 0] - sin_psi_hat * delta[:, 1] + state_in[:, 0]
        y_world = sin_psi_hat * delta[:, 0] + cos_psi_hat * delta[:, 1] + state_in[:, 1]

        X_hat = delta + state_in
        X_hat = X_hat.clone()
        X_hat[:, 0] = x_world
        X_hat[:, 1] = y_world
        X_hat[:, 3] = cos_psi_hat  # overwrite with projected values
        X_hat[:, 4] = sin_psi_hat

        return X_hat
