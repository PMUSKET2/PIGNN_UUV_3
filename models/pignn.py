"""
Physics-Informed Heterogeneous Graph Neural Network (PIGNN) v3 for BlueROV2.

Changes from v2:
    - forward() now outputs the continuous-time state derivative ẋ (shape
      (B, 9)) rather than the next state x̂(t+dt).  Integration is handled
      externally by an RK4 step in the notebook, matching the NODE baseline.

    - Removed from forward():
        * residual addition  (delta + state_in)
        * unit-circle projection on (cos ψ̂, sin ψ̂)   ← now lives in rk4_step
        * body→world rotation on position increments   ← now lives in f_body()

    - Added f_body() helper that applies the body→world rotation prior to
      the raw network output before returning ẋ.  This mirrors the NODE's
      f_theta() design: the network predicts derivatives in the body frame
      for (dx, dy), and the rotation into the world frame is an explicit
      physics prior, not something the network has to learn.

    Interface (unchanged externally):
        Input:  Z = [state(9) | control(4) | dt(1)]   shape (B, 14)
        Output: ẋ  (9D state derivative)               shape (B, 9)

        Units of ẋ:
            [0] dx/dt        m/s
            [1] dy/dt        m/s
            [2] dz/dt        m/s
            [3] d(cos ψ)/dt  1/s   (= -sin ψ · r,  enforced via rotation prior)
            [4] d(sin ψ)/dt  1/s   (=  cos ψ · r,  enforced via rotation prior)
            [5] du/dt        m/s²
            [6] dv/dt        m/s²
            [7] dw/dt        m/s²
            [8] dr/dt        rad/s²
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
# Full PIGNN model (v3 — derivative output with rotation prior)
# ---------------------------------------------------------------------------
class PIGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network for BlueROV2 dynamics.

    v3 changes:
        - forward() outputs ẋ (continuous-time derivative) not x̂(t+dt).
          Integration is handled externally by rk4_step() in the notebook.
        - Body→world rotation prior applied to (dx, dy) outputs, matching
          the NODE's f_theta() design.  The network predicts position
          derivatives in the body frame; rotation into the world frame is
          an explicit prior, not learned implicitly.
        - Residual connection, unit-circle projection, and in-forward
          body→world rotation all removed from forward() — these now live
          in the notebook's rk4_step() where they belong.

    Interface:
        Input:  Z = [state(9) | control(4) | dt(1)]  shape (B, 14)
        Output: ẋ                                      shape (B, 9)
    """

    def __init__(
        self,
        N_in:  int = 14,
        N_out: int = 9,
        hidden:     int = 64,
        msg_dim:    int = 64,
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
        """
        Compute the state derivative ẋ at (state, control, dt).

        The network predicts raw outputs in a mixed body/world frame.
        The body→world rotation prior is applied here as an explicit
        physics inductive bias: the network's (dx_body, dy_body) outputs
        are rotated into the world frame using the current heading
        (cos ψ, sin ψ) from the input state.  All other outputs (dz,
        d_cos_psi, d_sin_psi, du, dv, dw, dr) are returned as-is.

        Parameters
        ----------
        Z : (B, 14)  [state(9) | control(4) | dt(1)]

        Returns
        -------
        xdot : (B, 9)  state derivative ẋ in physical units
        """
        if Z.dim() == 1:
            Z = Z.unsqueeze(0)

        B      = Z.shape[0]
        device = Z.device

        # --- Build batch of graphs ---
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

        # Readout — raw 9D output, interpreted as ẋ in body/mixed frame
        hull_emb = batch["hull"].x
        raw      = self.readout(hull_emb)   # (B, 9)

        # --- Body→world rotation prior (mirrors NODE's f_theta) ---
        # The network outputs (dx_body, dy_body) — position derivatives in
        # the body frame.  We rotate them into the world frame using the
        # current heading from the input state.  This is an explicit
        # physics prior: the ROV moves in the direction it is pointing.
        cos_psi = Z[:, 3]   # cos ψ from input state
        sin_psi = Z[:, 4]   # sin ψ from input state

        dx_body = raw[:, 0]
        dy_body = raw[:, 1]
        dx_world = cos_psi * dx_body - sin_psi * dy_body
        dy_world = sin_psi * dx_body + cos_psi * dy_body

        # Assemble final derivative vector
        xdot = torch.cat([
            dx_world.unsqueeze(1),   # dx/dt  (world frame)
            dy_world.unsqueeze(1),   # dy/dt  (world frame)
            raw[:, 2:],              # dz/dt, d_cos_psi/dt, d_sin_psi/dt,
                                     # du/dt, dv/dt, dw/dt, dr/dt
        ], dim=1)

        return xdot
