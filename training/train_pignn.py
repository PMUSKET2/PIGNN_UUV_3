"""
Curriculum training script for the PIGNN v2.

Changes from v1:
    - Per-phase learning rates (lower LR for rollout phases)
    - Per-state loss weighting (computed from training data)
    - State weights passed through to all loss functions
"""

import os
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.pignn import PIGNN
from models.model_utility import (
    get_data_sets,
    convert_input_data,
    train,
    test_dev_set,
    get_state_weights,
)

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE      = 3
LR_FACTOR       = 0.5
LR_MIN          = 1e-5
NOISE_LEVEL     = 0.0
GRADIENT_METHOD = "direct"
HIDDEN          = 32
MSG_DIM         = 32
N_MP_LAYERS     = 2
EXP_NAME        = "pignn_v2_curriculum"
ALPHA_SMOOTH    = 0.6

# ---------------------------------------------------------------------------
# Curriculum phases — v2: per-phase LR
# ---------------------------------------------------------------------------
PHASES = [
    {
        "name":       "phase1_data_physics",
        "pinn":       True,
        "rollout":    False,
        "n_roll":     0,
        "max_epochs": 400,
        "patience":   60,
        "lr_patience": 30,
        "lr_init":    8e-3,       # standard LR for phase 1
    },
    {
        "name":       "phase2_rollout_short",
        "pinn":       True,
        "rollout":    True,
        "n_roll":     5,
        "max_epochs": 300,
        "patience":   50,
        "lr_patience": 25,
        "lr_init":    2e-3,       # reduced LR for rollout stability
    },
    {
        "name":       "phase3_rollout_long",
        "pinn":       True,
        "rollout":    True,
        "n_roll":     15,
        "max_epochs": 200,
        "patience":   40,
        "lr_patience": 20,
        "lr_init":    1e-3,       # further reduced for long rollout
    },
]


def run_phase(
    phase_cfg: dict,
    model: PIGNN,
    train_loader,
    dev_loader,
    writer: SummaryWriter,
    model_dir: str,
    global_epoch: int,
    device,
    state_weights: torch.Tensor = None,
):
    name       = phase_cfg["name"]
    pinn       = phase_cfg["pinn"]
    rollout    = phase_cfg["rollout"]
    n_roll     = phase_cfg["n_roll"]
    max_epochs = phase_cfg["max_epochs"]
    patience   = phase_cfg["patience"]
    lr_init    = phase_cfg["lr_init"]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  PINN={pinn}  rollout={rollout}  n_roll={n_roll}  lr={lr_init}")
    print(f"  max_epochs={max_epochs}  patience={patience}")
    print(f"{'='*60}\n")

    optimizer = AdamW(model.parameters(), lr=lr_init)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=LR_FACTOR,
        patience=phase_cfg["lr_patience"],
        threshold=1e-4,
        min_lr=LR_MIN,
    )

    best_dev      = float("inf")
    best_state    = None
    epochs_no_imp = 0
    l_dev_smooth  = 1.0

    for ep in trange(max_epochs, desc=name):
        l_train = train(
            model, train_loader, optimizer, global_epoch, device, writer,
            pinn=pinn, rollout=rollout,
            noise_level=NOISE_LEVEL,
            gradient_method=GRADIENT_METHOD,
            n_roll=n_roll,
            state_weights=state_weights,
        )

        l_dev = test_dev_set(model, dev_loader, global_epoch, device, writer)

        l_dev_smooth = ALPHA_SMOOTH * l_dev_smooth + (1 - ALPHA_SMOOTH) * l_dev
        scheduler.step(l_dev_smooth)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_epoch)
        writer.add_scalar("phase", PHASES.index(phase_cfg), global_epoch)

        if l_dev < best_dev:
            best_dev   = l_dev
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_imp = 0

            save_path = os.path.join(model_dir, f"{name}_best_epoch_{global_epoch}")
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_imp += 1

        writer.flush()
        global_epoch += 1

        if epochs_no_imp >= patience:
            print(f"\n  Early stop at epoch {ep} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    phase_end_path = os.path.join(model_dir, f"{name}_final")
    torch.save(model.state_dict(), phase_end_path)
    print(f"  Phase complete — best dev loss: {best_dev:.6f}")

    return global_epoch, best_dev


def main():
    train_loader, dev_loader, _, _ = get_data_sets(BATCH_SIZE)
    X0, U0, tc0, t0 = next(iter(train_loader))
    N_x  = X0.shape[-1]
    N_u  = U0.shape[-1]
    N_in = N_x + N_u + 1

    print(f"State dim:   {N_x}")
    print(f"Control dim: {N_u}")
    print(f"Input dim:   {N_in}")
    print(f"Device:      {device}")

    # Compute per-state weights from training data
    from data.data_utility import TrajectoryDataset
    train_ds = TrajectoryDataset("training_set")
    state_weights = get_state_weights(train_ds.X, device=device)
    print(f"State weights: {state_weights.cpu().numpy()}")

    model = PIGNN(
        N_in=N_in, N_out=N_x,
        hidden=HIDDEN, msg_dim=MSG_DIM, n_mp_layers=N_MP_LAYERS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters:  {n_params:,}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{EXP_NAME}_{timestamp}")
    model_dir = "models_saved"
    os.makedirs(model_dir, exist_ok=True)

    global_epoch = 0
    phase_results = []

    try:
        for phase_cfg in PHASES:
            global_epoch, best_dev = run_phase(
                phase_cfg, model, train_loader, dev_loader,
                writer, model_dir, global_epoch, device,
                state_weights=state_weights,
            )
            phase_results.append((phase_cfg["name"], best_dev, global_epoch))

        final_path = os.path.join(model_dir, f"{EXP_NAME}_final")
        torch.save(model.state_dict(), final_path)

        print(f"\n{'='*60}")
        print("  Training complete — summary")
        print(f"{'='*60}")
        for name, dev, ep in phase_results:
            print(f"  {name:30s}  best_dev={dev:.6f}  ended_at_epoch={ep}")
        print(f"  Total epochs: {global_epoch}")
        print(f"  Final model:  {final_path}")

        writer.add_hparams(
            {
                "hidden": HIDDEN, "msg_dim": MSG_DIM, "n_mp_layers": N_MP_LAYERS,
                "batch_size": BATCH_SIZE, "total_epochs": global_epoch,
                "phases": len(PHASES), "state_weighted": True,
            },
            {"final_dev_loss": phase_results[-1][1]},
        )
        writer.close()

    except Exception as e:
        print(f"\nTraining error: {e}")
        writer.close()
        raise


if __name__ == "__main__":
    main()
