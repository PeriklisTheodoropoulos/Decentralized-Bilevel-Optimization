import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torchvision.datasets as datasets
from torchvision import transforms
import random
import copy
import time
from tqdm import trange
import os
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from help_functions import split_node_data_non_iid, generate_graph, \
                           compute_pairwise_disagreement, compute_consensus_disagreement, \
                           compute_midpoint_grads, compute_stochastic_grad

from tqdm import trange
import itertools
from model import CNNFashion_Mnist
from torch.utils.data import Subset

def run_deviabo(
    train_loaders,
    val_loaders,
    test_loader,
    K_NODES,
    num_classes,
    graph_type    = 'fully connected',
    eta_x         = 1e-2,
    eta_y         = 1e-2,
    gamma         = 0.5,
    beta          = 0.5,
    T             = 500,
    seed          = 42,
    model_factory = None,
    log_every     = 10,
    verbose       = True,
    device        = None,
):
    """
    DeViABO with DIVERSITY-SEEKING aggregation (self-loops included in N_i).
    Adapted for FashionMNIST (CNN via model_factory, unsqueeze channel dim).

    Mathematical Formulation:
    ─────────────────────────
    Lower-level update (diversity-seeking aggregation weights):
        For each neighbor l ∈ N_i (including self):
            ℓ_i^{val}(y_l^t) = validation loss of model y_l on node i's val data

        x_i^{t+1} = Π_{X_i}(
            x_i^t + η_x · [ℓ_i^{val}(y_l^t)]_{l∈N_i} - η_x · 2β(x_i - 1/|N_i|)
        )

    Upper-level update (model parameters):
        y_i^{t+1} = y_i^t
                    - η_y · ∇_{y_i} f_i^{tr}(y_i^t ; ζ_i^t)
                    - η_y · (γ/2) · Σ_{l ∈ N_i, l≠i} x_{i,l}^{t+1} · (y_i^t - y_l^t)

    New additions vs original:
    ──────────────────────────
    • grad_norm_y_history : mean over nodes of ‖∇_{y_i} f_i^{tr}‖ at each log step
    • grad_norm_x_history : mean over nodes of ‖grad_x_i‖ (val-loss gradient used
                            in the x-update) at each log step
    • node_val_loss_history: per-node aggregated val loss at each log step,
                             shape [num_log_steps][K_NODES]
    All three keys are required by Diagnostic 5 and 6 in the tuning script.
    """

    assert model_factory is not None, \
        "Provide a model_factory callable, e.g. lambda: CNNFashion_Mnist(...)"

    # ════════════════════════════════════════════════════════════════
    # SETUP: Device, Reproducibility, Graph, Models
    # ════════════════════════════════════════════════════════════════

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if verbose:
        print(f"[DeViABO] Using device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Graph construction: G = (V, E)
    # Neighborhoods N_i include self-loop
    G = generate_graph(graph_type, K_NODES, seed=seed)
    neighborhoods = {
        i: sorted(set(G.neighbors(i)) | {i})
        for i in range(K_NODES)
    }

    if verbose:
        print(f"\n[DeViABO] graph={graph_type}, K={K_NODES}, "
              f"η_x={eta_x}, η_y={eta_y}, γ={gamma}, β={beta}, T={T}")
        print("Neighborhoods N_i (including self-loop):")
        for i, nb in neighborhoods.items():
            print(f"  Node {i}: N_{i} = {nb},  |N_{i}| = {len(nb)}")
    for i, nb in neighborhoods.items():
            print(f"  Node {i}: N_{i} = {nb},  |N_{i}| = {len(nb)}")
    # Model initialization — each node gets a different seed for diverse init
    models = []
    for k in range(K_NODES):
        torch.manual_seed(seed + k)
        models.append(model_factory().to(device))
    torch.manual_seed(seed)

    criterion = nn.CrossEntropyLoss()

    # Aggregation weights x_i ∈ Δ^{|N_i|-1}, initialized uniformly
    X = {
        i: torch.full(
            (len(neighborhoods[i]),),
            1.0 / len(neighborhoods[i]),
            device=device,
        )
        for i in range(K_NODES)
    }

    # Simplex projection: Π_{X_i}
    def simplex_projection(v):
        n    = v.shape[0]
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0)
        rho  = (
            u > (cssv - 1.0) / torch.arange(1, n + 1, dtype=v.dtype, device=v.device)
        ).nonzero()[-1].item()
        lam = (cssv[rho] - 1.0) / (rho + 1.0)
        return torch.clamp(v - lam, min=0.0)

    # Persistent data iterators
    train_iterators = [iter(train_loaders[i]) for i in range(K_NODES)]
    val_iterators   = [iter(val_loaders[i])   for i in range(K_NODES)]

    def get_batch(iterators, loaders, i):
        """Get next mini-batch; cycles loader automatically when exhausted."""
        try:
            xb, yb = next(iterators[i])
        except StopIteration:
            iterators[i] = iter(loaders[i])
            xb, yb = next(iterators[i])
        # FashionMNIST: [B,28,28] → [B,1,28,28]
        return xb.unsqueeze(1).to(device), yb.to(device)

    # ── Tracking metrics ─────────────────────────────────────────
    log_iters                      = []
    train_loss_history             = []
    val_loss_history               = []
    test_loss_history              = []
    test_acc_history               = []
    consensus_disagreement_history = []
    pairwise_disagreement_history  = []
    X_history                      = {i: [] for i in range(K_NODES)}

    # NEW: gradient norm and per-node val loss histories
    grad_norm_y_history            = []   # mean ‖∇_y f_i^{tr}‖ over nodes
    grad_norm_x_history            = []   # mean ‖grad_x_i‖ over nodes
    node_val_loss_history          = []   # [num_log_steps][K_NODES]

    # ── Per-iteration accumulators (reset each iteration) ────────
    # We accumulate grad norms every iteration and average at log steps
    _gnorm_y_buf = []   # one float per node per iteration
    _gnorm_x_buf = []   # one float per node per iteration

    # ════════════════════════════════════════════════════════════════
    # MAIN TRAINING LOOP
    # ════════════════════════════════════════════════════════════════
    pbar = trange(T, desc=f"[DeViABO | {graph_type} | K={K_NODES}]",
                  unit="iter", leave=True)

    for t in pbar:

        # Snapshot all parameters y_i^t before any updates
        params_snapshot = [
            [p.detach().clone() for p in models[i].parameters()]
            for i in range(K_NODES)
        ]

        # Per-iteration norm accumulators
        iter_gnorm_y = []
        iter_gnorm_x = []

        for i in range(K_NODES):
            nb = neighborhoods[i]

            X_val,   y_val   = get_batch(val_iterators,   val_loaders,   i)
            X_train, y_train = get_batch(train_iterators, train_loaders, i)

            # ════════════════════════════════════════════════════════
            # LOWER-LEVEL UPDATE: Learn aggregation weights x_i
            # Assign HIGH weight to DISSIMILAR / DIVERSE neighbors
            # ════════════════════════════════════════════════════════

            # Evaluate individual validation loss ℓ_i^{val}(y_l^t) for each l ∈ N_i
            individual_val_losses = torch.zeros(len(nb), device=device)

            for k, l in enumerate(nb):
                temp_model = model_factory().to(device)
                with torch.no_grad():
                    for p_idx, temp_param in enumerate(temp_model.parameters()):
                        temp_param.copy_(params_snapshot[l][p_idx])
                    logits = temp_model(X_val)
                    individual_val_losses[k] = criterion(logits, y_val).item()

            # Diversity-seeking gradient: high loss → increase weight
            grad_x = individual_val_losses

            # ── NEW: record ‖grad_x_i‖ ───────────────────────────
            iter_gnorm_x.append(grad_x.norm().item())

            # Regularization gradient: 2β(x_i - 1/|N_i|)
            with torch.no_grad():
                uniform_weight = 1.0 / len(nb)
                grad_beta = 2.0 * beta * (X[i] - uniform_weight)

            # Gradient ASCENT + simplex projection
            with torch.no_grad():
                X[i] = simplex_projection(
                    X[i] + eta_x * grad_x - eta_x * grad_beta
                )

            # uniform_weight = 1.0 / len(nb)
            # X[i] = torch.full((len(nb),), uniform_weight, device=device)
            # print(X[i].sum().item())

            # for i, nb in neighborhoods.items():
            # print(f"  Node {i}: N_{i} = {nb},  |N_{i}| = {len(nb)}")
            print(f"  Node {i}: N_{i} = {nb}, {X[i]}")

            # ════════════════════════════════════════════════════════
            # UPPER-LEVEL UPDATE: Model parameters y_i
            # ════════════════════════════════════════════════════════

            # Training loss gradient ∇_{y_i} f_i^{tr}(y_i^t)
            models[i].zero_grad()
            with torch.enable_grad():
                loss_tr = criterion(models[i](X_train), y_train)
                loss_tr.backward()

            # ── NEW: record ‖∇_{y_i} f_i^{tr}‖ after backward() ──
            gnorm_y_i = 0.0
            for p in models[i].parameters():
                if p.grad is not None:
                    gnorm_y_i += p.grad.detach().norm().item() ** 2
            iter_gnorm_y.append(gnorm_y_i ** 0.5)

            # y_i^{t+1} = y_i^t - η_y·∇f_i^{tr} - η_y·(γ/2)·Σ_{l≠i} x_{i,l}·(y_i - y_l)
            with torch.no_grad():
                for p_idx, param in enumerate(models[i].parameters()):
                    update = eta_y * param.grad.detach().clone()

                    consensus = torch.zeros_like(param)
                    for k, l in enumerate(nb):
                        if l == i:
                            continue
                        diff = params_snapshot[i][p_idx] - params_snapshot[l][p_idx]
                        consensus += X[i][k].item() * diff

                    update += eta_y * (gamma / 2.0) * consensus
                    param.sub_(update)

        # Append this iteration's per-node norms to the buffers
        _gnorm_y_buf.extend(iter_gnorm_y)
        _gnorm_x_buf.extend(iter_gnorm_x)

        # ════════════════════════════════════════════════════════════
        # LOGGING
        # ════════════════════════════════════════════════════════════
        if (t + 1) % log_every == 0:

            # (a) Avg local train loss — one fresh batch per node
            train_losses = []
            for i in range(K_NODES):
                X_b, y_b = next(iter(train_loaders[i]))
                X_b = X_b.unsqueeze(1).to(device)
                y_b = y_b.to(device)
                with torch.no_grad():
                    train_losses.append(criterion(models[i](X_b), y_b).item())
            avg_train_loss = sum(train_losses) / len(train_losses)

            # (b) Avg validation loss of aggregated models — one batch per node
            val_losses      = []
            node_val_losses = []   # NEW: per-node list for Diagnostic 6
            for i in range(K_NODES):
                nb   = neighborhoods[i]
                X_b, y_b = get_batch(val_iterators, val_loaders, i)
                y_hat_log = model_factory().to(device)
                with torch.no_grad():
                    for p_idx, (_, p_hat) in enumerate(y_hat_log.named_parameters()):
                        p_hat.copy_(
                            sum(
                                X[i][k].item() * list(models[nb[k]].parameters())[p_idx]
                                for k in range(len(nb))
                            )
                        )
                    node_vl = criterion(y_hat_log(X_b), y_b).item()
                val_losses.append(node_vl)
                node_val_losses.append(node_vl)   # NEW
            avg_val_loss = sum(val_losses) / len(val_losses)

            # (c) Global test loss & accuracy
            total_test_loss, total_correct, total_samples = 0.0, 0, 0
            for i in range(K_NODES):
                node_loss, node_correct, node_samples = 0.0, 0, 0
                with torch.no_grad():
                    for X_b, y_b in test_loader:
                        X_b = X_b.unsqueeze(1).to(device)
                        y_b = y_b.to(device)
                        logits        = models[i](X_b)
                        node_loss    += criterion(logits, y_b).item() * len(y_b)
                        node_correct += (logits.argmax(dim=1) == y_b).sum().item()
                        node_samples += len(y_b)
                total_test_loss += node_loss / node_samples
                total_correct   += node_correct
                total_samples   += node_samples
            avg_test_loss = total_test_loss / K_NODES
            avg_test_acc  = total_correct   / total_samples

            # (d) Snapshot X history
            for i in range(K_NODES):
                X_history[i].append(X[i].clone())

            total_nb = sum(len(neighborhoods[i]) for i in range(K_NODES))
            avg_x = sum(
                X[i][k].item()
                for i in range(K_NODES)
                for k in range(len(neighborhoods[i]))
            ) / total_nb

            # (e) Consensus disagreement: (1/K)·Σ_i ||y_i - ȳ||²
            with torch.no_grad():
                mean_params = [
                    torch.stack([list(models[i].parameters())[idx]
                                 for i in range(K_NODES)]).mean(dim=0)
                    for idx in range(len(list(models[0].parameters())))
                ]
                consensus_dis = sum(
                    (p - p_mean).pow(2).sum().item()
                    for i in range(K_NODES)
                    for p, p_mean in zip(models[i].parameters(), mean_params)
                ) / K_NODES

            # (f) Weighted pairwise disagreement: Σ_{i<j} x_{i,j}·||y_i - y_j||²
            with torch.no_grad():
                pairwise_dis = 0.0
                for i in range(K_NODES):
                    for k, l in enumerate(neighborhoods[i]):
                        if l <= i:
                            continue
                        diff_sq = sum(
                            (pi - pl).pow(2).sum().item()
                            for pi, pl in zip(models[i].parameters(),
                                              models[l].parameters())
                        )
                        pairwise_dis += X[i][k].item() * diff_sq

            # (g) NEW: gradient norms — mean over all (node × iter) since last log
            avg_gnorm_y = float(np.mean(_gnorm_y_buf)) if _gnorm_y_buf else float('nan')
            avg_gnorm_x = float(np.mean(_gnorm_x_buf)) if _gnorm_x_buf else float('nan')
            _gnorm_y_buf.clear()
            _gnorm_x_buf.clear()

            # (h) Store all metrics
            log_iters.append(t + 1)
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            test_loss_history.append(avg_test_loss)
            test_acc_history.append(avg_test_acc)
            consensus_disagreement_history.append(consensus_dis)
            pairwise_disagreement_history.append(pairwise_dis)
            grad_norm_y_history.append(avg_gnorm_y)     # NEW
            grad_norm_x_history.append(avg_gnorm_x)     # NEW
            node_val_loss_history.append(node_val_losses)  # NEW

            # ── tqdm postfix update ──────────────────────────────
            pbar.set_postfix({
                'loss'  : f'{avg_train_loss:.4f}',
                'acc'   : f'{avg_test_acc*100:.2f}%',
                'cons'  : f'{consensus_dis:.4f}',
                'avg_w' : f'{avg_x:.4f}',
                '|∇y|'  : f'{avg_gnorm_y:.3f}',   # NEW
                '|∇x|'  : f'{avg_gnorm_x:.3f}',   # NEW
            })

            if verbose:
                print(
                    f"Iter {t+1:>4d}/{T} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Test Loss: {avg_test_loss:.4f} | "
                    f"Test Acc: {avg_test_acc*100:.2f}% | "
                    f"Avg x_{{i,l}}: {avg_x:.4f} | "
                    f"Consensus Dis: {consensus_dis:.6f} | "
                    f"Pairwise Dis: {pairwise_dis:.6f} | "
                    f"‖∇_y‖: {avg_gnorm_y:.4f} | "   # NEW
                    f"‖∇_x‖: {avg_gnorm_x:.4f}"      # NEW
                )

    return {
        'log_iters'                      : log_iters,
        'train_loss_history'             : train_loss_history,
        'val_loss_history'               : val_loss_history,
        'test_loss_history'              : test_loss_history,
        'test_acc_history'               : test_acc_history,
        'consensus_disagreement_history' : consensus_disagreement_history,
        'pairwise_disagreement_history'  : pairwise_disagreement_history,
        'X_history'                      : X_history,
        'final_models'                   : models,
        'final_X'                        : {i: X[i].clone() for i in range(K_NODES)},
        'neighborhoods'                  : neighborhoods,
        # NEW keys consumed by Diagnostic 5, 6 and gradient norm plots:
        'grad_norm_y_history'            : grad_norm_y_history,
        'grad_norm_x_history'            : grad_norm_x_history,
        'node_val_loss_history'          : node_val_loss_history,
    }