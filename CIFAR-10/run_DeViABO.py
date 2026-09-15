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

from torch.func import functional_call
from tqdm import trange
import itertools
from torch.utils.data import Subset
from help_functions import augment_batch   # or wherever it lives

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
    log_callback  = None,       # ← callable(t, metrics_dict) or None
):
    """
    DeViABO with DIVERSITY-SEEKING aggregation (self-loops included in N_i).

    PERFORMANCE-OPTIMIZED VERSION — identical math to the original:
      - Reuses ONE scratch model instead of calling model_factory().to(device)
        inside every inner loop (this was the single biggest bottleneck).
      - Keeps intermediate values as GPU tensors, deferring .item() calls to
        the end of each block instead of forcing a sync every iteration.
      - Vectorizes the per-neighbor consensus update and the disagreement
        metrics using stacked tensors instead of Python for-loops with
        per-element .item() calls.
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

    models = []
    for k in range(K_NODES):
        torch.manual_seed(seed + k)
        models.append(model_factory().to(device))
    torch.manual_seed(seed)

    # ── NEW: single reusable scratch model(s), created ONCE ──────────
    scratch_val_model = model_factory().to(device)
    scratch_agg_model = model_factory().to(device)

    criterion = nn.CrossEntropyLoss()

    X = {
        i: torch.full(
            (len(neighborhoods[i]),),
            1.0 / len(neighborhoods[i]),
            device=device,
        )
        for i in range(K_NODES)
    }

    def simplex_projection(v):
        n    = v.shape[0]
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0)
        rho  = (
            u > (cssv - 1.0) / torch.arange(1, n + 1, dtype=v.dtype, device=v.device)
        ).nonzero()[-1].item()
        lam = (cssv[rho] - 1.0) / (rho + 1.0)
        return torch.clamp(v - lam, min=0.0)

    train_iterators = [iter(train_loaders[i]) for i in range(K_NODES)]
    val_iterators   = [iter(val_loaders[i])   for i in range(K_NODES)]

    def get_batch(iterators, loaders, i):
        try:
            xb, yb = next(iterators[i])
        except StopIteration:
            iterators[i] = iter(loaders[i])
            xb, yb = next(iterators[i])
        return xb.to(device), yb.to(device)

    # ── Tracking metrics ─────────────────────────────────────────
    log_iters                      = []
    train_loss_history             = []
    val_loss_history               = []
    test_loss_history              = []
    test_acc_history                = []
    consensus_disagreement_history = []
    pairwise_disagreement_history  = []
    X_history                      = {i: [] for i in range(K_NODES)}

    grad_norm_y_history            = []
    grad_norm_x_history            = []
    node_val_loss_history          = []

    _gnorm_y_buf = []
    _gnorm_x_buf = []

    # ── TIMING SETUP ────────────────────────────────────────────────
    _iter_time_accum = 0.0
    _iter_count_since_report = 0
    _log_time_accum = 0.0


    pbar = trange(T, desc=f"[DeViABO | {graph_type} | K={K_NODES}]",
                  unit="iter", leave=True)

    param_names = [name for name, _ in scratch_val_model.named_parameters()]
    buffer_dict = dict(scratch_val_model.named_buffers())
    for t in pbar:
        _iter_start = time.perf_counter()          # 


        params_snapshot = [
            [p.detach().clone() for p in models[i].parameters()]
            for i in range(K_NODES)
        ]

        iter_gnorm_y = []
        iter_gnorm_x = []

        for i in range(K_NODES):
            nb = neighborhoods[i]

            X_val,   y_val   = get_batch(val_iterators,   val_loaders,   i)
            X_train, y_train = get_batch(train_iterators, train_loaders, i)
            X_train = augment_batch(X_train, device)

            # ════════════════════════════════════════════════════════
            # LOWER-LEVEL UPDATE (diversity-seeking weights)
            # ════════════════════════════════════════════════════════

            individual_val_losses = torch.zeros(len(nb), device=device)
            with torch.no_grad():
                for k, l in enumerate(nb):
                    for p_idx, p in enumerate(scratch_val_model.parameters()):
                        p.copy_(params_snapshot[l][p_idx])
                    logits = scratch_val_model(X_val)
                    individual_val_losses[k] = criterion(logits, y_val)

            grad_x = individual_val_losses

            iter_gnorm_x.append(grad_x.detach().norm())

            with torch.no_grad():
                uniform_weight = 1.0 / len(nb)
                grad_beta = 2.0 * beta * (X[i] - uniform_weight)

            with torch.no_grad():
                X[i] = simplex_projection(
                    X[i] + eta_x * grad_x - eta_x * grad_beta
                )
            # individual_val_losses_list = []
            # with torch.no_grad():
            #     for l in nb:
            #         # Build the neighbor's parameter dict directly from the snapshot --
            #         # NO .copy_() calls, NO mutation of scratch_val_model at all.
            #         param_dict = {name: params_snapshot[l][idx] for idx, name in enumerate(param_names)}
            #         logits = functional_call(scratch_val_model, (param_dict, buffer_dict), (X_val,))
            #         individual_val_losses_list.append(criterion(logits, y_val))

            # individual_val_losses = torch.stack(individual_val_losses_list)

            # grad_x = individual_val_losses
            # iter_gnorm_x.append(grad_x.detach().norm())

            # with torch.no_grad():
            #     uniform_weight = 1.0 / len(nb)
            #     grad_beta = 2.0 * beta * (X[i] - uniform_weight)

            # with torch.no_grad():
            #     X[i] = simplex_projection(
            #         X[i] + eta_x * grad_x - eta_x * grad_beta
            #     )
            # print(f"  Node {i}: N_{i} = {nb}, {X[i]}")

            # ════════════════════════════════════════════════════════
            # UPPER-LEVEL UPDATE (model parameters)
            # ════════════════════════════════════════════════════════

            models[i].zero_grad()
            with torch.enable_grad():
                loss_tr = criterion(models[i](X_train), y_train)
                loss_tr.backward()

            grads = [p.grad.detach() for p in models[i].parameters() if p.grad is not None]
            if grads:
                gnorm_y_i = torch.sqrt(sum((g.pow(2).sum() for g in grads[1:]), grads[0].pow(2).sum()))
            else:
                gnorm_y_i = torch.tensor(0.0, device=device)
            iter_gnorm_y.append(gnorm_y_i)

            with torch.no_grad():
                mask_vec = torch.tensor(
                    [0.0 if l == i else 1.0 for l in nb],
                    device=device, dtype=X[i].dtype,
                )
                weights_i = X[i] * mask_vec

                for p_idx, param in enumerate(models[i].parameters()):
                    neighbor_stack = torch.stack(
                        [params_snapshot[l][p_idx] for l in nb], dim=0
                    )
                    diffs = params_snapshot[i][p_idx].unsqueeze(0) - neighbor_stack
                    w = weights_i.view(-1, *([1] * (diffs.dim() - 1)))
                    consensus = (w * diffs).sum(dim=0)

                    update = eta_y * param.grad.detach() + eta_y * (gamma / 2.0) * consensus
                    param.sub_(update)

            # models[i].zero_grad()
            # with torch.enable_grad():
            #     loss_tr = criterion(models[i](X_train), y_train)
            #     loss_tr.backward()

            # grads = [p.grad.detach() for p in models[i].parameters() if p.grad is not None]
            # if grads:
            #     gnorm_y_i = torch.sqrt(sum((g.pow(2).sum() for g in grads[1:]), grads[0].pow(2).sum()))
            # else:
            #     gnorm_y_i = torch.tensor(0.0, device=device)
            # iter_gnorm_y.append(gnorm_y_i)

            # with torch.no_grad():
            #     param_list = list(models[i].parameters())
            #     grad_list = [p.grad.detach() for p in param_list]

            #     # Zero-initialized accumulator, one tensor per parameter, same shapes as params_snapshot[i].
            #     consensus_list = torch._foreach_mul(params_snapshot[i], 0.0)

            #     for k, l in enumerate(nb):
            #         if l == i:
            #             continue   # self-loop excluded from consensus -- matches the original mask_vec logic
            #         w = X[i][k]    # scalar weight for this neighbor

            #         # ONE fused op across ALL parameter tensors, instead of looping p_idx over ~30-40 tensors:
            #         diffs = torch._foreach_sub(params_snapshot[i], params_snapshot[l])
            #         scaled = torch._foreach_mul(diffs, w)
            #         torch._foreach_add_(consensus_list, scaled)

            #     # Final combination: eta_y * grad + eta_y * (gamma/2) * consensus, applied in-place.
            #     scaled_grad = torch._foreach_mul(grad_list, eta_y)
            #     scaled_consensus = torch._foreach_mul(consensus_list, eta_y * gamma / 2.0)
            #     updates = torch._foreach_add(scaled_grad, scaled_consensus)
            #     torch._foreach_sub_(param_list, updates)


        # ← ADD THESE 3 LINES, right after the per-node `for i in range(K_NODES):` loop closes,
        #    and BEFORE the existing _gnorm_y_buf.extend(...) line:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _iter_time_accum += time.perf_counter() - _iter_start
        _iter_count_since_report += 1

        _gnorm_y_buf.extend(iter_gnorm_y)
        _gnorm_x_buf.extend(iter_gnorm_x)

        # ════════════════════════════════════════════════════════════
        # LOGGING
        # ════════════════════════════════════════════════════════════
        if (t + 1) % log_every == 0:
            _log_start = time.perf_counter()        # ← ADD THIS LINE


            train_losses = []
            for i in range(K_NODES):
                X_b, y_b = next(iter(train_loaders[i]))
                X_b = augment_batch(X_b, device)
                y_b = y_b.to(device)
                with torch.no_grad():
                    train_losses.append(criterion(models[i](X_b), y_b))
            avg_train_loss = (sum(train_losses) / len(train_losses)).item()

            val_losses      = []
            node_val_losses = []
            for i in range(K_NODES):
                nb   = neighborhoods[i]
                X_b, y_b = get_batch(val_iterators, val_loaders, i)
                with torch.no_grad():
                    for p_idx, p_hat in enumerate(scratch_agg_model.parameters()):
                        acc = None
                        for k in range(len(nb)):
                            term = X[i][k] * list(models[nb[k]].parameters())[p_idx]
                            acc = term if acc is None else acc + term
                        p_hat.copy_(acc)
                    node_vl = criterion(scratch_agg_model(X_b), y_b)
                val_losses.append(node_vl)
                node_val_losses.append(node_vl.item())
            avg_val_loss = (sum(val_losses) / len(val_losses)).item()

            total_test_loss, total_correct, total_samples = 0.0, 0, 0
            for i in range(K_NODES):
                node_loss, node_correct, node_samples = 0.0, 0, 0
                with torch.no_grad():
                    for X_b, y_b in test_loader:
                        X_b = X_b.to(device)
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

            for i in range(K_NODES):
                X_history[i].append(X[i].clone())

            total_nb = sum(len(neighborhoods[i]) for i in range(K_NODES))
            avg_x = (sum(X[i].sum() for i in range(K_NODES)) / total_nb).item()

            with torch.no_grad():
                mean_params = [
                    torch.stack([list(models[i].parameters())[idx]
                                 for i in range(K_NODES)]).mean(dim=0)
                    for idx in range(len(list(models[0].parameters())))
                ]
                consensus_dis_t = sum(
                    (p - p_mean).pow(2).sum()
                    for i in range(K_NODES)
                    for p, p_mean in zip(models[i].parameters(), mean_params)
                ) / K_NODES
                consensus_dis = consensus_dis_t.item()

            with torch.no_grad():
                pairwise_dis_t = None
                for i in range(K_NODES):
                    for k, l in enumerate(neighborhoods[i]):
                        if l <= i:
                            continue
                        diff_sq = sum(
                            (pi - pl).pow(2).sum()
                            for pi, pl in zip(models[i].parameters(),
                                              models[l].parameters())
                        )
                        term = X[i][k] * diff_sq
                        pairwise_dis_t = term if pairwise_dis_t is None else pairwise_dis_t + term
                pairwise_dis = pairwise_dis_t.item() if pairwise_dis_t is not None else 0.0

            avg_gnorm_y = torch.stack(_gnorm_y_buf).mean().item() if _gnorm_y_buf else float('nan')
            avg_gnorm_x = torch.stack(_gnorm_x_buf).mean().item() if _gnorm_x_buf else float('nan')
            _gnorm_y_buf.clear()
            _gnorm_x_buf.clear()

            log_iters.append(t + 1)
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            test_loss_history.append(avg_test_loss)
            test_acc_history.append(avg_test_acc)
            consensus_disagreement_history.append(consensus_dis)
            pairwise_disagreement_history.append(pairwise_dis)
            grad_norm_y_history.append(avg_gnorm_y)
            grad_norm_x_history.append(avg_gnorm_x)
            node_val_loss_history.append(node_val_losses)


            # ── Fire callback immediately after metrics are stored ────
            if log_callback is not None:
                log_callback(t + 1, {
                    'test_acc'   : avg_test_acc,
                    'test_loss'  : avg_test_loss,
                    'val_loss'   : avg_val_loss,
                    'train_loss' : avg_train_loss,
                    'consensus'  : consensus_dis,
                    'pairwise'   : pairwise_dis,
                    'avg_x'      : avg_x,
                    'current_X'  : {i: X[i].clone() for i in range(K_NODES)},  # ← add this

            })

            # ← ADD THESE LINES, right after all the history .append() calls,
            #    and BEFORE pbar.set_postfix({...}):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _log_elapsed = time.perf_counter() - _log_start
            _log_time_accum += _log_elapsed
            avg_iter_time = _iter_time_accum / max(_iter_count_since_report, 1)
            # print(
            #     f"\n[TIMING] iter {t+1}: "
            #     f"avg non-log iter time = {avg_iter_time*1000:.1f} ms/iter "
            #     f"(over {_iter_count_since_report} iters)\n"
            #     f"         | non-log time this window = {_iter_time_accum:.2f}s\n"
            #     f"         | LOGGING BLOCK took        = {_log_elapsed:.2f}s\n"
            #     f"         | cumulative logging time so far = {_log_time_accum/60:.2f} min"
            # )
            _iter_time_accum = 0.0
            _iter_count_since_report = 0

            pbar.set_postfix({
                'loss'  : f'{avg_train_loss:.4f}',
                'acc'   : f'{avg_test_acc*100:.2f}%',
                'cons'  : f'{consensus_dis:.4f}',
                'avg_w' : f'{avg_x:.4f}',
                '|∇y|'  : f'{avg_gnorm_y:.3f}',
                '|∇x|'  : f'{avg_gnorm_x:.3f}',
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
                    f"‖∇_y‖: {avg_gnorm_y:.4f} | "
                    f"‖∇_x‖: {avg_gnorm_x:.4f}"
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
        'grad_norm_y_history'            : grad_norm_y_history,
        'grad_norm_x_history'            : grad_norm_x_history,
        'node_val_loss_history'          : node_val_loss_history,
    }