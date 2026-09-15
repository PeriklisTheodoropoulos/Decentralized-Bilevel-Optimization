import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
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
from help_functions import split_node_data_non_iid, generate_graph,\
                            compute_pairwise_disagreement, compute_consensus_disagreement,\
                            compute_midpoint_grads, compute_stochastic_grad
from run_DeViABO import run_deviabo
import itertools
from model import CNNCifar10, CNNCifar10Deep,  CNNCifar10ResNet9
from torch.utils.data import Subset
from scipy.stats import pearsonr
from collections import defaultdict


print("Is CUDA available?:", torch.cuda.is_available())
print("Available GPU count:", torch.cuda.device_count())
print("Current device index:", torch.cuda.current_device())

# Bound host-side math parallelism so concurrent GPU experiments do not
# oversubscribe CPU cores during data handling and linear-algebra operations.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ═════════════════════════════════════════════════════════════════
# DeViABO Hyperparameter Tuning and Plotting Script — CIFAR-10
# The script keeps data, topology, initialization seed, and evaluation fixed
# across the grid so differences can be attributed to the four tuned scalars.
# ═════════════════════════════════════════════════════════════════
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

GRAPH_TYPE   = 'ring' #'caveman'

eta_y_values = [0.5]
gamma_values = [4.0]

eta_x_values = [5e-5]
beta_values  = [1e-4]
T          = 2000 #10000
SEED       = 42
BATCH_SIZE = 128
BATCH_SIZE_TEST = 500
Log_Every = 500

K_NODES    = 10 #21, 14
VAL_RATIO  = 0.2
mode       = 'dirichlet' #'clustered_dirichlet'
dir_level  = 0.5 #10.0
num_groups = 2 #3

run_prefix = (
    f"/home/ptheodorop/DFL_mnist/Eusome/Cifar10/Cifar10_github_code/{GRAPH_TYPE.replace(' ', '_')}_K{K_NODES}"
    f"_{mode}_dir_{dir_level}_T{T}_cifar10_eta_y_[{eta_y_values}]_gamma_[{gamma_values}]_eta_x_[{eta_x_values}]_beta_[{beta_values}]"
)

incremental_filename = (
    f"/home/ptheodorop/DFL_mnist/Eusome/Cifar10/Cifar10_github_code/DeViABO_incremental_K{K_NODES}_{GRAPH_TYPE.replace(' ', '_')}"
    f"_T{T}_{mode}_dir_[{dir_level}]_cifar10.txt"
)


def _write_incremental_result(filepath, config_id, total_configs,
                               result, metrics, first_entry):
    """Persist one completed configuration before the sweep continues.

    The first call creates a self-describing log and subsequent calls append to
    it. Consequently, completed runs survive interruption or a later failure.
    """
    sep  = "=" * 80
    sep2 = "-" * 80
    iters = metrics['log_iters']

    te_acc_str = ', '.join(f"{v*100:.2f}%" for v in metrics['test_acc_history'])
    te_los_str = ', '.join(f"{v:.4f}"      for v in metrics['test_loss_history'])
    tr_los_str = ', '.join(f"{v:.4f}"      for v in metrics['train_loss_history'])
    cons_str   = ', '.join(f"{v:.6f}"      for v in metrics['consensus_disagreement_history'])
    iters_str  = ', '.join(str(i)          for i in iters)
    log_gap    = (iters[1] - iters[0]) if len(iters) > 1 else 'N/A'

    lines = []
    if first_entry:
        lines += [
            sep,
            "  DeViABO INCREMENTAL EXPERIMENT LOG — CIFAR-10",
            f"  Graph={GRAPH_TYPE} | K={K_NODES} | T={T} | "
            f"mode={mode} | α={dir_level} | batch={BATCH_SIZE}",
            sep, "",
        ]

    lines += [
        sep2,
        f"  [{config_id}/{total_configs}]  {result['config_name']}",
        sep2,
        f"  eta_x : {result['eta_x']:.0e}   |   "
        f"eta_y : {result['eta_y']:.0e}   |   "
        f"  γ (consensus) = {result['gamma']:.2f}   |   "
        f"  β (reg)       = {result['beta']:.0e}   |   "
        f"  Finished at  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Training time: {result['time_seconds']:.1f}s",
        "",
        f"  FINAL METRICS (iter {iters[-1]}):",
        f"    Test  Accuracy  : {result['final_test_acc']:.2f}%",
        f"    Test  Loss      : {result['final_test_loss']:.4f}",
        f"    Train Loss      : {result['final_train_loss']:.4f}",
        f"    Consensus Dis.  : {result['final_consensus']:.6f}",
        f"    Intra/Inter w.  : {result['weight_ratio']:.4f}",
        "",
        f"  FULL HISTORY (logged every {log_gap} iters):",
        f"    Iters      : [{iters_str}]",
        f"    Test  Acc  : [{te_acc_str}]",
        f"    Test  Loss : [{te_los_str}]",
        f"    Train Loss : [{tr_los_str}]",
        f"    Consensus  : [{cons_str}]",
        "",
    ]

    mode_flag = 'w' if first_entry else 'a'
    with open(filepath, mode_flag) as f:
        f.write('\n'.join(lines) + '\n')


# ─────────────────────────────────────────────────────────────────
# Define Data
# Keep torchvision transforms disabled because partitioning operates on raw
# CIFAR-10 arrays; normalization is applied consistently after node splitting.
# ─────────────────────────────────────────────────────────────────
DATASET = "cifar10"
dataset_path = f'/home/ptheodorop/DFL_mnist/cifar10_example/datasets/{DATASET}'
dataset_train = datasets.CIFAR10(root=dataset_path, download=False, transform=None)
dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
dataset_test  = datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=None)
dataset_test.targets  = torch.as_tensor(np.array(dataset_test.targets))

number_classes, n_channels, img_size = 10, 3, 32




# ─────────────────────────────────────────────────────────────────
# DataLoaders
# The splitter returns raw uint8 tensors in NCHW form. Normalize every node
# with the same channel statistics only after partitioning, preserving the
# intended non-IID label allocation while standardizing the input scale.
# ─────────────────────────────────────────────────────────────────
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465],
                              device=device).view(1, 3, 1, 1)
CIFAR10_STD  = torch.tensor([0.2023, 0.1994, 0.2010],
                              device=device).view(1, 3, 1, 1)


def normalize_cifar10(t) -> torch.Tensor:
    """Convert CIFAR-10 batches to normalized float32 tensors on the device.

    NumPy inputs use NHWC layout and are transposed to NCHW. Values that still
    have the uint8 scale are mapped to [0, 1] before channel normalization;
    already-scaled tensors are not divided again.
    """
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).permute(0, 3, 1, 2)
    x = t.to(device, dtype=torch.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return (x - CIFAR10_MEAN) / CIFAR10_STD


def make_loaders(val_ratio: float, seed: int):
    """Create node-local train/validation loaders and one global test loader.

    All partitions are materialized on the selected device once, avoiding
    repeated host-to-device transfers during decentralized optimization.
    Training loaders shuffle locally; validation and test loaders remain
    deterministic so configurations are evaluated on the same sample order.
    """
    data_chunks, test_data = split_node_data_non_iid(
        dataset_train = dataset_train,
        dataset_test  = dataset_test,
        K             = K_NODES,
        val_ratio     = VAL_RATIO,
        seed          = seed,
        mode          = mode,
        alpha         = dir_level,
        num_groups    = num_groups,
    )

    _train_loaders = []
    _val_loaders   = []

    for node_id in range(K_NODES):
        tr_data, tr_labels = data_chunks[node_id]['train']
        tr_data_gpu   = normalize_cifar10(tr_data)
        tr_labels_gpu = tr_labels.to(device, dtype=torch.long)
        _train_loaders.append(
            DataLoader(
                TensorDataset(tr_data_gpu, tr_labels_gpu),
                batch_size  = BATCH_SIZE,
                shuffle     = True,
                num_workers = 0,
                pin_memory  = False,
            )
        )

        if val_ratio > 0.0:
            vl_data, vl_labels = data_chunks[node_id]['val']
            if len(vl_data) > 0:
                vl_data_gpu   = normalize_cifar10(vl_data)
                vl_labels_gpu = vl_labels.to(device, dtype=torch.long)
                _val_loaders.append(
                    DataLoader(
                        TensorDataset(vl_data_gpu, vl_labels_gpu),
                        batch_size  = BATCH_SIZE,
                        shuffle     = False,
                        num_workers = 0,
                        pin_memory  = False,
                    )
                )
            else:
                _val_loaders.append(None)
        else:
            _val_loaders.append(None)

    te_imgs, te_labels = test_data
    te_imgs_gpu   = normalize_cifar10(te_imgs)
    te_labels_gpu = te_labels.to(device, dtype=torch.long)
    _test_loader  = DataLoader(
        TensorDataset(te_imgs_gpu, te_labels_gpu),
        batch_size  = BATCH_SIZE_TEST,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    return _train_loaders, _val_loaders, _test_loader


train_loaders, val_loaders, test_loader = make_loaders(val_ratio=VAL_RATIO, seed=SEED)
# ── Model factory ─────────────────────────────────────────────────
# Return a fresh network whenever requested so decentralized nodes receive
# separate parameter objects while sharing one architecture.
model_factory = lambda: CNNCifar10Deep(n_class=number_classes, device=device)


# ─────────────────────────────────────────────────────────────────
# Hyperparameter Grid
# The grid jointly probes upper-level adaptation, lower-level optimization,
# consensus coupling, and regularization rather than varying them in isolation.
# ─────────────────────────────────────────────────────────────────
total_configs = (len(eta_x_values) * len(eta_y_values)
                 * len(gamma_values) * len(beta_values))

all_results = []
all_metrics = {}

print("=" * 80)
print("DeViABO HYPERPARAMETER TUNING — CIFAR-10")
print(f"Total configurations: {total_configs}")
print(f"K_NODES={K_NODES}, graph={GRAPH_TYPE}, T={T}, batch={BATCH_SIZE}, "
      f"mode={mode}, alpha={dir_level}")
print("=" * 80)


# ─────────────────────────────────────────────────────────────────
# Grid Search Loop
# Evaluate the complete Cartesian product while reusing the same loaders and
# seed, which makes cross-configuration comparisons controlled.
# ─────────────────────────────────────────────────────────────────
config_id   = 0
first_entry = True

for eta_x, eta_y, gamma, beta in itertools.product(
        eta_x_values, eta_y_values, gamma_values, beta_values):

    config_id  += 1
    config_name = f"ηx={eta_x:.0e}_ηy={eta_y:.0e}_γ={gamma:.2f}_β={beta:.0e}"

    print(f"\n[{config_id}/{total_configs}] "
          f"η_x={eta_x:.0e}, η_y={eta_y:.0e}, γ={gamma:.2f}, β={beta:.0e}")

    start_time = time.time()

    try:
        deviabo_metrics = run_deviabo(
            train_loaders = train_loaders,
            val_loaders   = val_loaders,
            test_loader   = test_loader,
            K_NODES       = K_NODES,
            num_classes   = number_classes,
            graph_type    = GRAPH_TYPE,
            eta_y         = eta_y,
            eta_x         = eta_x,
            gamma         = gamma,
            beta          = beta,
            T             = T,
            seed          = SEED,
            model_factory = model_factory,
            log_every     = Log_Every,
            verbose       = False,
            device        = device,
        )

        elapsed_time = time.time() - start_time

        final_test_acc   = deviabo_metrics['test_acc_history'][-1]
        final_test_loss  = deviabo_metrics['test_loss_history'][-1]
        final_train_loss = deviabo_metrics['train_loss_history'][-1]
        final_val_loss   = deviabo_metrics['val_loss_history'][-1]
        final_consensus  = deviabo_metrics['consensus_disagreement_history'][-1]

        final_X       = deviabo_metrics['final_X']
        neighborhoods = deviabo_metrics['neighborhoods']

        nodes_per_class = 1
        intra_weights   = []
        inter_weights   = []

        for i in range(K_NODES):
            class_i = i // nodes_per_class
            nb_i    = neighborhoods[i]
            for k, l in enumerate(nb_i):
                if l == i:
                    continue
                class_l = l // nodes_per_class
                weight  = final_X[i][k].item()
                if class_i == class_l:
                    intra_weights.append(weight)
                else:
                    inter_weights.append(weight)

        avg_intra    = np.mean(intra_weights) if intra_weights else 0.0
        avg_inter    = np.mean(inter_weights) if inter_weights else 0.0
        weight_ratio = avg_intra / (avg_inter + 1e-8)

        result = {
            'config_id'       : config_id,
            'config_name'     : config_name,
            'eta_x'           : eta_x,
            'eta_y'           : eta_y,
            'gamma'           : gamma,
            'beta'            : beta,
            'final_test_acc'  : final_test_acc * 100,
            'final_test_loss' : final_test_loss,
            'final_train_loss': final_train_loss,
            'final_val_loss'  : final_val_loss,
            'final_consensus' : final_consensus,
            'weight_ratio'    : weight_ratio,
            'time_seconds'    : elapsed_time,
        }

        all_results.append(result)
        all_metrics[config_name] = deviabo_metrics

        # Save immediately so a long sweep remains recoverable.
        _write_incremental_result(
            filepath      = incremental_filename,
            config_id     = config_id,
            total_configs = total_configs,
            result        = result,
            metrics       = deviabo_metrics,
            first_entry   = first_entry,
        )
        first_entry = False

        print(f"  ✓ Final Acc: {final_test_acc*100:.2f}% | "
              f"Val Loss: {final_val_loss:.4f} | "
              f"Weight Ratio: {weight_ratio:.3f} | "
              f"Time: {elapsed_time:.1f}s  → saved to '{incremental_filename}'")

    except Exception as e:
        # Isolate one failed configuration instead of terminating the grid.
        import traceback
        print(f"  ✗ FAILED: {str(e)}")
        traceback.print_exc()
        fail_lines = [
            f"  [{config_id}/{total_configs}]  {config_name}  ✗ FAILED",
            f"  Error: {str(e)}",
            "",
        ]
        mode_flag = 'w' if first_entry else 'a'
        with open(incremental_filename, mode_flag) as f:
            f.write('\n'.join(fail_lines) + '\n')
        first_entry = False
        continue


# ─────────────────────────────────────────────────────────────────
# Results DataFrame
# Rank only successful runs; failed configurations remain documented in the
# incremental log but cannot participate in metric-based selection.
# ─────────────────────────────────────────────────────────────────
if len(all_results) == 0:
    print("\n✗ No configurations completed successfully. Exiting.")
    raise SystemExit(1)

df_results = pd.DataFrame(all_results)
df_results = df_results.sort_values('final_test_acc', ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING COMPLETE")
print("=" * 80)
print(df_results[['config_name', 'final_test_acc', 'final_val_loss',
                   'weight_ratio', 'time_seconds']].to_string(index=False))

df_results.to_csv(f'{run_prefix}_deviabo_hyperparameter_results.csv', index=False)
print(f"\n✓ Results saved to '{run_prefix}_deviabo_hyperparameter_results.csv'")


# ═════════════════════════════════════════════════════════════════
# BEST CONFIGURATION
# Test accuracy is the selection criterion used consistently for reporting and
# visual highlighting of the winning trajectory.
# ═════════════════════════════════════════════════════════════════
best_config = df_results.iloc[0]

print("\n" + "=" * 80)
print("★ BEST HYPERPARAMETER CONFIGURATION (by Test Accuracy) ★")
print("=" * 80)
print(f"Configuration : {best_config['config_name']}")
print(f"η_x           : {best_config['eta_x']:.0e}")
print(f"η_y           : {best_config['eta_y']:.0e}")
print(f"γ (consensus) : {best_config['gamma']:.2f}")
print(f"β (reg)       : {best_config['beta']:.0e}")
print("-" * 80)
print(f"Final Test Accuracy   : {best_config['final_test_acc']:.2f}%")
print(f"Final Test Loss       : {best_config['final_test_loss']:.4f}")
print(f"Final Val Loss        : {best_config['final_val_loss']:.4f}")
print(f"Final Train Loss      : {best_config['final_train_loss']:.4f}")
print(f"Final Consensus Dis.  : {best_config['final_consensus']:.4f}")
print(f"Intra/Inter Ratio     : {best_config['weight_ratio']:.2f}")
print(f"Training Time         : {best_config['time_seconds']:.1f}s")
print("=" * 80)

sep = "=" * 80
with open(incremental_filename, 'a') as f:
    f.write('\n'.join([
        "", sep,
        "  ★ BEST CONFIGURATION (by Test Accuracy) ★", sep,
        f"  {best_config['config_name']}",
        f"  η_x={best_config['eta_x']:.0e}  η_y={best_config['eta_y']:.0e}"
        f"  γ (consensus) = {best_config['gamma']:.2f}",
        f"  β (reg)       = {best_config['beta']:.0e}",
        f"  Test Accuracy : {best_config['final_test_acc']:.2f}%",
        f"  Test Loss     : {best_config['final_test_loss']:.4f}",
        f"  Train Loss    : {best_config['final_train_loss']:.4f}",
        f"  Consensus Dis.: {best_config['final_consensus']:.6f}",
        f"  Intra/Inter w.: {best_config['weight_ratio']:.4f}",
        sep, "",
    ]) + '\n')


# ═════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═════════════════════════════════════════════════════════════════
n_configs = len(all_results)
colors = plt.cm.tab10(np.linspace(0, 1, min(n_configs, 10)))
if n_configs > 10:
    colors = plt.cm.tab20(np.linspace(0, 1, n_configs))


def _plot_metric(metric_key, ylabel, title, filename, transform=None, marker='o'):
    """Plot one metric for every configuration and emphasize the winner.

    Retaining non-winning trajectories reveals sensitivity to the grid instead
    of showing the selected run without experimental context.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, result in enumerate(all_results):
        cname   = result['config_name']
        metrics = all_metrics[cname]
        iters   = metrics['log_iters']
        values  = metrics[metric_key]
        if transform:
            values = [transform(v) for v in values]
        is_best = cname == best_config['config_name']
        ax.plot(iters, values,
                lw=3.5 if is_best else 2,
                label=f"{cname} ★ BEST" if is_best else cname,
                color='red' if is_best else colors[idx],
                alpha=1.0 if is_best else 0.7,
                marker=marker, markersize=5 if is_best else 3,
                markevery=5, zorder=10 if is_best else 1)
    ax.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'{run_prefix}_{filename}', dpi=150, bbox_inches='tight')
    plt.show()


def _plot_grad_norm(norm_key, ylabel, title, filename, marker='o'):
    """Plot an available gradient-norm trajectory across configurations.

    Missing histories are skipped so diagnostics remain compatible with runs
    produced when optional gradient logging is disabled.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for idx, result in enumerate(all_results):
        cname   = result['config_name']
        metrics = all_metrics[cname]
        values  = metrics.get(norm_key, None)
        if values is None or len(values) == 0:
            continue
        iters   = metrics['log_iters']
        is_best = cname == best_config['config_name']
        ax.plot(iters, values,
                lw=3.5 if is_best else 2,
                label=f"{cname} ★ BEST" if is_best else cname,
                color='red' if is_best else colors[idx],
                alpha=1.0 if is_best else 0.7,
                marker=marker, markersize=5 if is_best else 3,
                markevery=5, zorder=10 if is_best else 1)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'{run_prefix}_{filename}', dpi=150, bbox_inches='tight')
    plt.show()


# ═════════════════════════════════════════════════════════════════
# STANDARD METRIC PLOTS
# ═════════════════════════════════════════════════════════════════
_plot_metric('test_acc_history',
             'Test Accuracy (%)', 'DeViABO — CIFAR-10: Test Accuracy',
             'deviabo_c10_test_accuracy.png',
             transform=lambda v: v * 100, marker='o')

_plot_metric('val_loss_history',
             'Val Loss (Aggregated)', 'DeViABO — CIFAR-10: Validation Loss',
             'deviabo_c10_val_loss.png', marker='D')

_plot_metric('train_loss_history',
             'Train Loss', 'DeViABO — CIFAR-10: Train Loss',
             'deviabo_c10_train_loss.png', marker='s')

_plot_metric('test_loss_history',
             'Test Loss', 'DeViABO — CIFAR-10: Test Loss',
             'deviabo_c10_test_loss.png', marker='^')

_plot_metric('consensus_disagreement_history',
             'Consensus Disagreement', 'DeViABO — CIFAR-10: Consensus Disagreement',
             'deviabo_c10_consensus_disagreement.png', marker='d')


# ═════════════════════════════════════════════════════════════════
# GRADIENT NORM PLOTS
# Compare loss and gradient behavior on the same communication-round axis to
# distinguish unstable updates from network-consensus effects.
# ═════════════════════════════════════════════════════════════════
has_grad_norms_plot = any(
    'grad_norm_x_history' in all_metrics[r['config_name']]
    and len(all_metrics[r['config_name']]['grad_norm_x_history']) > 0
    for r in all_results
)

if has_grad_norms_plot:
    _plot_grad_norm('grad_norm_x_history',
                    '‖∇_x f‖  (mean over nodes)',
                    'DeViABO — CIFAR-10: Upper-level Gradient Norm  ‖∇_x f‖',
                    'deviabo_c10_grad_norm_x.png', marker='o')

    _plot_grad_norm('grad_norm_y_history',
                    '‖∇_y g‖  (mean over nodes)',
                    'DeViABO — CIFAR-10: Lower-level Gradient Norm  ‖∇_y g‖',
                    'deviabo_c10_grad_norm_y.png', marker='s')

    best_cname   = best_config['config_name']
    best_metrics = all_metrics[best_cname]
    best_iters   = best_metrics['log_iters']
    best_val     = best_metrics['val_loss_history']
    best_gnx     = best_metrics.get('grad_norm_x_history', [])
    best_gny     = best_metrics.get('grad_norm_y_history', [])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(best_iters, best_val, color='steelblue', lw=2.5, label='Val Loss')
    axes[0].set_ylabel('Val Loss', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    if best_gnx:
        axes[1].plot(best_iters, best_gnx, color='crimson', lw=2.5, label='‖∇_x f‖')
        axes[1].set_ylabel('‖∇_x f‖', fontsize=10, fontweight='bold')
        axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    if best_gny:
        axes[2].plot(best_iters, best_gny, color='darkorange', lw=2.5, label='‖∇_y g‖')
        axes[2].set_ylabel('‖∇_y g‖', fontsize=10, fontweight='bold')
        axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Communication Round', fontsize=11, fontweight='bold')
    fig.suptitle(f'DeViABO — CIFAR-10: Val Loss & Gradient Norms\n★ BEST: {best_cname}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f'{run_prefix}_deviabo_c10_best_val_and_gradnorms.png', dpi=150, bbox_inches='tight')
    plt.show()

else:
    print("\n⚠ Gradient norm plots skipped — 'grad_norm_x_history' / "
          "'grad_norm_y_history' not returned by run_deviabo.")


# ═════════════════════════════════════════════════════════════════════════
# OSCILLATION DIAGNOSTICS
# Examine complementary signatures of persistent oscillation: consensus
# disagreement, step-size sensitivity, coupling strength, gradient magnitude,
# and node position relative to heterogeneous data groups.
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("OSCILLATION DIAGNOSTICS")
print("=" * 80)

diag_lines = [
    "=" * 80,
    "  OSCILLATION DIAGNOSTICS — DeViABO CIFAR-10",
    "=" * 80, "",
]


def oscillation_amplitude(series):
    """Measure persistent peak-to-peak variation after a 20% burn-in.

    Removing the transient prevents normal early optimization progress from
    being mislabeled as steady-state oscillation.
    """
    s = np.array(series, dtype=float)
    burnin = max(1, len(s) // 5)
    s = s[burnin:]
    if np.any(np.isnan(s)):
        return float('nan')
    return float(np.max(s) - np.min(s))


def spike_correlation(a, b):
    """Return Pearson correlation and p-value for finite trajectory pairs.

    Correlation is undefined for too few samples or near-constant signals; NaN
    is returned in those cases rather than treating numerical noise as evidence.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan'), float('nan')
    r, p = pearsonr(a, b)
    return float(r), float(p)


# ── Diagnostic 1 — Val Loss vs Consensus Disagreement ────────────
# Synchronous spikes support consensus tension as an oscillation mechanism.
print("\n── Diagnostic 1: Val Loss vs Consensus Disagreement ──")
diag_lines += ["── Diagnostic 1: Val Loss vs Consensus Disagreement ──", ""]

for result in all_results:
    cname     = result['config_name']
    metrics   = all_metrics[cname]
    iters     = metrics['log_iters']
    val_loss  = metrics['val_loss_history']
    consensus = metrics['consensus_disagreement_history']

    r, p     = spike_correlation(val_loss, consensus)
    amp_val  = oscillation_amplitude(val_loss)
    amp_cons = oscillation_amplitude(consensus)
    r_str = f"{r:.3f}" if not np.isnan(r) else "N/A"
    p_str = f"{p:.3f}" if not np.isnan(p) else "N/A"

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(iters, val_loss,  color='steelblue', lw=2, label='Val Loss')
    ax2.plot(iters, consensus, color='darkorange', lw=2, linestyle='--', label='Consensus Disagreement')
    ax1.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Val Loss',               color='steelblue',  fontsize=10, fontweight='bold')
    ax2.set_ylabel('Consensus Disagreement', color='darkorange', fontsize=10, fontweight='bold')
    ax1.set_title(f'Diag-1: Val Loss vs Consensus | {cname}\n'
                  f'Pearson r={r_str}  (p={p_str})  |  '
                  f'Amp(val)={amp_val:.4f}  Amp(cons)={amp_cons:.6f}',
                  fontsize=10, fontweight='bold')
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'{run_prefix}_diag1_val_vs_consensus_{cname.replace("/","_")}.png', dpi=130, bbox_inches='tight')
    plt.show()

    if not np.isnan(r) and r > 0.5 and p < 0.05:
        verdict = ("CONSENSUS TENSION likely dominant cause. "
                   "→ Consider increasing γ or using a denser graph.")
    elif not np.isnan(r) and r < 0.2:
        verdict = "Consensus tension NOT the primary cause. Look at stochastic noise or bilevel coupling."
    else:
        verdict = "Moderate correlation — consensus tension partially contributes."

    print(f"  {cname}: r={r_str}, p={p_str} → {verdict}")
    diag_lines += [f"  Config : {cname}",
                   f"  Pearson r(val_loss, consensus) = {r_str}  (p={p_str})",
                   f"  Amp(val_loss) = {amp_val:.4f}  |  Amp(consensus) = {amp_cons:.6f}",
                   f"  → {verdict}", ""]


# ── Diagnostic 2 — Amplitude vs η_y ──────────────────────────────
# Within matched configurations, increasing amplitude with eta_y indicates
# lower-level overshooting or stochastic-update sensitivity.
print("\n── Diagnostic 2: Oscillation amplitude vs η_y ──")
diag_lines += ["── Diagnostic 2: Oscillation amplitude vs η_y ──", ""]

groups = defaultdict(list)
for result in all_results:
    groups[(result['eta_x'], result['gamma'], result['beta'])].append(result)

for key, group in groups.items():
    if len(group) < 2:
        continue
    eta_x_g, gamma_g, beta_g = key
    group_sorted = sorted(group, key=lambda r: r['eta_y'])
    etas = [r['eta_y'] for r in group_sorted]
    amps = [oscillation_amplitude(all_metrics[r['config_name']]['val_loss_history']) for r in group_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(etas, amps, marker='o', lw=2, color='crimson')
    for eta, amp in zip(etas, amps):
        if not np.isnan(amp):
            ax.annotate(f'{amp:.4f}', (eta, amp), textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center')
    ax.set_xscale('log')
    ax.set_xlabel('η_y (log scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Val Loss Oscillation Amplitude', fontsize=11, fontweight='bold')
    ax.set_title(f'Diag-2: Amplitude vs η_y\nη_x={eta_x_g:.0e}, γ={gamma_g:.2f}, β={beta_g:.0e}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'{run_prefix}_diag2_amp_vs_etay_etax{eta_x_g:.0e}_g{gamma_g:.2f}_b{beta_g:.0e}.png', dpi=130, bbox_inches='tight')
    plt.show()

    amps_arr = np.array(amps, dtype=float)
    if np.any(np.isnan(amps_arr)):
        verdict = "Could not assess — NaN amplitudes detected."
    else:
        ranks_amp = np.argsort(amps_arr)
        mono_inc  = all(ranks_amp[i] <= ranks_amp[i+1] for i in range(len(ranks_amp)-1))
        if mono_inc:
            verdict = "STOCHASTIC NOISE / η_y OVERSHOOTING confirmed. → Reduce η_y or add LR decay."
        elif amps_arr[0] == np.min(amps_arr):
            verdict = "Amplitude lowest at smallest η_y — partial η_y sensitivity."
        else:
            verdict = "Amplitude NOT monotone with η_y — stochastic noise is NOT the primary driver."

    print(f"  η_x={eta_x_g:.0e}, γ={gamma_g:.2f}, β={beta_g:.0e}: {verdict}")
    diag_lines += [f"  Group: η_x={eta_x_g:.0e}, γ={gamma_g:.2f}, β={beta_g:.0e}",
                   f"  η_y values : {etas}",
                   f"  Amplitudes : {[f'{a:.4f}' if not np.isnan(a) else 'NaN' for a in amps_arr]}",
                   f"  → {verdict}", ""]


# ── Diagnostic 3 — Amplitude vs γ ────────────────────────────────
# Sensitivity across matched gamma values identifies consensus coupling as a
# material source of instability.
print("\n── Diagnostic 3: Oscillation amplitude vs γ ──")
diag_lines += ["── Diagnostic 3: Oscillation amplitude vs γ ──", ""]

groups_g = defaultdict(list)
for result in all_results:
    groups_g[(result['eta_x'], result['eta_y'], result['beta'])].append(result)

for key, group in groups_g.items():
    if len(group) < 2:
        continue
    eta_x_g, eta_y_g, beta_g = key
    group_sorted = sorted(group, key=lambda r: r['gamma'])
    gammas = [r['gamma'] for r in group_sorted]
    amps   = [oscillation_amplitude(all_metrics[r['config_name']]['val_loss_history']) for r in group_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gammas, amps, marker='s', lw=2, color='darkorchid')
    for g, amp in zip(gammas, amps):
        if not np.isnan(amp):
            ax.annotate(f'{amp:.4f}', (g, amp), textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center')
    ax.set_xlabel('γ (consensus weight)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Val Loss Oscillation Amplitude', fontsize=11, fontweight='bold')
    ax.set_title(f'Diag-3: Amplitude vs γ\nη_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, β={beta_g:.0e}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'{run_prefix}_diag3_amp_vs_gamma_etax{eta_x_g:.0e}_etay{eta_y_g:.0e}_b{beta_g:.0e}.png', dpi=130, bbox_inches='tight')
    plt.show()

    amps_arr   = np.array(amps, dtype=float)
    valid_amps = amps_arr[~np.isnan(amps_arr)]
    if len(valid_amps) < 2:
        verdict = "Could not assess — insufficient non-NaN amplitudes."
    else:
        rel_range = (np.max(valid_amps) - np.min(valid_amps)) / (np.mean(valid_amps) + 1e-8)
        if rel_range > 0.3:
            verdict = f"CONSENSUS TENSION confirmed: amplitude varies {rel_range*100:.0f}% across γ. → Tune γ."
        else:
            verdict = f"Amplitude stable across γ (rel. range {rel_range*100:.0f}%). Consensus NOT primary driver."

    print(f"  η_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, β={beta_g:.0e}: {verdict}")
    diag_lines += [f"  Group: η_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, β={beta_g:.0e}",
                   f"  γ values   : {gammas}",
                   f"  Amplitudes : {[f'{a:.4f}' if not np.isnan(a) else 'NaN' for a in amps_arr]}",
                   f"  → {verdict}", ""]


# ── Diagnostic 4 — Amplitude vs β ────────────────────────────────
# A decrease with beta indicates that slowing the x dynamics relative to the
# y dynamics stabilizes the bilevel interaction.
print("\n── Diagnostic 4: Oscillation amplitude vs β ──")
diag_lines += ["── Diagnostic 4: Oscillation amplitude vs β ──", ""]

groups_b = defaultdict(list)
for result in all_results:
    groups_b[(result['eta_x'], result['eta_y'], result['gamma'])].append(result)

for key, group in groups_b.items():
    if len(group) < 2:
        continue
    eta_x_g, eta_y_g, gamma_g = key
    group_sorted = sorted(group, key=lambda r: r['beta'])
    betas = [r['beta'] for r in group_sorted]
    amps  = [oscillation_amplitude(all_metrics[r['config_name']]['val_loss_history']) for r in group_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(betas, amps, marker='^', lw=2, color='seagreen')
    for b, amp in zip(betas, amps):
        if not np.isnan(amp):
            ax.annotate(f'{amp:.4f}', (b, amp), textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center')
    ax.set_xlabel('β (regularization)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Val Loss Oscillation Amplitude', fontsize=11, fontweight='bold')
    ax.set_title(f'Diag-4: Amplitude vs β\nη_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, γ={gamma_g:.2f}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'{run_prefix}_diag4_amp_vs_beta_etax{eta_x_g:.0e}_etay{eta_y_g:.0e}_g{gamma_g:.2f}.png', dpi=130, bbox_inches='tight')
    plt.show()

    amps_arr = np.array(amps, dtype=float)
    if np.any(np.isnan(amps_arr)):
        verdict = "Could not assess — NaN amplitudes detected."
    else:
        mono_dec = all(amps_arr[i] >= amps_arr[i+1] for i in range(len(amps_arr)-1))
        if mono_dec:
            verdict = "BILEVEL COUPLING confirmed: amplitude strictly decreases with β. → Use larger β."
        elif amps_arr[-1] < amps_arr[0]:
            verdict = "Partial bilevel coupling effect. Increasing β helps but is not the sole fix."
        else:
            verdict = "Amplitude does NOT decrease with β. Bilevel coupling is NOT the primary driver."

    print(f"  η_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, γ={gamma_g:.2f}: {verdict}")
    diag_lines += [f"  Group: η_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, γ={gamma_g:.2f}",
                   f"  β values   : {betas}",
                   f"  Amplitudes : {[f'{a:.4f}' if not np.isnan(a) else 'NaN' for a in amps_arr]}",
                   f"  → {verdict}", ""]


# ── Diagnostic 5 — Gradient norm vs Val Loss ─────────────────────
# Correlated loss and gradient spikes point to update magnitude/noise rather
# than consensus disagreement as the immediate driver.
print("\n── Diagnostic 5: Gradient norm vs Val Loss ──")
diag_lines += ["── Diagnostic 5: Gradient norm vs Val Loss ──", ""]

has_grad_norms = any(
    'grad_norm_x_history' in all_metrics[r['config_name']]
    and len(all_metrics[r['config_name']]['grad_norm_x_history']) > 0
    for r in all_results
)

if not has_grad_norms:
    msg = ("  ⚠ 'grad_norm_x_history'/'grad_norm_y_history' not found. "
           "Add norm logging to run_DeViABO.py.")
    print(msg)
    diag_lines.append(msg)
else:
    for result in all_results:
        cname    = result['config_name']
        metrics  = all_metrics[cname]
        iters    = metrics['log_iters']
        val_loss = metrics['val_loss_history']
        gnx      = metrics.get('grad_norm_x_history', None)
        gny      = metrics.get('grad_norm_y_history', None)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(iters, val_loss, color='steelblue', lw=2, label='Val Loss')
        axes[0].set_ylabel('Val Loss', fontweight='bold'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

        r_x = float('nan')
        if gnx is not None and len(gnx) > 0:
            r_x, p_x = spike_correlation(val_loss, gnx)
            r_x_str = f"{r_x:.3f}" if not np.isnan(r_x) else "N/A"
            axes[1].plot(iters, gnx, color='crimson', lw=2, label='‖∇_x f‖')
            axes[1].set_ylabel('‖∇_x f‖', fontweight='bold')
            axes[1].legend(fontsize=8, title=f'r={r_x_str}'); axes[1].grid(True, alpha=0.3)

        r_y = float('nan')
        if gny is not None and len(gny) > 0:
            r_y, p_y = spike_correlation(val_loss, gny)
            r_y_str = f"{r_y:.3f}" if not np.isnan(r_y) else "N/A"
            axes[2].plot(iters, gny, color='darkorange', lw=2, label='‖∇_y g‖')
            axes[2].set_ylabel('‖∇_y g‖', fontweight='bold')
            axes[2].legend(fontsize=8, title=f'r={r_y_str}'); axes[2].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Communication Round', fontweight='bold')
        fig.suptitle(f'Diag-5: Gradient Norms vs Val Loss | {cname}', fontsize=11, fontweight='bold')
        fig.tight_layout()
        plt.savefig(f'{run_prefix}_diag5_gradnorm_{cname.replace("/","_")}.png', dpi=130, bbox_inches='tight')
        plt.show()

        r_x_str = f"{r_x:.3f}" if not np.isnan(r_x) else "N/A"
        r_y_str = f"{r_y:.3f}" if not np.isnan(r_y) else "N/A"
        if (not np.isnan(r_x) and r_x > 0.5) or (not np.isnan(r_y) and r_y > 0.5):
            verdict = "STOCHASTIC GRADIENT NOISE confirmed. → Reduce η_x/η_y or add gradient clipping."
        else:
            verdict = "Gradient norms do NOT strongly correlate with val loss spikes."
        diag_lines += [f"  Config: {cname}", f"  r(val,‖∇_x‖)={r_x_str}  r(val,‖∇_y‖)={r_y_str}",
                       f"  → {verdict}", ""]


# ── Diagnostic 6 — Per-node val loss decomposition ───────────────
# Contrast nodes connected across data groups with nodes whose neighborhoods
# remain within-group to expose topology–heterogeneity interaction.
print("\n── Diagnostic 6: Per-node val loss decomposition ──")
diag_lines += ["── Diagnostic 6: Per-node val loss decomposition ──", ""]

has_node_val = any(
    'node_val_loss_history' in all_metrics[r['config_name']]
    and len(all_metrics[r['config_name']]['node_val_loss_history']) > 0
    for r in all_results
)

if not has_node_val:
    msg = ("  ⚠ 'node_val_loss_history' not found. "
           "Add per-node val loss logging to run_DeViABO.py.")
    print(msg)
    diag_lines.append(msg)
else:
    node_colors = plt.cm.Set1(np.linspace(0, 1, K_NODES))
    for result in all_results:
        cname    = result['config_name']
        metrics  = all_metrics[cname]
        iters    = metrics['log_iters']
        node_val = np.array(metrics['node_val_loss_history'], dtype=float)

        group_of      = [i // (K_NODES // num_groups) for i in range(K_NODES)]
        neighborhoods = metrics['neighborhoods']
        inter_nodes, intra_nodes = [], []
        for i in range(K_NODES):
            has_inter = any(group_of[j] != group_of[i] for j in neighborhoods[i] if j != i)
            (inter_nodes if has_inter else intra_nodes).append(i)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(K_NODES):
            style = '-' if i in inter_nodes else '--'
            label = f'Node {i} (inter, g={group_of[i]})' if i in inter_nodes else f'Node {i} (intra, g={group_of[i]})'
            ax.plot(iters, node_val[:, i], lw=2, linestyle=style, color=node_colors[i], label=label)
        ax.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
        ax.set_ylabel('Node Val Loss', fontsize=11, fontweight='bold')
        ax.set_title(f'Diag-6: Per-node Val Loss | {cname}\nSolid=inter-group, Dashed=intra-group',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(f'{run_prefix}_diag6_pernode_val_{cname.replace("/","_")}.png', dpi=130, bbox_inches='tight')
        plt.show()

        amp_inter = float(np.nanmean([oscillation_amplitude(node_val[:, i]) for i in inter_nodes])) if inter_nodes else 0.0
        amp_intra = float(np.nanmean([oscillation_amplitude(node_val[:, i]) for i in intra_nodes])) if intra_nodes else 0.0
        if inter_nodes and amp_inter > 1.5 * amp_intra:
            verdict = (f"NON-IID DIVERGENCE confirmed: inter-group ×{amp_inter/max(amp_intra,1e-8):.1f} more. "
                       "→ Increase Dirichlet α or clip x-updates.")
        elif inter_nodes and amp_inter > amp_intra:
            verdict = f"Partial non-IID effect (×{amp_inter/max(amp_intra,1e-8):.2f})."
        else:
            verdict = "Oscillation uniform — non-IID topology is NOT the primary driver."

        print(f"  {cname}: inter-amp={amp_inter:.4f}, intra-amp={amp_intra:.4f} → {verdict}")
        diag_lines += [f"  Config: {cname}",
                       f"  Amp(inter)={amp_inter:.4f}  Amp(intra)={amp_intra:.4f}",
                       f"  → {verdict}", ""]


# ── Diagnostic Summary ────────────────────────────────────────────
# Consolidate the available signatures and rank configurations by persistent
# post-burn-in oscillation amplitude.
print("\n── Oscillation Diagnostic Summary ──")
diag_lines += ["", "── Oscillation Diagnostic Summary ──", ""]

summary_verdicts = []
for result in all_results:
    cname     = result['config_name']
    val_loss  = all_metrics[cname]['val_loss_history']
    consensus = all_metrics[cname]['consensus_disagreement_history']
    amp       = oscillation_amplitude(val_loss)
    r_c, _    = spike_correlation(val_loss, consensus)

    causes = []
    if not np.isnan(r_c) and r_c > 0.5:
        causes.append('CONSENSUS TENSION')
    if result['eta_y'] == max(eta_y_values):
        causes.append('POSSIBLE η_y OVERSHOOTING')
    if not causes:
        causes.append('UNDETERMINED — add grad_norm and per-node logging')

    verdict_str = ' + '.join(causes)
    amp_str = f"{amp:.4f}" if not np.isnan(amp) else "N/A"
    r_c_str = f"{r_c:.3f}" if not np.isnan(r_c) else "N/A"
    summary_verdicts.append((cname, amp if not np.isnan(amp) else -1.0, r_c, verdict_str))

    print(f"  {cname}\n    Amplitude: {amp_str}  r(val,cons): {r_c_str}\n    Verdict: {verdict_str}\n")
    diag_lines += [f"  {cname}", f"    Amplitude: {amp_str}  r(val,cons): {r_c_str}",
                   f"    Verdict  : {verdict_str}", ""]

summary_verdicts.sort(key=lambda x: x[1], reverse=True)
diag_lines += ["  Ranked by oscillation amplitude (most → least):", ""]
for cname, amp, r_c, verdict in summary_verdicts:
    amp_str = f"{amp:.4f}" if amp >= 0 else "N/A"
    line = f"    [{amp_str}] {cname}  —  {verdict}"
    print(line); diag_lines.append(line)

diag_filename = f'{run_prefix}_oscillation_diagnostics.txt'
with open(diag_filename, 'w') as f:
    f.write('\n'.join(diag_lines))
print(f"\n✓ Oscillation diagnostic report saved to '{diag_filename}'")


# ═════════════════════════════════════════════════════════════════
# SAVE EXPERIMENT SUMMARY TEXT FILE
# Store setup, complete trajectories, rankings, and the selected run together
# so results can be interpreted without reconstructing console output.
# ═════════════════════════════════════════════════════════════════
sep  = "=" * 80
sep2 = "-" * 80

summary_lines = [
    sep, "  DeViABO EXPERIMENT SUMMARY — CIFAR-10", sep, "",
    "── EXPERIMENT SETUP ──────────────────────────────────────────────────────────",
    f"  Dataset          : CIFAR-10",
    f"  Graph type       : {GRAPH_TYPE}",
    f"  Nodes (K)        : {K_NODES}",
    f"  Iterations (T)   : {T}",
    f"  Batch size       : {BATCH_SIZE}",
    f"  Seed             : {SEED}",
    f"  Data split mode  : {mode}",
    f"  Dirichlet α      : {dir_level}",
    f"  num_groups       : {num_groups}",
    f"  Val ratio        : {VAL_RATIO}",
    "",
    "── MODEL ARCHITECTURE ────────────────────────────────────────────────────────",
    f"  Model            : CNNCifar10Deep",
    "",
    "── NORMALISATION ─────────────────────────────────────────────────────────────",
    f"  Train            : RandomCrop(32, pad=4) + RandomHFlip + Normalize",
    f"  Val / Test       : Normalize only",
    f"  Mean             : (0.4914, 0.4822, 0.4465)",
    f"  Std              : (0.2023, 0.1994, 0.2010)",
    "",
    "── HYPERPARAMETER GRID ───────────────────────────────────────────────────────",
    f"  eta_x values     : {eta_x_values}",
    f"  eta_y values     : {eta_y_values}",
    f"  gamma values     : {gamma_values}",
    f"  beta  values     : {beta_values}",
    f"  Total configs    : {total_configs}",
    "",
]

for rank, (_, row) in enumerate(df_results.iterrows(), start=1):
    cname   = row['config_name']
    metrics = all_metrics[cname]
    is_best = cname == best_config['config_name']
    iters   = metrics['log_iters']
    log_gap = int(iters[1] - iters[0]) if len(iters) > 1 else 0

    te_acc_str = ', '.join(f"{v*100:.2f}%" for v in metrics['test_acc_history'])
    te_los_str = ', '.join(f"{v:.4f}"       for v in metrics['test_loss_history'])
    va_los_str = ', '.join(f"{v:.4f}"       for v in metrics['val_loss_history'])
    tr_los_str = ', '.join(f"{v:.4f}"       for v in metrics['train_loss_history'])
    cons_str   = ', '.join(f"{v:.6f}"       for v in metrics['consensus_disagreement_history'])
    iters_str  = ', '.join(str(i)           for i in iters)

    summary_lines += [
        "", sep2,
        f"  Rank {rank:>2d}{'  ★ BEST ★' if is_best else ''}  |  {cname}", sep2,
        f"  eta_x={row['eta_x']:.0e}  eta_y={row['eta_y']:.0e}  gamma={row['gamma']:.2f}  beta={row['beta']:.0e}",
        "",
        f"  FINAL METRICS (iter {iters[-1]}):",
        f"    Test  Accuracy  : {row['final_test_acc']:.2f}%",
        f"    Test  Loss      : {row['final_test_loss']:.4f}",
        f"    Val   Loss      : {row['final_val_loss']:.4f}",
        f"    Train Loss      : {row['final_train_loss']:.4f}",
        f"    Consensus Dis.  : {row['final_consensus']:.6f}",
        f"    Intra/Inter w.  : {row['weight_ratio']:.4f}",
        f"    Training time   : {row['time_seconds']:.1f}s",
        "",
        f"  FULL HISTORY (logged every {log_gap} iters):",
        f"    Iters     : [{iters_str}]",
        f"    Test  Acc : [{te_acc_str}]",
        f"    Test  Loss: [{te_los_str}]",
        f"    Val   Loss: [{va_los_str}]",
        f"    Train Loss: [{tr_los_str}]",
        f"    Consensus : [{cons_str}]",
    ]

summary_lines += [
    "", sep, "  BEST CONFIGURATION SUMMARY", sep,
    f"  Config         : {best_config['config_name']}",
    f"  eta_x          : {best_config['eta_x']:.0e}",
    f"  eta_y          : {best_config['eta_y']:.0e}",
    f"  gamma          : {best_config['gamma']:.2f}",
    f"  beta           : {best_config['beta']:.0e}",
    f"  Test Accuracy  : {best_config['final_test_acc']:.2f}%",
    f"  Test Loss      : {best_config['final_test_loss']:.4f}",
    f"  Val Loss       : {best_config['final_val_loss']:.4f}",
    f"  Train Loss     : {best_config['final_train_loss']:.4f}",
    f"  Consensus Dis. : {best_config['final_consensus']:.6f}",
    f"  Intra/Inter w. : {best_config['weight_ratio']:.4f}",
    f"  Training time  : {best_config['time_seconds']:.1f}s",
    sep, "",
]

summary_filename = f'{run_prefix}_deviabo_summary.txt'
with open(summary_filename, "w") as f:
    f.write("\n".join(summary_lines))
print(f"\n✓ Experiment summary saved to '{summary_filename}'")