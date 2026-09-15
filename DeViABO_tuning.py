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
import time
from tqdm import trange
import os
import torch.nn.functional as F
from help_functions import split_node_data_non_iid, generate_graph,\
                            compute_pairwise_disagreement, compute_consensus_disagreement,\
                            compute_midpoint_grads, compute_stochastic_grad
from run_DeViABO import run_deviabo
import itertools
from model import CNNFashion_Mnist
from scipy.stats import pearsonr
from collections import defaultdict


import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import trange

from help_functions import (
    generate_graph,
    compute_pairwise_disagreement,
    compute_consensus_disagreement,
    compute_midpoint_grads,
    compute_stochastic_grad,
)


# Cap host-side math threads so parallel experiment processes do not
# oversubscribe CPU cores while the main work runs on the GPU.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# ═════════════════════════════════════════════════════════════════
# DeViABO Hyperparameter Tuning and Plotting Script — FashionMNIST
# ═════════════════════════════════════════════════════════════════
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


GRAPH_TYPE   = 'ring' #'SBM' #'caveman' #'fully connected', 'ring' , 'line' ,'star' , 'Grid'

eta_y_values = [0.5] 
gamma_values = [4.0] 
eta_x_values = [5e-5]
beta_values =  [1e-5]


T           = 1000 #4000
SEED        = 42
BATCH_SIZE  = 100
hidden_size = (120, 84)
conv_size   = (6, 16)
kernel_size = 5
padding     = 2

K_NODES    = 10 #21 #14 
VAL_RATIO  = 0.2
mode       = 'dirichlet' #'clustered_dirichlet' 
num_informative_nodes = 1   # I — number of informative nodes (I < K)
num_informative_patterns = 3  # P — number of exclusive patterns (P < n_classes)
dir_level_I = 10.0  

dir_level  =   0.3 #10.0

num_groups = 2 #3 

if mode == 'dirichlet':
    run_prefix = (
        f"{GRAPH_TYPE.replace(' ', '_')}_K{K_NODES}"
        f"_{mode}_alpha{dir_level}_T{T}_beta_[{beta_values}]_gamma_[{gamma_values}]_eta_x_[{eta_x_values}]_eta_y_[{eta_y_values}]"
    )
elif mode == 'clustered_dirichlet':
    run_prefix = (
            f"{GRAPH_TYPE.replace(' ', '_')}_K{K_NODES}"
            f"_{mode}_groups_[{num_groups}]_alpha{dir_level}_T{T}_beta_[{beta_values}]_gamma_[{gamma_values}]_eta_x_[{eta_x_values}]_eta_y_[{eta_y_values}]"
        )
else:
    run_prefix = (
            f"{GRAPH_TYPE.replace(' ', '_')}_K{K_NODES}"
            f"_{mode}_groups_[{num_groups}]_alpha{dir_level}_T{T}"
        )
# ── Incremental results file (written after EACH experiment) ──────
incremental_filename = (
    f"DeViABO_incremental_K{K_NODES}_{GRAPH_TYPE.replace(' ', '_')}"
    f"_T{T}_{mode}_dir_[{dir_level}].txt"
)
def _write_incremental_result(filepath, config_id, total_configs,
                               result, metrics, first_entry):
    """Append one finished experiment to the incremental log.

    The first call writes a self-describing header; later calls append. This
    keeps a long sweep recoverable if it is interrupted or a config fails.
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
            "  DeViABO INCREMENTAL EXPERIMENT LOG — FashionMNIST",
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
# Dataset preparation
# Normalize to [-1, 1] so the tanh-based CNN receives zero-centered inputs.
# ─────────────────────────────────────────────────────────────────
transform_fm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

DATASET      = 'fashion-mnist'
dataset_path = f'./datasets/{DATASET}'
dataset_train = datasets.FashionMNIST(root=dataset_path, download=False, transform=transform_fm)
dataset_test  = datasets.FashionMNIST(root=dataset_path, train=False, download=False, transform=transform_fm)
number_classes, n_channels, img_size = 10, 1, 28

print("dataset_train_counts:", torch.bincount(dataset_train.targets))
print("dataset_test_counts: ", torch.bincount(dataset_test.targets))


# ─────────────────────────────────────────────────────────────────
# Data split
# Build the non-IID partition once so every hyperparameter configuration is
# compared on an identical node-level data allocation.
# ─────────────────────────────────────────────────────────────────
data_chunks, test_data = split_node_data_non_iid(
    dataset_train, dataset_test,
    K=K_NODES, val_ratio=VAL_RATIO,
    seed=SEED, mode=mode,
    alpha=dir_level,
    num_groups=num_groups,
    num_informative_nodes = num_informative_nodes,   # I — number of informative nodes (I < K)
    num_informative_patterns = num_informative_patterns,  # P — number of exclusive patterns (P < n_classes)
    dir_level_I = dir_level_I,  
)



# ─────────────────────────────────────────────────────────────────
# DataLoaders
# Move each local partition onto the device up front to avoid repeated
# host-to-device transfers inside the optimization loop.
# ─────────────────────────────────────────────────────────────────
train_loaders = []
val_loaders   = []

for node_id in range(K_NODES):
    train_data, train_labels = data_chunks[node_id]['train']
    train_data_gpu   = train_data.to(device, dtype=torch.float32) / 255.0
    train_labels_gpu = train_labels.to(device, dtype=torch.long)
    train_loaders.append(DataLoader(TensorDataset(train_data_gpu, train_labels_gpu),
                                    batch_size=BATCH_SIZE, shuffle=True))

    val_data, val_labels = data_chunks[node_id]['val']
    if len(val_data) > 0:
        val_data_gpu   = val_data.to(device, dtype=torch.float32) / 255.0
        val_labels_gpu = val_labels.to(device, dtype=torch.long)
        val_loaders.append(DataLoader(TensorDataset(val_data_gpu, val_labels_gpu),
                                      batch_size=BATCH_SIZE, shuffle=False))
    else:
        val_loaders.append(None)

test_imgs, test_labels = test_data
test_imgs_gpu   = test_imgs.to(device, dtype=torch.float32) / 255.0
test_labels_gpu = test_labels.to(device, dtype=torch.long)
test_loader = DataLoader(TensorDataset(test_imgs_gpu, test_labels_gpu),
                         batch_size=BATCH_SIZE, shuffle=False)


# ── Model factory ─────────────────────────────────────────────────
# Return a fresh model per call so each node starts from independent
# parameters sharing the same architecture.
model_factory = lambda: CNNFashion_Mnist(
    n_class     = number_classes,
    conv_size   = conv_size,
    kernel_size = kernel_size,
    hidden_size = hidden_size,
    padding     = padding,
    device      = device,
)


# ─────────────────────────────────────────────────────────────────
# Hyperparameter Grid
# ─────────────────────────────────────────────────────────────────
total_configs = (len(eta_x_values) * len(eta_y_values)
                 * len(gamma_values) * len(beta_values))

all_results = []
all_metrics = {}

print("=" * 80)
print("DeViABO HYPERPARAMETER TUNING — FashionMNIST")
print(f"Total configurations: {total_configs}")
print(f"K_NODES={K_NODES}, graph={GRAPH_TYPE}, T={T}, batch={BATCH_SIZE}, "
      f"mode={mode}, alpha={dir_level}")
print("=" * 80)


# ─────────────────────────────────────────────────────────────────
# Grid Search Loop
# Iterate the full Cartesian product to retain every interaction among the
# upper-level, lower-level, consensus, and regularization parameters.
# ─────────────────────────────────────────────────────────────────
config_id = 0
first_entry  = True   # first write creates the log; later writes append
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
            log_every     = 10,
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

        # ── Persist this run before the next config so an interrupted ─
        #    sweep still retains every completed experiment. ───────────
        _write_incremental_result(
            filepath      = incremental_filename,
            config_id     = config_id,
            total_configs = total_configs,
            result        = result,
            metrics       = deviabo_metrics,
            first_entry   = first_entry,
        )
        first_entry = False
        # ──────────────────────────────────────────────────────────

        print(f"  ✓ Final Acc: {final_test_acc*100:.2f}% | "
              f"Val Loss: {final_val_loss:.4f} | "
              f"Weight Ratio: {weight_ratio:.3f} | "
              f"Time: {elapsed_time:.1f}s  → saved to '{incremental_filename}'")

    except Exception as e:
        import traceback
        print(f"  ✗ FAILED: {str(e)}")
        traceback.print_exc()
        # ── Log failure immediately too ───────────────────────────
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
# Rank by test accuracy; the top row drives highlighting in every plot.
# ═════════════════════════════════════════════════════════════════
best_config = df_results.iloc[0]

print("\n" + "=" * 80)
print("★ BEST HYPERPARAMETER CONFIGURATION (by Test Accuracy) ★")
print("=" * 80)
print(f"Configuration : {best_config['config_name']}")
print(f"η_x           : {best_config['eta_x']:.0e}")
print(f"η_y           : {best_config['eta_y']:.0e}")
print(f"γ (consensus) : {best_config['gamma']:.0e}")
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

# ── Append best-config banner to incremental file ────────────────
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


def _plot_metric(metric_key, ylabel, title, filename,
                 transform=None, marker='o'):
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, result in enumerate(all_results):
        cname   = result['config_name']
        metrics = all_metrics[cname]
        iters   = metrics['log_iters']
        values  = metrics[metric_key]
        if transform:
            values = [transform(v) for v in values]
        is_best = cname == best_config['config_name']
        ax.plot(
            iters, values,
            lw        = 3.5 if is_best else 2,
            label     = f"{cname} ★ BEST" if is_best else cname,
            color     = 'red' if is_best else colors[idx],
            alpha     = 1.0 if is_best else 0.7,
            marker    = marker,
            markersize= 5 if is_best else 3,
            markevery = 5,
            zorder    = 10 if is_best else 1,
        )
    ax.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'{run_prefix}_{filename}', dpi=150, bbox_inches='tight')
    plt.show()


def _plot_grad_norm(norm_key, ylabel, title, filename, marker='o'):
    """Plot a gradient-norm history across configs, skipping runs that lack it.

    Keeping the analysis optional lets it coexist with lighter logging modes
    that do not return gradient norms.
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
        ax.plot(
            iters, values,
            lw        = 3.5 if is_best else 2,
            label     = f"{cname} ★ BEST" if is_best else cname,
            color     = 'red' if is_best else colors[idx],
            alpha     = 1.0 if is_best else 0.7,
            marker    = marker,
            markersize= 5 if is_best else 3,
            markevery = 5,
            zorder    = 10 if is_best else 1,
        )
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
             'Test Accuracy (%)', 'DeViABO — FashionMNIST: Test Accuracy',
             'deviabo_fm_test_accuracy.png',
             transform=lambda v: v * 100, marker='o')

_plot_metric('val_loss_history',
             'Val Loss (Aggregated)', 'DeViABO — FashionMNIST: Validation Loss',
             'deviabo_fm_val_loss.png',
             marker='D')

_plot_metric('train_loss_history',
             'Train Loss', 'DeViABO — FashionMNIST: Train Loss',
             'deviabo_fm_train_loss.png',
             marker='s')

_plot_metric('test_loss_history',
             'Test Loss', 'DeViABO — FashionMNIST: Test Loss',
             'deviabo_fm_test_loss.png',
             marker='^')

_plot_metric('consensus_disagreement_history',
             'Consensus Disagreement', 'DeViABO — FashionMNIST: Consensus Disagreement',
             'deviabo_fm_consensus_disagreement.png',
             marker='d')


# ═════════════════════════════════════════════════════════════════
# GRADIENT NORM PLOTS
# Gradient-norm trajectories separate optimizer instability from
# disagreement introduced by decentralized communication.
# ═════════════════════════════════════════════════════════════════
has_grad_norms_plot = any(
    'grad_norm_x_history' in all_metrics[r['config_name']]
    and len(all_metrics[r['config_name']]['grad_norm_x_history']) > 0
    for r in all_results
)

if has_grad_norms_plot:

    # ── ‖∇_x f‖ all configs ──────────────────────────────────────
    _plot_grad_norm(
        'grad_norm_x_history',
        '‖∇_x f‖  (mean over nodes)',
        'DeViABO — FashionMNIST: Upper-level Gradient Norm  ‖∇_x f‖',
        'deviabo_fm_grad_norm_x.png',
        marker='o',
    )

    # ── ‖∇_y g‖ all configs ──────────────────────────────────────
    _plot_grad_norm(
        'grad_norm_y_history',
        '‖∇_y g‖  (mean over nodes)',
        'DeViABO — FashionMNIST: Lower-level Gradient Norm  ‖∇_y g‖',
        'deviabo_fm_grad_norm_y.png',
        marker='s',
    )

    # ── Combined 3-panel for BEST config ─────────────────────────
    best_cname   = best_config['config_name']
    best_metrics = all_metrics[best_cname]
    best_iters   = best_metrics['log_iters']
    best_val     = best_metrics['val_loss_history']
    best_gnx     = best_metrics.get('grad_norm_x_history', [])
    best_gny     = best_metrics.get('grad_norm_y_history', [])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(best_iters, best_val,
                 color='steelblue', lw=2.5, label='Val Loss')
    axes[0].set_ylabel('Val Loss', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    if best_gnx:
        axes[1].plot(best_iters, best_gnx,
                     color='crimson', lw=2.5, label='‖∇_x f‖')
        axes[1].set_ylabel('‖∇_x f‖', fontsize=10, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

    if best_gny:
        axes[2].plot(best_iters, best_gny,
                     color='darkorange', lw=2.5, label='‖∇_y g‖')
        axes[2].set_ylabel('‖∇_y g‖', fontsize=10, fontweight='bold')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Communication Round', fontsize=11, fontweight='bold')
    fig.suptitle(
        f'DeViABO — FashionMNIST: Val Loss & Gradient Norms\n★ BEST: {best_cname}',
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout()
    plt.savefig(f'{run_prefix}_deviabo_fm_best_val_and_gradnorms.png',
                dpi=150, bbox_inches='tight')
    plt.show()

else:
    print("\n⚠ Gradient norm plots skipped — 'grad_norm_x_history' / "
          "'grad_norm_y_history' not returned by run_deviabo.\n"
          "  Add norm logging to run_DeViABO.py as described in Diagnostic 5.")


# ═════════════════════════════════════════════════════════════════════════
# OSCILLATION DIAGNOSTICS
# Each diagnostic varies one explanatory factor at a time — consensus
# coupling, parameter sensitivity, gradient behavior, or node heterogeneity —
# to attribute validation-loss oscillation to a specific cause.
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("OSCILLATION DIAGNOSTICS")
print("=" * 80)

diag_lines = [
    "=" * 80,
    "  OSCILLATION DIAGNOSTICS — DeViABO FashionMNIST",
    "=" * 80, "",
]


# ── Helpers ───────────────────────────────────────────────────────
def oscillation_amplitude(series):
    """Peak-to-peak amplitude after a 20% burn-in.

    Discarding the transient focuses the statistic on persistent oscillation
    rather than expected early optimization movement.
    """
    s = np.array(series, dtype=float)
    burnin = max(1, len(s) // 5)
    s = s[burnin:]
    if np.any(np.isnan(s)):
        return float('nan')
    return float(np.max(s) - np.min(s))


def spike_correlation(a, b):
    """Pearson r and p-value between two series; NaN when undefined.

    Returns NaN if fewer than three finite samples remain or variance is
    negligible, so uninformative correlations are not reported as signal.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan'), float('nan')
    r, p = pearsonr(a, b)
    return float(r), float(p)


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC 1 — Val Loss vs Consensus Disagreement overlay
# High positive correlation implies consensus tension drives the spikes.
# ─────────────────────────────────────────────────────────────────
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

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(iters, val_loss,  color='steelblue',  lw=2, label='Val Loss')
    ax2.plot(iters, consensus, color='darkorange',  lw=2,
             linestyle='--', label='Consensus Disagreement')
    ax1.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Val Loss',               color='steelblue',  fontsize=10, fontweight='bold')
    ax2.set_ylabel('Consensus Disagreement', color='darkorange', fontsize=10, fontweight='bold')
    r_str = f"{r:.3f}" if not np.isnan(r) else "N/A"
    p_str = f"{p:.3f}" if not np.isnan(p) else "N/A"
    ax1.set_title(
        f'Diag-1: Val Loss vs Consensus | {cname}\n'
        f'Pearson r={r_str}  (p={p_str})  |  '
        f'Amp(val)={amp_val:.4f}  Amp(cons)={amp_cons:.6f}',
        fontsize=10, fontweight='bold'
    )
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f'{run_prefix}_diag1_val_vs_consensus_{cname.replace("/","_")}.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.show()

    if not np.isnan(r) and r > 0.5 and p < 0.05:
        verdict = ("CONSENSUS TENSION likely dominant cause. "
                   "Val loss spikes co-occur with consensus disagreement spikes. "
                   "→ Consider increasing γ or using a denser graph.")
    elif not np.isnan(r) and r < 0.2:
        verdict = ("Consensus tension NOT the primary cause "
                   "(low correlation). Look at stochastic noise or bilevel coupling.")
    else:
        verdict = "Moderate correlation — consensus tension partially contributes."

    print(f"  {cname}: r={r_str}, p={p_str} → {verdict}")
    diag_lines += [
        f"  Config : {cname}",
        f"  Pearson r(val_loss, consensus) = {r_str}  (p={p_str})",
        f"  Amp(val_loss) = {amp_val:.4f}  |  Amp(consensus) = {amp_cons:.6f}",
        f"  → {verdict}", "",
    ]


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC 2 — Oscillation amplitude vs η_y
# Hold the other parameters fixed; amplitude rising with η_y indicates
# lower-level step-size overshooting.
# ─────────────────────────────────────────────────────────────────
print("\n── Diagnostic 2: Oscillation amplitude vs η_y ──")
diag_lines += ["── Diagnostic 2: Oscillation amplitude vs η_y ──", ""]

groups = defaultdict(list)
for result in all_results:
    key = (result['eta_x'], result['gamma'], result['beta'])
    groups[key].append(result)

for key, group in groups.items():
    if len(group) < 2:
        continue
    eta_x_g, gamma_g, beta_g = key
    group_sorted = sorted(group, key=lambda r: r['eta_y'])
    etas = [r['eta_y'] for r in group_sorted]
    amps = [oscillation_amplitude(
                all_metrics[r['config_name']]['val_loss_history'])
            for r in group_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(etas, amps, marker='o', lw=2, color='crimson')
    for eta, amp in zip(etas, amps):
        if not np.isnan(amp):
            ax.annotate(f'{amp:.4f}', (eta, amp),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=8, ha='center')
    ax.set_xscale('log')
    ax.set_xlabel('η_y (log scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Val Loss Oscillation Amplitude', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Diag-2: Amplitude vs η_y\n'
        f'η_x={eta_x_g:.0e}, γ={gamma_g:.2f}, β={beta_g:.0e}',
        fontsize=10, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = (f'{run_prefix}_diag2_amp_vs_etay'
             f'_etax{eta_x_g:.0e}_g{gamma_g:.2f}_b{beta_g:.0e}.png')
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.show()

    amps_arr = np.array(amps, dtype=float)
    if np.any(np.isnan(amps_arr)):
        verdict = "Could not assess — NaN amplitudes detected (check val_loader)."
    else:
        ranks_amp = np.argsort(amps_arr)
        mono_increasing = all(ranks_amp[i] <= ranks_amp[i + 1]
                              for i in range(len(ranks_amp) - 1))
        if mono_increasing:
            verdict = ("STOCHASTIC NOISE / η_y OVERSHOOTING confirmed. "
                       "Amplitude strictly increases with η_y. "
                       "→ Reduce η_y or add LR decay.")
        elif amps_arr[0] == np.min(amps_arr):
            verdict = ("Amplitude lowest at smallest η_y — partial η_y sensitivity. "
                       "Reducing η_y will help but is not the only cause.")
        else:
            verdict = ("Amplitude NOT monotone with η_y — stochastic noise is NOT "
                       "the primary driver. Focus on consensus/topology diagnostics.")

    print(f"  η_x={eta_x_g:.0e}, γ={gamma_g:.2f}, β={beta_g:.0e}: {verdict}")
    diag_lines += [
        f"  Group: η_x={eta_x_g:.0e}, γ={gamma_g:.2f}, β={beta_g:.0e}",
        f"  η_y values : {etas}",
        f"  Amplitudes : {[f'{a:.4f}' if not np.isnan(a) else 'NaN' for a in amps_arr]}",
        f"  → {verdict}", "",
    ]


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC 3 — Oscillation amplitude vs γ
# Large amplitude variation across γ points to consensus weight as a driver.
# ─────────────────────────────────────────────────────────────────
print("\n── Diagnostic 3: Oscillation amplitude vs γ ──")
diag_lines += ["── Diagnostic 3: Oscillation amplitude vs γ ──", ""]

groups_g = defaultdict(list)
for result in all_results:
    key = (result['eta_x'], result['eta_y'], result['beta'])
    groups_g[key].append(result)

for key, group in groups_g.items():
    if len(group) < 2:
        continue
    eta_x_g, eta_y_g, beta_g = key
    group_sorted = sorted(group, key=lambda r: r['gamma'])
    gammas = [r['gamma'] for r in group_sorted]
    amps   = [oscillation_amplitude(
                  all_metrics[r['config_name']]['val_loss_history'])
              for r in group_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gammas, amps, marker='s', lw=2, color='darkorchid')
    for g, amp in zip(gammas, amps):
        if not np.isnan(amp):
            ax.annotate(f'{amp:.4f}', (g, amp),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=8, ha='center')
    ax.set_xlabel('γ (consensus weight)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Val Loss Oscillation Amplitude', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Diag-3: Amplitude vs γ\n'
        f'η_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, β={beta_g:.0e}',
        fontsize=10, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = (f'{run_prefix}_diag3_amp_vs_gamma'
             f'_etax{eta_x_g:.0e}_etay{eta_y_g:.0e}_b{beta_g:.0e}.png')
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.show()

    amps_arr   = np.array(amps, dtype=float)
    valid_amps = amps_arr[~np.isnan(amps_arr)]
    if len(valid_amps) < 2:
        verdict = "Could not assess — insufficient non-NaN amplitudes."
    else:
        amp_range = float(np.max(valid_amps) - np.min(valid_amps))
        amp_mean  = float(np.mean(valid_amps))
        rel_range = amp_range / (amp_mean + 1e-8)
        if rel_range > 0.3:
            verdict = (f"CONSENSUS TENSION confirmed: amplitude varies "
                       f"{rel_range*100:.0f}% relative to mean across γ values. "
                       "→ γ is a key factor; tuning it reduces oscillation.")
        else:
            verdict = (f"Amplitude stable across γ (rel. range {rel_range*100:.0f}%). "
                       "Consensus weight is NOT the primary oscillation driver.")

    print(f"  η_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, β={beta_g:.0e}: {verdict}")
    diag_lines += [
        f"  Group: η_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, β={beta_g:.0e}",
        f"  γ values   : {gammas}",
        f"  Amplitudes : {[f'{a:.4f}' if not np.isnan(a) else 'NaN' for a in amps_arr]}",
        f"  → {verdict}", "",
    ]


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC 4 — Oscillation amplitude vs β
# Amplitude falling as β grows suggests bilevel coupling: larger β slows the
# x-update relative to y and stabilizes training.
# ─────────────────────────────────────────────────────────────────
print("\n── Diagnostic 4: Oscillation amplitude vs β ──")
diag_lines += ["── Diagnostic 4: Oscillation amplitude vs β ──", ""]

groups_b = defaultdict(list)
for result in all_results:
    key = (result['eta_x'], result['eta_y'], result['gamma'])
    groups_b[key].append(result)

for key, group in groups_b.items():
    if len(group) < 2:
        continue
    eta_x_g, eta_y_g, gamma_g = key
    group_sorted = sorted(group, key=lambda r: r['beta'])
    betas = [r['beta'] for r in group_sorted]
    amps  = [oscillation_amplitude(
                 all_metrics[r['config_name']]['val_loss_history'])
             for r in group_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(betas, amps, marker='^', lw=2, color='seagreen')
    for b, amp in zip(betas, amps):
        if not np.isnan(amp):
            ax.annotate(f'{amp:.4f}', (b, amp),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=8, ha='center')
    ax.set_xlabel('β (regularization)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Val Loss Oscillation Amplitude', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Diag-4: Amplitude vs β\n'
        f'η_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, γ={gamma_g:.2f}',
        fontsize=10, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = (f'{run_prefix}_diag4_amp_vs_beta'
             f'_etax{eta_x_g:.0e}_etay{eta_y_g:.0e}_g{gamma_g:.2f}.png')
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.show()

    amps_arr = np.array(amps, dtype=float)
    if np.any(np.isnan(amps_arr)):
        verdict = "Could not assess — NaN amplitudes detected."
    else:
        mono_dec = all(amps_arr[i] >= amps_arr[i + 1]
                       for i in range(len(amps_arr) - 1))
        if mono_dec:
            verdict = ("BILEVEL COUPLING confirmed: amplitude strictly decreases "
                       "as β increases (larger β slows x-updates relative to y). "
                       "→ Use larger β to stabilise.")
        elif amps_arr[-1] < amps_arr[0]:
            verdict = ("Partial bilevel coupling effect: amplitude lower at large β. "
                       "Increasing β helps but is not the sole fix.")
        else:
            verdict = ("Amplitude does NOT decrease with β. "
                       "Bilevel coupling is NOT the primary driver.")

    print(f"  η_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, γ={gamma_g:.2f}: {verdict}")
    diag_lines += [
        f"  Group: η_x={eta_x_g:.0e}, η_y={eta_y_g:.0e}, γ={gamma_g:.2f}",
        f"  β values   : {betas}",
        f"  Amplitudes : {[f'{a:.4f}' if not np.isnan(a) else 'NaN' for a in amps_arr]}",
        f"  → {verdict}", "",
    ]


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC 5 — Gradient norm histories (if available)
# Correlate loss spikes with ‖∇_x f‖ and ‖∇_y g‖ to detect raw gradient noise.
# ─────────────────────────────────────────────────────────────────
print("\n── Diagnostic 5: Gradient norm vs Val Loss ──")
diag_lines += ["── Diagnostic 5: Gradient norm vs Val Loss ──", ""]

has_grad_norms = any(
    'grad_norm_x_history' in all_metrics[r['config_name']]
    and len(all_metrics[r['config_name']]['grad_norm_x_history']) > 0
    for r in all_results
)

if not has_grad_norms:
    msg = ("  ⚠ 'grad_norm_x_history' / 'grad_norm_y_history' not found in "
           "run_deviabo output.\n"
           "  Add gradient norm logging inside run_DeViABO.py at each log step:\n"
           "    norm_x = mean over nodes of ||∇_x f_i||\n"
           "    norm_y = mean over nodes of ||∇_y g_i||\n"
           "  Then return: 'grad_norm_x_history': [...], 'grad_norm_y_history': [...]")
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
        axes[0].set_ylabel('Val Loss', fontweight='bold')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        r_x, p_x = float('nan'), float('nan')
        if gnx is not None and len(gnx) > 0:
            r_x, p_x = spike_correlation(val_loss, gnx)
            r_x_str  = f"{r_x:.3f}" if not np.isnan(r_x) else "N/A"
            p_x_str  = f"{p_x:.3f}" if not np.isnan(p_x) else "N/A"
            axes[1].plot(iters, gnx, color='crimson', lw=2, label='‖∇_x f‖')
            axes[1].set_ylabel('‖∇_x f‖', fontweight='bold')
            axes[1].legend(fontsize=8, title=f'r={r_x_str}, p={p_x_str}')
            axes[1].grid(True, alpha=0.3)

        r_y, p_y = float('nan'), float('nan')
        if gny is not None and len(gny) > 0:
            r_y, p_y = spike_correlation(val_loss, gny)
            r_y_str  = f"{r_y:.3f}" if not np.isnan(r_y) else "N/A"
            p_y_str  = f"{p_y:.3f}" if not np.isnan(p_y) else "N/A"
            axes[2].plot(iters, gny, color='darkorange', lw=2, label='‖∇_y g‖')
            axes[2].set_ylabel('‖∇_y g‖', fontweight='bold')
            axes[2].legend(fontsize=8, title=f'r={r_y_str}, p={p_y_str}')
            axes[2].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Communication Round', fontweight='bold')
        fig.suptitle(f'Diag-5: Gradient Norms vs Val Loss | {cname}',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
        fname = f'{run_prefix}_diag5_gradnorm_{cname.replace("/","_")}.png'
        plt.savefig(fname, dpi=130, bbox_inches='tight')
        plt.show()

        r_x_str = f"{r_x:.3f}" if not np.isnan(r_x) else "N/A"
        r_y_str = f"{r_y:.3f}" if not np.isnan(r_y) else "N/A"
        if (not np.isnan(r_x) and r_x > 0.5) or (not np.isnan(r_y) and r_y > 0.5):
            verdict = ("STOCHASTIC GRADIENT NOISE / STEP SIZE confirmed: "
                       "gradient norm spikes correlate with val loss spikes. "
                       "→ Reduce η_x or η_y, add gradient clipping.")
        else:
            verdict = ("Gradient norms do NOT strongly correlate with val loss spikes. "
                       "Raw gradient variance is not the primary driver.")

        diag_lines += [
            f"  Config: {cname}",
            f"  r(val_loss, ‖∇_x‖) = {r_x_str}",
            f"  r(val_loss, ‖∇_y‖) = {r_y_str}",
            f"  → {verdict}", "",
        ]


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC 6 — Per-node val loss: inter-group vs intra-group
# Boundary nodes (neighbors in another data group) oscillating more than
# interior nodes signals non-IID divergence across the topology.
# ─────────────────────────────────────────────────────────────────
print("\n── Diagnostic 6: Per-node val loss decomposition ──")
diag_lines += ["── Diagnostic 6: Per-node val loss decomposition ──", ""]

has_node_val = any(
    'node_val_loss_history' in all_metrics[r['config_name']]
    and len(all_metrics[r['config_name']]['node_val_loss_history']) > 0
    for r in all_results
)

if not has_node_val:
    msg = ("  ⚠ 'node_val_loss_history' not found in run_deviabo output.\n"
           "  Add per-node val loss logging inside run_DeViABO.py at each log step:\n"
           "    per_node_losses = [eval_val_loss(models[i], val_loaders[i])"
           " for i in range(K_NODES)]\n"
           "  Then return: 'node_val_loss_history': [...]  "
           "  shape [num_log_steps][K_NODES]")
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
            has_inter = any(group_of[j] != group_of[i]
                            for j in neighborhoods[i] if j != i)
            (inter_nodes if has_inter else intra_nodes).append(i)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(K_NODES):
            style = '-' if i in inter_nodes else '--'
            label = (f'Node {i} (inter-group, g={group_of[i]})'
                     if i in inter_nodes
                     else f'Node {i} (intra-group, g={group_of[i]})')
            ax.plot(iters, node_val[:, i],
                    lw=2, linestyle=style,
                    color=node_colors[i], label=label)
        ax.set_xlabel('Communication Round', fontsize=11, fontweight='bold')
        ax.set_ylabel('Node Val Loss',        fontsize=11, fontweight='bold')
        ax.set_title(
            f'Diag-6: Per-node Val Loss | {cname}\n'
            f'Solid = inter-group nodes, Dashed = intra-group',
            fontsize=10, fontweight='bold'
        )
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f'{run_prefix}_diag6_pernode_val_{cname.replace("/","_")}.png'
        plt.savefig(fname, dpi=130, bbox_inches='tight')
        plt.show()

        amp_inter = float(np.nanmean([oscillation_amplitude(node_val[:, i])
                                      for i in inter_nodes])) if inter_nodes else 0.0
        amp_intra = float(np.nanmean([oscillation_amplitude(node_val[:, i])
                                      for i in intra_nodes])) if intra_nodes else 0.0

        if inter_nodes and amp_inter > 1.5 * amp_intra:
            verdict = (f"NON-IID DIVERGENCE confirmed: inter-group nodes oscillate "
                       f"{amp_inter / max(amp_intra, 1e-8):.1f}× more than intra-group. "
                       "→ Increase Dirichlet α or clip x-updates for boundary nodes.")
        elif inter_nodes and amp_inter > amp_intra:
            verdict = (f"Partial non-IID effect (×{amp_inter / max(amp_intra, 1e-8):.2f}). "
                       "Not the dominant cause.")
        else:
            verdict = ("Oscillation uniform across nodes — "
                       "non-IID topology is NOT the primary driver.")

        print(f"  {cname}: inter-amp={amp_inter:.4f}, intra-amp={amp_intra:.4f} → {verdict}")
        diag_lines += [
            f"  Config: {cname}",
            f"  Inter-group nodes  : {inter_nodes}",
            f"  Intra-group nodes  : {intra_nodes}",
            f"  Amp(inter-group)   : {amp_inter:.4f}",
            f"  Amp(intra-group)   : {amp_intra:.4f}",
            f"  → {verdict}", "",
        ]


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n── Oscillation Diagnostic Summary ──")
diag_lines += ["", "── Oscillation Diagnostic Summary ──", ""]

summary_verdicts = []
for result in all_results:
    cname     = result['config_name']
    metrics   = all_metrics[cname]
    val_loss  = metrics['val_loss_history']
    consensus = metrics['consensus_disagreement_history']

    amp    = oscillation_amplitude(val_loss)
    r_c, _ = spike_correlation(val_loss, consensus)

    causes = []
    if not np.isnan(r_c) and r_c > 0.5:
        causes.append('CONSENSUS TENSION')
    if result['eta_y'] == max(eta_y_values):
        causes.append('POSSIBLE η_y OVERSHOOTING (largest η_y in grid)')
    if not causes:
        causes.append('UNDETERMINED — add grad_norm and per-node logging')

    verdict_str = ' + '.join(causes)
    amp_str = f"{amp:.4f}" if not np.isnan(amp) else "N/A"
    r_c_str = f"{r_c:.3f}" if not np.isnan(r_c) else "N/A"
    summary_verdicts.append((cname, amp if not np.isnan(amp) else -1.0,
                              r_c, verdict_str))

    print(f"  {cname}")
    print(f"    Amplitude  : {amp_str}")
    print(f"    r(val,cons): {r_c_str}")
    print(f"    Verdict    : {verdict_str}\n")
    diag_lines += [
        f"  {cname}",
        f"    Oscillation amplitude : {amp_str}",
        f"    r(val_loss, consensus): {r_c_str}",
        f"    Verdict               : {verdict_str}",
        "",
    ]

summary_verdicts.sort(key=lambda x: x[1], reverse=True)
print("\n  Configs ranked by oscillation amplitude (most → least):")
diag_lines += ["  Configs ranked by oscillation amplitude (most → least):", ""]
for cname, amp, r_c, verdict in summary_verdicts:
    amp_str = f"{amp:.4f}" if amp >= 0 else "N/A"
    line    = f"    [{amp_str}] {cname}  —  {verdict}"
    print(line)
    diag_lines.append(line)

diag_filename = f'{run_prefix}_oscillation_diagnostics.txt'
with open(diag_filename, 'w') as f:
    f.write('\n'.join(diag_lines))
print(f"\n✓ Oscillation diagnostic report saved to '{diag_filename}'")


# ═════════════════════════════════════════════════════════════════
# SAVE EXPERIMENT SUMMARY TEXT FILE
# ═════════════════════════════════════════════════════════════════
sep  = "=" * 80
sep2 = "-" * 80
summary_lines = [
    sep,
    "  DeViABO EXPERIMENT SUMMARY — FashionMNIST",
    sep, "",
    "── EXPERIMENT SETUP ──────────────────────────────────────────────────────────",
    f"  Dataset          : FashionMNIST",
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
    f"  Model            : CNNFashion_Mnist",
    f"  Conv sizes       : {conv_size}",
    f"  Kernel size      : {kernel_size}",
    f"  Padding          : {padding}",
    f"  Hidden sizes     : {hidden_size}",
    f"  Num classes      : {number_classes}",
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
        f"  Rank {rank:>2d}{'  ★ BEST ★' if is_best else ''}  |  {cname}",
        sep2,
        f"  eta_x : {row['eta_x']:.0e}  |  eta_y : {row['eta_y']:.0e}"
        f"  |  gamma : {row['gamma']:.2f}  |  beta : {row['beta']:.0e}",
        "",
        f"  FINAL METRICS (at iter {iters[-1]}):",
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
    "", sep,
    "  BEST CONFIGURATION SUMMARY", sep,
    f"  Config name    : {best_config['config_name']}",
    f"  eta_x          : {best_config['eta_x']:.0e}",
    f"  eta_y          : {best_config['eta_y']:.0e}",
    f"  gamma          : {best_config['gamma']:.2f}",
    f"  beta           : {best_config['beta']:.0e}",
    f"  Test  Accuracy : {best_config['final_test_acc']:.2f}%",
    f"  Test  Loss     : {best_config['final_test_loss']:.4f}",
    f"  Val   Loss     : {best_config['final_val_loss']:.4f}",
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