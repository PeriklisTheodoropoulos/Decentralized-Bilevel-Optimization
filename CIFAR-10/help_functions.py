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
# ─────────────────────────────────────────────────────────────────
# split data across nodes
# ─────────────────────────────────────────────────────────────────
def split_node_data_non_iid(
    dataset_train, dataset_test, K, val_ratio, seed=42,
    mode='class_pair', alpha=5.0, num_groups=None,
    alpha_min=0.01,
    alpha_max=1.0,
    nodes_per_cluster=2,
):
    assert mode in ('class_pair', 'class_multi', 'clustered', 'class_group',
                    'dirichlet', 'clustered_dirichlet', 'spatial', 'cobo_split'), (
        "mode must be one of: 'class_pair', 'class_multi', 'clustered', "
        "'class_group', 'dirichlet', 'clustered_dirichlet', 'spatial', 'cobo_split'."
    )

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ─────────────────────────────────────────────────────────────
    # Normalize any source format to [N, C, H, W] tensor
    # ─────────────────────────────────────────────────────────────
    def _to_chw_tensor(data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.array(data))

        if data.ndim == 4:
            # Already channels-first [N, C, H, W] — C is small (1 or 3)
            if data.shape[1] in (1, 3):
                return data                              # already correct
            # Channels-last [N, H, W, C] — C is last and small
            if data.shape[-1] in (1, 3):
                return data.permute(0, 3, 1, 2).contiguous()
        elif data.ndim == 3:
            # FashionMNIST: [N, H, W] grayscale → [N, 1, H, W]
            # data = data.unsqueeze(1)
            data = data

        return data

    def _to_label_tensor(targets):
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(np.array(targets))
        return targets.long()

    # ─────────────────────────────────────────────────────────────
    # Extract data — always [N, C, H, W] from this point forward
    # ─────────────────────────────────────────────────────────────
    data_train  = _to_chw_tensor(dataset_train.data)    # [N, C, H, W]
    label_train = _to_label_tensor(dataset_train.targets)

    label_sorted, indices = torch.sort(label_train.clone().detach())
    data_sorted = data_train[indices]                   # [N, C, H, W]

    classes   = torch.unique(label_sorted)
    n_classes = len(classes)

    print(f"\n[Dataset Info] Total train samples: {len(data_train)}, "
          f"Classes: {n_classes}, Test samples: {len(dataset_test)}, "
          f"Shape: {tuple(data_train.shape[1:])}")

    class_indices = {
        c.item(): (label_sorted == c).nonzero(as_tuple=True)[0].numpy()
        for c in classes
    }

    # Correct empty-node shape: (C, H, W)
    _sample_shape = tuple(data_train.shape[1:])

    # ── Test data: converted once, returned by all modes ─────────
    test_data = (
        _to_chw_tensor(dataset_test.data),
        _to_label_tensor(dataset_test.targets),
    )

    # ─────────────────────────────────────────────────────────────
    # Helper: train/val split per node
    # ─────────────────────────────────────────────────────────────
    def _make_data_chunks(node_indices_list, node_labels_override=None):
        data_chunks = {}

        for node_id, idx_array in enumerate(node_indices_list):
            node_data = data_sorted[idx_array]          # [n, C, H, W]

            if node_labels_override is not None and node_id in node_labels_override:
                node_labels = node_labels_override[node_id]
            else:
                node_labels = label_sorted[idx_array]

            # ── Empty node ───────────────────────────────────────
            if len(idx_array) == 0:
                data_chunks[node_id] = {
                    'train': (torch.empty(0, *_sample_shape),
                              torch.empty(0, dtype=torch.long)),
                    'val':   (torch.empty(0, *_sample_shape),
                              torch.empty(0, dtype=torch.long)),
                }
                print(f"  Node {node_id}: no data, skipped.")
                continue

            # ── No validation split ──────────────────────────────
            if val_ratio == 0.0:
                data_chunks[node_id] = {
                    'train': (node_data, node_labels),
                    'val':   (torch.empty(0, *_sample_shape),
                              torch.empty(0, dtype=torch.long)),
                }
                dist = dict(zip(*np.unique(node_labels.numpy(), return_counts=True)))
                print(f"  Node {node_id}: train={len(node_data)}, val=0, dist={dist}")
                continue

            # ── Normal train / val split ─────────────────────────
            node_labels_np = node_labels.numpy()
            _, counts      = np.unique(node_labels_np, return_counts=True)
            can_stratify   = np.all(counts >= 2)

            try:
                idx_tr, idx_vl = train_test_split(
                    np.arange(len(idx_array)),
                    test_size=val_ratio,
                    random_state=seed,
                    stratify=node_labels_np if can_stratify else None,
                )
            except ValueError:
                idx_tr, idx_vl = train_test_split(
                    np.arange(len(idx_array)),
                    test_size=val_ratio,
                    random_state=seed,
                )

            data_chunks[node_id] = {
                'train': (node_data[idx_tr], node_labels[idx_tr]),
                'val':   (node_data[idx_vl], node_labels[idx_vl]),
            }
            dist = dict(zip(*np.unique(node_labels[idx_tr].numpy(), return_counts=True)))
            print(f"  Node {node_id}: train={len(idx_tr)}, val={len(idx_vl)}, dist={dist}")

        return data_chunks

    # ─────────────────────────────────────────────────────────────
    # Mode 1: class_pair
    # ─────────────────────────────────────────────────────────────
    if mode == 'class_pair':
        assert K % 2 == 0, "K must be even for mode='class_pair'."
        assert n_classes == K // 2, (
            f"mode='class_pair' needs K = 2 × n_classes = {2*n_classes}, got K={K}."
        )
        print(f"[split | class_pair]  K={K}, n_classes={n_classes}, nodes_per_class=2")

        node_indices = []
        for c_idx, c in enumerate(classes):
            idx = class_indices[c.item()]
            np.random.default_rng(seed + c_idx).shuffle(idx)
            mid = len(idx) // 2
            node_indices.append(idx[:mid])
            node_indices.append(idx[mid:])

        return _make_data_chunks(node_indices), test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 2: class_multi
    # ─────────────────────────────────────────────────────────────
    if mode == 'class_multi':
        assert K % n_classes == 0, (
            f"K={K} must be divisible by n_classes={n_classes}."
        )
        nodes_per_class = K // n_classes
        print(f"[split | class_multi]  K={K}, n_classes={n_classes}, "
              f"nodes_per_class={nodes_per_class}")

        node_indices = []
        for c_idx, c in enumerate(classes):
            idx = class_indices[c.item()]
            np.random.default_rng(seed + c_idx).shuffle(idx)
            for chunk in np.array_split(idx, nodes_per_class):
                node_indices.append(chunk)

        return _make_data_chunks(node_indices), test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 3: clustered
    # ─────────────────────────────────────────────────────────────
    if mode == 'clustered':
        assert K >= n_classes, (
            f"K={K} must be >= n_classes={n_classes} for mode='clustered'."
        )
        node_ids = np.arange(K)
        clusters = np.array_split(node_ids, n_classes)
        print(f"[split | clustered]  K={K}, n_classes={n_classes}, "
              f"cluster sizes={[len(c) for c in clusters]}")

        node_indices = [None] * K
        for c_idx, c in enumerate(classes):
            cluster = clusters[c_idx]
            idx     = class_indices[c.item()]
            np.random.default_rng(seed + c_idx).shuffle(idx)
            for node_id, chunk in zip(cluster, np.array_split(idx, len(cluster))):
                node_indices[node_id] = chunk

        return _make_data_chunks(node_indices), test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 4: class_group
    # ─────────────────────────────────────────────────────────────
    if mode == 'class_group':
        assert K >= 2,        "K must be >= 2 for mode='class_group'."
        assert K < n_classes, (
            f"mode='class_group' is for K < n_classes. "
            f"Got K={K}, n_classes={n_classes}."
        )
        G = num_groups if num_groups is not None else max(1, K // 2)
        assert 1 <= G <= K,    f"num_groups={G} must be in [1, K={K}]."
        assert G <= n_classes, f"num_groups={G} must be <= n_classes={n_classes}."

        node_groups  = np.array_split(np.arange(K), G)
        class_groups = np.array_split(np.arange(n_classes), G)

        print(f"[split | class_group]  K={K}, n_classes={n_classes}, G={G}")
        for g_idx in range(G):
            print(f"  Group {g_idx}: nodes {node_groups[g_idx].tolist()} "
                  f"→ classes {class_groups[g_idx].tolist()}")

        node_indices = [None] * K
        for g_idx in range(G):
            pooled_idx = np.concatenate([
                class_indices[int(classes[c_id].item())]
                for c_id in class_groups[g_idx]
            ])
            np.random.default_rng(seed + g_idx).shuffle(pooled_idx)
            for node_id, chunk in zip(node_groups[g_idx],
                                      np.array_split(pooled_idx,
                                                     len(node_groups[g_idx]))):
                node_indices[node_id] = chunk

        return _make_data_chunks(node_indices), test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 5: dirichlet
    # ─────────────────────────────────────────────────────────────
    if mode == 'dirichlet':
        print(f"[split | dirichlet]  K={K}, n_classes={n_classes}, α={alpha}")

        proportions  = np.random.dirichlet(alpha=np.repeat(alpha, K), size=len(classes))
        node_indices = [[] for _ in range(K)]

        for class_idx, c in enumerate(classes):
            c_indices        = class_indices[c.item()]
            n_class_samples  = len(c_indices)
            class_prop       = proportions[class_idx]
            samples_per_node = (class_prop * n_class_samples).astype(int)
            diff = n_class_samples - np.sum(samples_per_node)
            for i in range(diff):
                samples_per_node[i % K] += 1
            start = 0
            for node_id in range(K):
                count = samples_per_node[node_id]
                if count > 0:
                    node_indices[node_id].extend(c_indices[start:start + count])
                    start += count

        node_indices = [np.array(sorted(idx)) for idx in node_indices]
        return _make_data_chunks(node_indices), test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 6: clustered_dirichlet
    # ─────────────────────────────────────────────────────────────
    if mode == 'clustered_dirichlet':
        G = num_groups if num_groups is not None else max(1, K // 2)
        assert 1 <= G <= K,    f"num_groups={G} must be in [1, K={K}]."
        assert G <= n_classes, f"num_groups={G} must be <= n_classes={n_classes}."

        node_groups  = np.array_split(np.arange(K), G)
        class_groups = np.array_split(np.arange(n_classes), G)

        print(f"[split | clustered_dirichlet]  K={K}, n_classes={n_classes}, "
              f"G={G}, α={alpha}")
        for g in range(G):
            print(f"  Group {g}: nodes {node_groups[g].tolist()} "
                  f"→ classes {class_groups[g].tolist()}")

        node_indices = [[] for _ in range(K)]

        for g_idx in range(G):
            node_group  = node_groups[g_idx]
            class_group = class_groups[g_idx]
            n_g         = len(node_group)
            rng         = np.random.default_rng(seed + g_idx)
            proportions = rng.dirichlet(alpha=np.repeat(alpha, n_g),
                                        size=len(class_group))

            for local_c_idx, global_c_id in enumerate(class_group):
                c         = classes[global_c_id].item()
                c_indices = class_indices[c].copy()
                rng.shuffle(c_indices)
                n_samples        = len(c_indices)
                samples_per_node = (proportions[local_c_idx] * n_samples).astype(int)
                diff = n_samples - samples_per_node.sum()
                for i in range(diff):
                    samples_per_node[i % n_g] += 1
                start = 0
                for local_node_idx, node_id in enumerate(node_group):
                    count = samples_per_node[local_node_idx]
                    node_indices[node_id].extend(c_indices[start:start + count])
                    start += count

        node_indices = [np.array(sorted(idx)) for idx in node_indices]
        return _make_data_chunks(node_indices), test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 7: spatial
    # ─────────────────────────────────────────────────────────────
    if mode == 'spatial':
        G = num_groups if num_groups is not None else max(2, K // 2)
        assert 2 <= G <= K,    f"num_groups={G} must be in [2, K={K}]."
        assert G <= n_classes, f"num_groups={G} must be <= n_classes={n_classes}."

        node_groups  = np.array_split(np.arange(K), G)
        class_groups = np.array_split(np.arange(n_classes), G)

        decay_lambda = 2.0
        cross_alpha  = np.zeros((G, G))
        for gi in range(G):
            for gj in range(G):
                dist = abs(gi - gj)
                cross_alpha[gi, gj] = (alpha if dist == 0 else
                    alpha_min + (alpha_max - alpha_min) * np.exp(-decay_lambda * dist))

        print(f"[split | spatial]  K={K}, n_classes={n_classes}, G={G}")
        print(f"  alpha={alpha:.3f}, alpha_min={alpha_min:.4f}, "
              f"alpha_max={alpha_max:.3f}")
        for gi in range(G):
            row = "  ".join(f"{cross_alpha[gi, gj]:.3f}" for gj in range(G))
            print(f"  Cluster {gi} (nodes {node_groups[gi].tolist()}, "
                  f"classes {class_groups[gi].tolist()}): [{row}]")

        rng        = np.random.default_rng(seed)
        node_props = np.zeros((K, n_classes))

        for gi, node_group in enumerate(node_groups):
            conc_vec = np.zeros(n_classes)
            for gj, cls_group in enumerate(class_groups):
                conc_vec[cls_group] = cross_alpha[gi, gj]
            for node_id in node_group:
                node_rng = np.random.default_rng(seed + int(node_id))
                node_props[node_id] = node_rng.dirichlet(conc_vec)

        node_indices = [[] for _ in range(K)]
        for class_idx, c in enumerate(classes):
            c_indices = class_indices[c.item()].copy()
            rng.shuffle(c_indices)
            n_samples = len(c_indices)
            col       = node_props[:, class_idx]
            col       = col / col.sum()
            samples_per_node = (col * n_samples).astype(int)
            frac      = col * n_samples - samples_per_node
            remainder = n_samples - samples_per_node.sum()
            samples_per_node[np.argsort(-frac)[:remainder]] += 1
            start = 0
            for k in range(K):
                count = samples_per_node[k]
                if count > 0:
                    node_indices[k].extend(c_indices[start:start + count])
                start += count

        node_indices = [np.array(sorted(idx)) for idx in node_indices]
        return _make_data_chunks(node_indices), test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 8: cobo_split
    # ─────────────────────────────────────────────────────────────
    if mode == 'cobo_split':
        assert K % nodes_per_cluster == 0, (
            f"K={K} must be divisible by nodes_per_cluster={nodes_per_cluster}."
        )
        G = K // nodes_per_cluster
        print(f"[split | cobo_split]  K={K}, n_classes={n_classes}, "
              f"G={G} clusters × {nodes_per_cluster} nodes each")

        node_groups   = np.array_split(np.arange(K), G)
        all_indices   = np.arange(len(data_sorted))
        rng_pool      = np.random.default_rng(seed)
        rng_pool.shuffle(all_indices)
        cluster_pools = np.array_split(all_indices, G)

        node_indices = [None] * K
        for g_idx, node_group in enumerate(node_groups):
            pool = cluster_pools[g_idx].copy()
            np.random.default_rng(seed + g_idx).shuffle(pool)
            for node_id, chunk in zip(node_group,
                                      np.array_split(pool, nodes_per_cluster)):
                node_indices[node_id] = np.sort(chunk)

        class_list   = classes.numpy()
        permutations = {}
        for g_idx in range(G):
            perm_rng = np.random.default_rng(seed + 1000 + g_idx)
            perm     = class_list.copy()
            perm_rng.shuffle(perm)
            permutations[g_idx] = perm
            print(f"  Cluster {g_idx} (nodes {node_groups[g_idx].tolist()}): "
                  f"π_{g_idx} = {class_list.tolist()} → {perm.tolist()}")

        node_labels_override = {}
        for g_idx, node_group in enumerate(node_groups):
            perm = permutations[g_idx]
            for node_id in node_group:
                orig_labels = label_sorted[node_indices[node_id]]
                node_labels_override[node_id] = torch.tensor(
                    perm[orig_labels.numpy()], dtype=torch.long
                )

        for node_id in range(K):
            perm_labs = node_labels_override[node_id]
            dist      = dict(zip(*np.unique(perm_labs.numpy(), return_counts=True)))
            g_idx     = next(gi for gi, ng in enumerate(node_groups)
                             if node_id in ng)
            print(f"  Node {node_id} (cluster {g_idx}): "
                  f"n={len(node_indices[node_id])}, permuted_dist={dist}")

        return _make_data_chunks(node_indices,
                                 node_labels_override=node_labels_override), test_data
    
# ─────────────────────────────────────────────────────────────────
# plot_distribution
# ─────────────────────────────────────────────────────────────────
def plot_distribution(data_chunks, num_classes=10, title='Data Distribution per Node'):
    """
    Plot the class distribution across nodes.
    
    Args:
        data_chunks: dict mapping node_id -> {'train': (data, labels), 'val': (data, labels)}
        num_classes: number of classes in the dataset (default: 10 for FashionMNIST)
        title: plot title
    """
    num_nodes = len(data_chunks)
    counts_train = []
    counts_val = []
    counts_total = []
    
    for node_id in range(num_nodes):
        # Get train labels
        train_labels = data_chunks[node_id]['train'][1]
        if isinstance(train_labels, np.ndarray):
            train_labels = torch.tensor(train_labels)
        bc_train = torch.bincount(train_labels, minlength=num_classes)
        counts_train.append(bc_train.numpy())
        
        # Get val labels
        val_labels = data_chunks[node_id]['val'][1]
        if isinstance(val_labels, np.ndarray):
            val_labels = torch.tensor(val_labels)
        bc_val = torch.bincount(val_labels, minlength=num_classes)
        counts_val.append(bc_val.numpy())
        
        # Total (train + val)
        bc_total = bc_train + bc_val
        counts_total.append(bc_total.numpy())
    
    counts_train = np.array(counts_train)
    counts_val = np.array(counts_val)
    counts_total = np.array(counts_total)
    nodes = np.arange(num_nodes)
    
    # ─────────────────────────────────────────────────────────────────
    # Plot 1: Total distribution (train + val)
    # ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Color palette for classes
    colors = plt.cm.tab10(np.arange(num_classes))
    
    # Subplot 1: Total distribution
    ax = axes[0]
    bottom = np.zeros(num_nodes)
    for c in range(num_classes):
        ax.bar(nodes, counts_total[:, c], bottom=bottom, 
               label=f'Class {c}', color=colors[c], alpha=0.8)
        bottom += counts_total[:, c]
    
    ax.set_xlabel('Node ID', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Samples', fontsize=11, fontweight='bold')
    ax.set_title('Total Distribution (Train + Val)', fontsize=12, fontweight='bold')
    ax.set_xticks(nodes)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Train distribution only
    ax = axes[1]
    bottom = np.zeros(num_nodes)
    for c in range(num_classes):
        ax.bar(nodes, counts_train[:, c], bottom=bottom, 
               label=f'Class {c}', color=colors[c], alpha=0.8)
        bottom += counts_train[:, c]
    
    ax.set_xlabel('Node ID', fontsize=11, fontweight='bold')
    ax.set_ylabel('Train Samples', fontsize=11, fontweight='bold')
    ax.set_title('Train Distribution Only', fontsize=12, fontweight='bold')
    ax.set_xticks(nodes)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Validation distribution only
    ax = axes[2]
    bottom = np.zeros(num_nodes)
    for c in range(num_classes):
        ax.bar(nodes, counts_val[:, c], bottom=bottom, 
               label=f'Class {c}', color=colors[c], alpha=0.8)
        bottom += counts_val[:, c]
    
    ax.set_xlabel('Node ID', fontsize=11, fontweight='bold')
    ax.set_ylabel('Val Samples', fontsize=11, fontweight='bold')
    ax.set_title('Validation Distribution Only', fontsize=12, fontweight='bold')
    ax.set_xticks(nodes)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # ─────────────────────────────────────────────────────────────────
    # Print statistics
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("DATA DISTRIBUTION SUMMARY")
    print("="*70)
    print(f"{'Node':<6} {'Train':<10} {'Val':<10} {'Total':<10} {'Dominant Class':<20}")
    print("-"*70)
    
    for node_id in range(num_nodes):
        train_count = counts_train[node_id].sum()
        val_count = counts_val[node_id].sum()
        total_count = counts_total[node_id].sum()
        dominant_class = np.argmax(counts_total[node_id])
        dominant_pct = (counts_total[node_id][dominant_class] / total_count * 100) if total_count > 0 else 0
        
        print(f"{node_id:<6} {train_count:<10} {val_count:<10} {total_count:<10} "
              f"Class {dominant_class} ({dominant_pct:.1f}%)")
    
    print("-"*70)
    print(f"{'TOTAL':<6} {counts_train.sum():<10.0f} {counts_val.sum():<10.0f} "
          f"{counts_total.sum():<10.0f}")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────
    # Additional plot: Heatmap of class distribution
    # ─────────────────────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(counts_total.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(num_nodes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(nodes)
    ax.set_yticklabels([f'Class {c}' for c in range(num_classes)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Samples', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(num_nodes):
        for j in range(num_classes):
            count = counts_total[i, j]
            if count > 0:
                text_color = 'white' if count > counts_total.max() / 2 else 'black'
                text = ax.text(i, j, int(count), ha='center', va='center', 
                              color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Node ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} - Heatmap', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────
# Generate Grid Graph
# ─────────────────────────────────────────────────────────────────
def generate_3d_grid_by_node_count(k_nodes):
    best_dims = (1, 1, k_nodes)
    best_diff = float('inf')
    
    # Calculate the 3 closest integer factors whose product equals k_nodes
    for i in range(1, k_nodes + 1):
        if k_nodes % i == 0:
            remaining_1 = k_nodes // i
            for j in range(1, remaining_1 + 1):
                if remaining_1 % j == 0:
                    l = remaining_1 // j
                    dims = sorted([i, j, l])
                    # Measure how close the dimensions are to a perfect cube
                    diff = (dims[2] - dims[0]) + (dims[1] - dims[0])
                    if diff < best_diff:
                        best_diff = diff
                        best_dims = tuple(dims)
                        
    print(f"Calculated 3D Grid Dimensions: {best_dims}")
    
    # 1. Generate the raw 3D Grid Graph
    G = nx.grid_graph(dim=best_dims)
    
    # 2. Convert tuple names like (0, 0, 0) into integers starting from 0
    G_flattened = nx.convert_node_labels_to_integers(G, first_label=0)
    
    return G_flattened
# ─────────────────────────────────────────────────────────────────
# Generate Graph
# ─────────────────────────────────────────────────────────────────
def generate_graph(graph_type, K, seed):
    """
    Generates a graph of a specified type with K nodes and adds self-loops.

    Args:
        graph_type (str): Type of graph to generate ('ring', 'fully connected', 'line', 'Barabasi').
        K (int): Number of nodes in the graph.

    Returns:
        networkx.Graph: The generated graph.
    """
    if K <= 0:
        raise ValueError("Number of nodes (K) must be positive.")

    if graph_type == 'ring':
        G = nx.cycle_graph(K)
    elif graph_type == 'caveman':
        G = nx.connected_caveman_graph(3, 7)
        #  G = nx.connected_caveman_graph(2, 7)
        # G = nx.connected_caveman_graph(3, 4)
        # G = nx.connected_caveman_graph(3, 5)
        # G = nx.connected_caveman_graph(2, 5)

    elif graph_type == 'tree':
        G = nx.full_rary_tree(r=3, n=K)
    elif graph_type == 'fully connected':
        G = nx.complete_graph(K)
    elif graph_type == 'line':
        G = nx.path_graph(K)
    elif graph_type == 'star':
        G = nx.star_graph(K)
    elif graph_type == 'Grid':
        G = generate_3d_grid_by_node_count(K)
    elif graph_type == 'Barabasi':
        if K < 2:
            # Barabasi-Albert requires at least 2 nodes to connect
            G = nx.complete_graph(K) # Handle small K separately or raise error
        else:
            # m is the number of edges to attach from a new node to existing nodes
            # For simplicity, we'll use m=1, can be adjusted
            G = nx.barabasi_albert_graph(K, m=2, seed= seed)
    elif graph_type == 'SBM':
        sizes = [5, 5, 5]
        # Define connection probabilities for the linear chain setup
        probs = [
            [0.6, 0.1, 0.0],  # Cluster 1 internal, C1<->C2, C1<->C3 (Strictly 0)
            [0.1, 0.6, 0.1],  # C2<->C1, Cluster 2 internal, C2<->C3
            [0.0, 0.1, 0.6],  # C3<->C1 (Strictly 0), C3<->C2, Cluster 3 internal
        ]

        
        # Loop internally until a completely connected graph draw is found
        current_seed = seed
        while True:
            G = nx.stochastic_block_model(sizes, probs, seed=current_seed)
            if nx.is_connected(G):
                break
            current_seed += 1  # Shift seed forward if the graph is disconnected
            
    else:
        # Added 'SBM' and 'caveman' to the allowed list error message
        raise ValueError("Invalid graph_type. Choose from 'ring', 'fully connected', 'line', 'Barabasi', 'caveman', 'SBM'.")


    # Add self-loops to all nodes
    for node in G.nodes():
        G.add_edge(node, node)

    return G


# ─────────────────────────────────────────────────────────────────
# Helper: stochastic gradient ∇_{y_i} f_i^{tr}(y_i^t ; ζ_i^t)
# ─────────────────────────────────────────────────────────────────
# def compute_stochastic_grad(model, loader, criterion, device):
    # model.zero_grad()
    # X_batch, y_batch = next(iter(loader))
    # # X_batch = X_batch.unsqueeze(1).to(device)   # [B,28,28] → [B,1,28,28]
    # X_batch = X_batch.to(device)   # [B,28,28] → [B,1,28,28]
    # y_batch = y_batch.to(device)
    # with torch.enable_grad():
    #     out = model(X_batch)
    #     loss = criterion(out, y_batch)
    #     loss.backward()
    # return {
    #     name: param.grad.detach().clone()
    #     for name, param in model.named_parameters()
    #     if param.grad is not None
    # }
# def compute_stochastic_grad(model, loader, criterion, device):
#     model.zero_grad()
#     X_batch, y_batch = next(iter(loader))
#     X_batch = augment_batch(X_batch,device)        # ← augment on GPU
#     with torch.enable_grad():
#         loss = criterion(model(X_batch), y_batch)
#         loss.backward()
#     return {
#         name: param.grad.detach().clone()
#         for name, param in model.named_parameters()
#         if param.grad is not None
#     }
def compute_stochastic_grad(model, batch, criterion, device):
    X_batch, y_batch = batch                        # already on device
    X_batch = augment_batch(X_batch, device)
    model.zero_grad()
    with torch.enable_grad():
        criterion(model(X_batch), y_batch).backward()
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
# def compute_stochastic_grad(model, loader, criterion, device):
#     model.zero_grad()
#     X_batch, y_batch = next(iter(loader))
#     # X_batch = X_batch.unsqueeze(1).to(device)   # [B,28,28] → [B,1,28,28]
#     X_batch = X_batch.to(device)   # [B,28,28] → [B,1,28,28]
#     y_batch = y_batch.to(device)
#     with torch.enable_grad():
#         out = model(X_batch)
#         loss = criterion(out, y_batch)
#         loss.backward()
#     return {
#         name: param.grad.detach().clone()
#         for name, param in model.named_parameters()
#         if param.grad is not None
#     }

# ─────────────────────────────────────────────────────────────────
# Helper: midpoint model z_{i,j} = (y_i^t + y_j^t) / 2
#         and its gradients w.r.t. f_i^{tr} and f_j^{tr}
# ─────────────────────────────────────────────────────────────────
# def compute_midpoint_grads(model_i, model_j, loader_i, loader_j,
#                            criterion, model_factory, device):
    # z_ij = model_factory().to(device)
# def compute_midpoint_grads(model_i, model_j, loader_i, loader_j,
#                            criterion, z_ij, device):
#     with torch.no_grad():
#         for (_, p_i), (_, p_j), (_, p_z) in zip(
#             model_i.named_parameters(),
#             model_j.named_parameters(),
#             z_ij.named_parameters(),
#         ):
#             p_z.copy_((p_i + p_j) / 2.0)

#     # ∇_{y} f_i^{tr}(z_{i,j} ; ζ_i^t)
#     with torch.enable_grad():
#         z_ij.zero_grad()
#         X_i, y_i = next(iter(loader_i))
#         # X_i = X_i.unsqueeze(1).to(device)       # [B,28,28] → [B,1,28,28]
#         # X_i = X_i.to(device)       # [B,28,28] → [B,1,28,28]
#         # y_i = y_i.to(device)
#         X_i = augment_batch(X_i,device)           # ← augment on GPU
#         criterion(z_ij(X_i), y_i).backward()
#         grad_i_mid = [p.grad.detach().clone() for p in z_ij.parameters()]

#     # ∇_{y} f_j^{tr}(z_{i,j} ; ζ_j^t)
#     with torch.enable_grad():
#         z_ij.zero_grad()
#         X_j, y_j = next(iter(loader_j))
#         # X_j = X_j.unsqueeze(1).to(device)       # [B,28,28] → [B,1,28,28]
#         # X_j = X_j.to(device)       # [B,28,28] → [B,1,28,28]
#         # y_j = y_j.to(device)
#         X_i = augment_batch(X_i,device)           # ← augment on GPU
#         criterion(z_ij(X_j), y_j).backward()
#         grad_j_mid = [p.grad.detach().clone() for p in z_ij.parameters()]

#     return grad_i_mid, grad_j_mid

def compute_midpoint_grads(model_i, model_j, batch_i, batch_j,
                           criterion, z_ij, device):
    with torch.no_grad():
        for (_, p_i), (_, p_j), (_, p_z) in zip(
            model_i.named_parameters(),
            model_j.named_parameters(),
            z_ij.named_parameters(),
        ):
            p_z.copy_((p_i + p_j) / 2.0)

    X_i, y_i = batch_i                              # already on device
    X_j, y_j = batch_j                              # already on device
    X_i = augment_batch(X_i, device)
    X_j = augment_batch(X_j, device)               # ✅ was X_i before

    with torch.enable_grad():
        z_ij.zero_grad()
        criterion(z_ij(X_i), y_i).backward()
        grad_i_mid = [p.grad.detach().clone() for p in z_ij.parameters()]

    with torch.enable_grad():
        z_ij.zero_grad()
        criterion(z_ij(X_j), y_j).backward()
        grad_j_mid = [p.grad.detach().clone() for p in z_ij.parameters()]

    return grad_i_mid, grad_j_mid




# def compute_midpoint_grads(model_i, model_j, data_i, data_j,
#                            criterion, z_ij, device):

#     # ── Accept loader or pre-sampled (X, y) tuple ────────────────
#     if isinstance(data_i, (list, tuple)):
#         X_i, y_i = data_i
#         X_j, y_j = data_j
#     else:
#         X_i, y_i = next(iter(data_i))
#         X_j, y_j = next(iter(data_j))
#         X_i, y_i = X_i.to(device), y_i.to(device)
#         X_j, y_j = X_j.to(device), y_j.to(device)

#     # ── z_{i,j} = (y_i + y_j) / 2  — vectorized, single GPU op ──
#     with torch.no_grad():
#         vec_i = torch.nn.utils.parameters_to_vector(model_i.parameters())
#         vec_j = torch.nn.utils.parameters_to_vector(model_j.parameters())
#         torch.nn.utils.vector_to_parameters((vec_i + vec_j) / 2.0, z_ij.parameters())

#     # ── ∇f_i(z_{i,j} ; ζ_i^t) ────────────────────────────────────
#     with torch.enable_grad():
#         z_ij.zero_grad()
#         criterion(z_ij(X_i), y_i).backward()
#         grad_i_vec = torch.nn.utils.parameters_to_vector(
#             [p.grad for p in z_ij.parameters()]
#         ).detach().clone()

#     # ── ∇f_j(z_{i,j} ; ζ_j^t) ────────────────────────────────────
#     with torch.enable_grad():
#         z_ij.zero_grad()
#         criterion(z_ij(X_j), y_j).backward()
#         grad_j_vec = torch.nn.utils.parameters_to_vector(
#             [p.grad for p in z_ij.parameters()]
#         ).detach().clone()

#     return grad_i_vec, grad_j_vec


# ─────────────────────────────────────────────────────────────────
# Helper: consensus disagreement
#   (1/K) Σ_i ||y_i - ȳ||_2^2
# ─────────────────────────────────────────────────────────────────
def compute_consensus_disagreement(models, K):
    with torch.no_grad():
        mean_params = [
            torch.stack([list(models[i].parameters())[idx]
                         for i in range(K)]).mean(dim=0)
            for idx in range(len(list(models[0].parameters())))
        ]
        total = sum(
            (p - p_mean).pow(2).sum().item()
            for i in range(K)
            for p, p_mean in zip(models[i].parameters(), mean_params)
        )
    return total / K

# def compute_consensus_disagreement(vecs, K):
#     with torch.no_grad():
#         mean_vec = vecs.mean(dim=0)
#         total    = ((vecs - mean_vec) ** 2).sum().item()
#     return total / K

# # ─────────────────────────────────────────────────────────────────
# # Helper: pairwise disagreement
# #   Σ_{i<j} x_{i,j} ||y_i - y_j||_2^2
# # ─────────────────────────────────────────────────────────────────
def compute_pairwise_disagreement(models, X, K):
    with torch.no_grad():
        total = sum(
            X[i, j].item() * sum(
                (p_i - p_j).pow(2).sum().item()
                for p_i, p_j in zip(models[i].parameters(),
                                    models[j].parameters())
            )
            for i in range(K)
            for j in range(i + 1, K)
            if X[i, j].item() > 0
        )
    return total


# def compute_pairwise_disagreement(vecs, X, K):
#     with torch.no_grad():
#         total = 0.0
#         for i in range(K):
#             for j in range(i + 1, K):
#                 if X[i, j] > 0:
#                     diff  = vecs[i] - vecs[j]
#                     total += (X[i, j] * diff.pow(2).sum()).item()
#     return total



# def augment_batch(x: torch.Tensor, device) -> torch.Tensor:
#     """RandomCrop(32, padding=4) + RandomHorizontalFlip — applied fresh every batch."""
#     N = x.shape[0]
#     x = x.clone()
#     mask = torch.rand(N, device=device) < 0.5
#     x[mask] = x[mask].flip(-1)
#     x = F.pad(x, (4, 4, 4, 4), mode='reflect')
#     tops  = torch.randint(0, 8, (N,), device=device)
#     lefts = torch.randint(0, 8, (N,), device=device)
#     return torch.stack([x[n, :, tops[n]:tops[n]+32, lefts[n]:lefts[n]+32] for n in range(N)])
## Vectorized augment_batch function
def augment_batch(x: torch.Tensor, device) -> torch.Tensor:
    N, C = x.shape[0], x.shape[1]
    x = x.clone()
    mask = torch.rand(N, device=device) < 0.5
    x[mask] = x[mask].flip(-1)
    x = F.pad(x, (4, 4, 4, 4), mode='reflect')

    tops  = torch.randint(0, 8, (N,), device=device)
    lefts = torch.randint(0, 8, (N,), device=device)

    n_idx = torch.arange(N, device=device).view(N, 1, 1, 1)
    c_idx = torch.arange(C, device=device).view(1, C, 1, 1)
    i_idx = (tops.view(N, 1, 1, 1)  + torch.arange(32, device=device).view(1, 1, 32, 1))
    j_idx = (lefts.view(N, 1, 1, 1) + torch.arange(32, device=device).view(1, 1, 1, 32))

    return x[n_idx, c_idx, i_idx, j_idx]
