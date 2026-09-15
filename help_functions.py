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
# def split_node_data_non_iid(dataset_train, dataset_test, K, val_ratio, seed=42,
#                              mode='class_pair', alpha=0.5, num_groups=None):
def split_node_data_non_iid(
    dataset_train, dataset_test, K, val_ratio, seed=42,
    mode='class_pair', alpha=5.0, num_groups=None,
    alpha_min=0.01,    # cross-cluster α for far clusters  (high heterogeneity)
    alpha_max=1.0,     # cross-cluster α for adjacent clusters (mild het. spillover)
    nodes_per_cluster=2,   # clients per cluster (paper uses 2)
    num_informative_nodes = None,   # I — number of informative nodes (I < K)
    num_informative_patterns = None,  # P — number of exclusive patterns (P < n_classes)
    dir_level_I = 0.1,  
):
    """
    Split FashionMNIST (or any PyTorch Dataset) into K nodes in a non-IID fashion.

    mode='class_pair'  (original — 2 nodes per class)
    ────────────────────────────────────────────────────
        Nodes 0,1 → class 0;  nodes 2,3 → class 1;  ...
        Requires K even and K == 2 * n_classes.

    mode='class_multi'  (generalised class_pair)
    ─────────────────────────────────────────────
        Any number of nodes per class.
        Requires K divisible by n_classes.
        K // n_classes nodes share each class.

    mode='clustered'  (extreme silo, any K >= n_classes)
    ──────────────────────────────────────────────────────
        K nodes are partitioned into n_classes clusters as evenly as
        possible.  Every node in a cluster receives data from exactly
        ONE class.  Works for any K >= n_classes.

    mode='class_group'  (NEW — K < n_classes, groups of nodes share class subsets)
    ──────────────────────────────────────────────────────────────────────────────────
        For when K < n_classes.  Nodes are split into num_groups groups;
        classes are split into num_groups subsets.  Each node-group receives
        data from its corresponding class-subset.

        Example: K=6, n_classes=10, num_groups=3
          Group 0 → nodes [0,1] → classes [0,1,2,3]
          Group 1 → nodes [2,3] → classes [4,5,6]
          Group 2 → nodes [4,5] → classes [7,8,9]

        num_groups defaults to K//2 when not provided.
        Works for any K >= 2 with K < n_classes.

    mode='dirichlet'  (soft heterogeneity)
    ────────────────────────────────────────
        Each node receives data sampled from Dir(α) over classes.
        Small α → highly non-IID;  large α → nearly IID.
        Works for any K and any number of classes.

    Args:
        dataset_train (torch.utils.data.Dataset): PyTorch training dataset
        dataset_test  (torch.utils.data.Dataset): PyTorch test dataset
        K             (int)   : number of nodes
        val_ratio     (float) : fraction of each node's data for validation
                                (0.0 = skip val split, give all to train)
        seed          (int)   : RNG seed
        mode          (str)   : 'class_pair' | 'class_multi' | 'clustered' |
                                'class_group' | 'dirichlet'
        alpha         (float) : Dirichlet concentration (mode='dirichlet' only)
        num_groups    (int)   : Number of node/class groups (mode='class_group' only).
                                Defaults to K // 2.

    Returns:
        data_chunks: dict mapping node_id -> {
            'train': (data_tensor, label_tensor),
            'val':   (data_tensor, label_tensor)
        }
        test_data: (data_tensor, label_tensor) for global test set
    """

    # assert mode in ('class_pair', 'class_multi', 'clustered', 'class_group', 'dirichlet', 'clustered_dirichlet', 'spatial','cobo_split'), (
    #     "mode must be one of: 'class_pair', 'class_multi', 'clustered', 'class_group', 'dirichlet'."
    # )
    assert mode in ('class_pair', 'class_multi', 'clustered', 'class_group',
                     'dirichlet', 'clustered_dirichlet', 'spatial',
                     'cobo_split', 'informative_nodes'), (
        "mode must be one of: 'class_pair', 'class_multi', 'clustered', "
        "'class_group', 'dirichlet', 'clustered_dirichlet', 'spatial', "
        "'cobo_split', 'informative_nodes'."
    )

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ─────────────────────────────────────────────────────────────
    # Extract data and labels from PyTorch dataset
    # ─────────────────────────────────────────────────────────────
    data_train  = dataset_train.data
    label_train = dataset_train.targets

    if not isinstance(data_train, torch.Tensor):
        data_train = torch.tensor(data_train)
    if not isinstance(label_train, torch.Tensor):
        label_train = torch.tensor(label_train)

    label_sorted, indices = torch.sort(label_train.clone().detach())
    data_sorted = data_train[indices]

    classes   = torch.unique(label_sorted)
    n_classes = len(classes)

    print(f"\n[Dataset Info] Total train samples: {len(data_train)}, "
          f"Classes: {n_classes}, Test samples: {len(dataset_test)}")

    class_indices = {
        c.item(): (label_sorted == c).nonzero(as_tuple=True)[0].numpy()
        for c in classes
    }

    def _make_data_chunks(node_indices_list, node_labels_override=None):
        data_chunks = {}

        for node_id, idx_array in enumerate(node_indices_list):
            node_data = data_sorted[idx_array]

            # ── Use permuted labels if provided (cobo_split), else original ──
            if node_labels_override is not None and node_id in node_labels_override:
                node_labels = node_labels_override[node_id]
            else:
                node_labels = label_sorted[idx_array]

            # ── Empty node ───────────────────────────────────────
            if len(idx_array) == 0:
                data_chunks[node_id] = {
                    'train': (torch.empty(0, *data_train.shape[1:]),
                            torch.empty(0, dtype=torch.long)),
                    'val':   (torch.empty(0, *data_train.shape[1:]),
                            torch.empty(0, dtype=torch.long))
                }
                print(f"  Node {node_id}: no data, skipped.")
                continue

            # ── No validation split ──────────────────────────────
            if val_ratio == 0.0:
                data_chunks[node_id] = {
                    'train': (node_data, node_labels),
                    'val':   (torch.empty(0, *data_train.shape[1:]),
                            torch.empty(0, dtype=torch.long))
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
                'val':   (node_data[idx_vl], node_labels[idx_vl])
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

        data_chunks = _make_data_chunks(node_indices)
        test_data   = (dataset_test.data, dataset_test.targets)
        return data_chunks, test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 2: class_multi
    # ─────────────────────────────────────────────────────────────
    if mode == 'class_multi':
        assert K % n_classes == 0, (
            f"K={K} must be divisible by n_classes={n_classes}.\n"
            f"Valid K for {n_classes} classes: {[n_classes * m for m in range(1, 10)]}."
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

        data_chunks = _make_data_chunks(node_indices)
        test_data   = (dataset_test.data, dataset_test.targets)
        return data_chunks, test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 3: clustered
    # ─────────────────────────────────────────────────────────────
    if mode == 'clustered':
        assert K >= n_classes, (
            f"K={K} must be >= n_classes={n_classes} for mode='clustered'. "
            f"Use mode='class_group' for K < n_classes."
        )

        node_ids = np.arange(K)
        clusters = np.array_split(node_ids, n_classes)

        cluster_sizes = [len(c) for c in clusters]
        print(f"[split | clustered]  K={K}, n_classes={n_classes}, "
              f"cluster sizes={cluster_sizes} (cluster c → class c exclusively)")

        node_indices = [None] * K

        for c_idx, c in enumerate(classes):
            cluster = clusters[c_idx]
            idx     = class_indices[c.item()]
            np.random.default_rng(seed + c_idx).shuffle(idx)

            data_chunks_class = np.array_split(idx, len(cluster))
            for node_id, chunk in zip(cluster, data_chunks_class):
                node_indices[node_id] = chunk

        data_chunks = _make_data_chunks(node_indices)
        test_data   = (dataset_test.data, dataset_test.targets)
        return data_chunks, test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 4: class_group  (K < n_classes — node groups share class subsets)
    # ─────────────────────────────────────────────────────────────
    if mode == 'class_group':
        assert K >= 2, "K must be >= 2 for mode='class_group'."
        assert K < n_classes, (
            f"mode='class_group' is designed for K < n_classes. "
            f"Got K={K}, n_classes={n_classes}. "
            f"Use mode='clustered' for K >= n_classes."
        )

        # Determine number of groups G
        G = num_groups if num_groups is not None else max(1, K // 2)
        assert 1 <= G <= K, (
            f"num_groups={G} must be between 1 and K={K}."
        )
        assert G <= n_classes, (
            f"num_groups={G} must be <= n_classes={n_classes}."
        )

        # Split K nodes into G groups (as evenly as possible)
        node_ids    = np.arange(K)
        node_groups = np.array_split(node_ids, G)

        # Split n_classes class indices into G subsets (as evenly as possible)
        class_ids    = np.arange(n_classes)
        class_groups = np.array_split(class_ids, G)

        node_group_sizes  = [len(g) for g in node_groups]
        class_group_sizes = [len(c) for c in class_groups]
        print(f"[split | class_group]  K={K}, n_classes={n_classes}, G={G}")
        print(f"  Node group sizes : {node_group_sizes}")
        print(f"  Class group sizes: {class_group_sizes}")
        for g_idx in range(G):
            ng = node_groups[g_idx].tolist()
            cg = class_groups[g_idx].tolist()
            print(f"  Group {g_idx}: nodes {ng} → classes {cg}")

        node_indices = [None] * K

        for g_idx in range(G):
            node_group  = node_groups[g_idx]   # e.g. array([0, 1])
            class_group = class_groups[g_idx]  # e.g. array([0, 1, 2, 3])

            # Pool all samples from classes in this group
            pooled_idx = np.concatenate([
                class_indices[int(classes[c_id].item())]
                for c_id in class_group
            ])
            np.random.default_rng(seed + g_idx).shuffle(pooled_idx)

            # Split pooled data equally across nodes in this group
            node_chunks = np.array_split(pooled_idx, len(node_group))
            for node_id, chunk in zip(node_group, node_chunks):
                node_indices[node_id] = chunk

        data_chunks = _make_data_chunks(node_indices)
        test_data   = (dataset_test.data, dataset_test.targets)
        return data_chunks, test_data

    # ─────────────────────────────────────────────────────────────
    # Mode 5: dirichlet  (soft heterogeneity)
    # ─────────────────────────────────────────────────────────────
    if mode == 'dirichlet':
        print(f"[split | dirichlet]  K={K}, n_classes={n_classes}, α={alpha}")

        proportions = np.random.dirichlet(alpha=np.repeat(alpha, K), size=len(classes))

        node_indices = [[] for _ in range(K)]

        for class_idx, c in enumerate(classes):
            c_indices       = class_indices[c.item()]
            n_class_samples = len(c_indices)

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

        data_chunks = _make_data_chunks(node_indices)
        test_data   = (dataset_test.data, dataset_test.targets)
        return data_chunks, test_data
    # ─────────────────────────────────────────────────────────────
    # Mode 6: clustered_dirichlet
    # (groups of nodes + Dirichlet within each group's class subset)
    # ─────────────────────────────────────────────────────────────
    if mode == 'clustered_dirichlet':
        G = num_groups if num_groups is not None else max(1, K // 2)
        assert 1 <= G <= K,        f"num_groups={G} must be between 1 and K={K}."
        assert G <= n_classes,     f"num_groups={G} must be <= n_classes={n_classes}."

        node_ids     = np.arange(K)
        node_groups  = np.array_split(node_ids, G)           # G node groups
        class_ids    = np.arange(n_classes)
        class_groups = np.array_split(class_ids, G)          # G class subsets

        print(f"[split | clustered_dirichlet]  K={K}, n_classes={n_classes}, G={G}, α={alpha}")
        for g in range(G):
            print(f"  Group {g}: nodes {node_groups[g].tolist()} "
                f"→ classes {class_groups[g].tolist()}")

        node_indices = [[] for _ in range(K)]

        for g_idx in range(G):
            node_group  = node_groups[g_idx]    # nodes in this group
            class_group = class_groups[g_idx]   # classes available to this group
            n_g         = len(node_group)       # nodes in this group
            n_c         = len(class_group)      # classes in this group

            rng = np.random.default_rng(seed + g_idx)

            # Dirichlet proportions: shape (n_c, n_g) — per class, how to split across nodes
            proportions = rng.dirichlet(alpha=np.repeat(alpha, n_g), size=n_c)

            for local_c_idx, global_c_id in enumerate(class_group):
                c          = classes[global_c_id].item()
                c_indices  = class_indices[c].copy()
                rng.shuffle(c_indices)
                n_samples  = len(c_indices)

                # Allocate samples to nodes in this group via Dirichlet proportions
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

        data_chunks = _make_data_chunks(node_indices)
        test_data   = (dataset_test.data, dataset_test.targets)
        return data_chunks, test_data
    if mode == 'spatial':

        G = num_groups if num_groups is not None else max(2, K // 2)
        assert 2 <= G <= K,       f"num_groups={G} must be in [2, K={K}]."
        assert G <= n_classes,    f"num_groups={G} must be <= n_classes={n_classes}."

        # ── 1. Partition ordered nodes into G consecutive clusters ─────────
        node_ids    = np.arange(K)
        node_groups = np.array_split(node_ids, G)          # list of G arrays

        # ── 2. Assign disjoint class subsets to each cluster ───────────────
        class_ids    = np.arange(n_classes)
        class_groups = np.array_split(class_ids, G)        # list of G arrays

        # ── 3. Precompute cross-cluster α matrix (G × G) ──────────────────
        #   Entry [i, j]: concentration that cluster i gets from cluster j's classes.
        #   Diagonal (i==j): within-cluster α (this is the dominant signal).
        #   decay_lambda controls how fast cross-α falls with distance.
        decay_lambda = 2.0   # exp(-2*1)≈0.135, exp(-2*2)≈0.018 — fast falloff
        cross_alpha  = np.zeros((G, G))
        for gi in range(G):
            for gj in range(G):
                dist = abs(gi - gj)
                if dist == 0:
                    cross_alpha[gi, gj] = alpha           # within-cluster
                else:
                    # decays from alpha_max toward alpha_min as distance grows
                    cross_alpha[gi, gj] = (alpha_min
                        + (alpha_max - alpha_min) * np.exp(-decay_lambda * dist))

        print(f"[split | spatial]  K={K}, n_classes={n_classes}, G={G}")
        print(f"  alpha (within)={alpha:.3f}, alpha_min={alpha_min:.4f}, "
              f"alpha_max={alpha_max:.3f}")
        print(f"  Cross-cluster α matrix (G×G):")
        for gi in range(G):
            row = "  ".join(f"{cross_alpha[gi, gj]:.3f}" for gj in range(G))
            node_range = node_groups[gi].tolist()
            cls_range  = class_groups[gi].tolist()
            print(f"    Cluster {gi} (nodes {node_range}, classes {cls_range}): [{row}]")

        # ── 4. Build per-node class proportions ────────────────────────────
        #   For each node in cluster gi, draw a Dirichlet sample per cluster gj,
        #   then concatenate into a full n_classes proportions vector.

        rng = np.random.default_rng(seed)
        node_props = np.zeros((K, n_classes))   # row k → class proportions for node k

        for gi, node_group in enumerate(node_groups):
            n_g = len(node_group)

            # Build the concentration vector over ALL n_classes for each node in gi.
            # Classes from cluster gj get concentration cross_alpha[gi, gj].
            conc_vec = np.zeros(n_classes)
            for gj, cls_group in enumerate(class_groups):
                conc_vec[cls_group] = cross_alpha[gi, gj]

            # Each node draws independently from Dir(conc_vec)
            for local_idx, node_id in enumerate(node_group):
                node_rng = np.random.default_rng(seed + int(node_id))
                props = node_rng.dirichlet(conc_vec)
                node_props[node_id] = props

        # ── 5. Allocate samples per class ──────────────────────────────────
        node_indices = [[] for _ in range(K)]

        for class_idx, c in enumerate(classes):
            c_indices = class_indices[c.item()].copy()
            rng.shuffle(c_indices)
            n_samples = len(c_indices)

            # Column class_idx of node_props gives each node's share of this class
            col  = node_props[:, class_idx]
            col  = col / col.sum()                          # normalise across nodes

            samples_per_node = (col * n_samples).astype(int)

            # Fix rounding: assign remainders to nodes with largest fractional parts
            frac      = col * n_samples - samples_per_node
            remainder = n_samples - samples_per_node.sum()
            top_nodes = np.argsort(-frac)[:remainder]
            samples_per_node[top_nodes] += 1

            start = 0
            for k in range(K):
                count = samples_per_node[k]
                if count > 0:
                    node_indices[k].extend(c_indices[start:start + count])
                start += count

        node_indices = [np.array(sorted(idx)) for idx in node_indices]

        data_chunks = _make_data_chunks(node_indices)

        test_data = (dataset_test.data, dataset_test.targets)
        return data_chunks, test_data
    
    # ─────────────────────────────────────────────────────────────
    # Mode 8: cobo_split  (COBO NeurIPS 2024 — label permutation)
    # ─────────────────────────────────────────────────────────────
    if mode == 'cobo_split':
        assert K % nodes_per_cluster == 0, (
            f"K={K} must be divisible by nodes_per_cluster={nodes_per_cluster}."
        )
        G = K // nodes_per_cluster   # number of clusters

        print(f"[split | cobo_split]  K={K}, n_classes={n_classes}, "
              f"G={G} clusters × {nodes_per_cluster} nodes each")

        # ── Step 1: build cluster membership ─────────────────────
        node_ids    = np.arange(K)
        node_groups = np.array_split(node_ids, G)   # G arrays of node IDs

        # ── Step 2: pool ALL training data, split evenly per cluster
        all_indices = np.arange(len(data_sorted))
        rng_pool    = np.random.default_rng(seed)
        rng_pool.shuffle(all_indices)

        # Each cluster gets an equal share of the full dataset
        cluster_pools = np.array_split(all_indices, G)

        # ── Step 3: within each cluster, split pool equally across nodes
        node_indices = [None] * K
        for g_idx, node_group in enumerate(node_groups):
            pool = cluster_pools[g_idx].copy()
            np.random.default_rng(seed + g_idx).shuffle(pool)
            node_chunks = np.array_split(pool, nodes_per_cluster)
            for node_id, chunk in zip(node_group, node_chunks):
                node_indices[node_id] = np.sort(chunk)

        # ── Step 4: generate one unique label permutation per cluster
        #   π_c: {0,...,C-1} → {0,...,C-1} bijection, distinct per cluster
        class_list = classes.numpy()    # [0, 1, ..., C-1]

        permutations = {}
        for g_idx in range(G):
            perm_rng = np.random.default_rng(seed + 1000 + g_idx)  # distinct seed space
            perm     = class_list.copy()
            perm_rng.shuffle(perm)
            permutations[g_idx] = perm
            node_range = node_groups[g_idx].tolist()
            print(f"  Cluster {g_idx} (nodes {node_range}):  "
                  f"π_{g_idx} = {class_list.tolist()} → {perm.tolist()}")

        # ── Step 5: apply permutations — build per-node label tensors
        node_labels_override = {}
        for g_idx, node_group in enumerate(node_groups):
            perm = permutations[g_idx]          # length-C array: perm[y] = new_y
            for node_id in node_group:
                idx_array    = node_indices[node_id]
                orig_labels  = label_sorted[idx_array]        # original labels
                # remap: apply π_c element-wise
                permuted     = torch.tensor(
                    perm[orig_labels.numpy()], dtype=torch.long
                )
                node_labels_override[node_id] = permuted

        # ── Step 6: log per-node stats ────────────────────────────
        for node_id in range(K):
            idx_array = node_indices[node_id]
            perm_labs = node_labels_override[node_id]
            dist      = dict(zip(*np.unique(perm_labs.numpy(), return_counts=True)))
            g_idx     = next(gi for gi, ng in enumerate(node_groups)
                             if node_id in ng)
            print(f"  Node {node_id} (cluster {g_idx}): "
                  f"n={len(idx_array)}, permuted_dist={dist}")

        # ── Step 7: build data_chunks with permuted labels ────────
        data_chunks = _make_data_chunks(node_indices,
                                        node_labels_override=node_labels_override)

        # ── Test set: NOT permuted (global evaluation on original labels)
        test_data = (dataset_test.data, dataset_test.targets)
        return data_chunks, test_data
        # ─────────────────────────────────────────────────────────────

    if mode == 'informative_nodes':
        I = num_informative_nodes
        P = num_informative_patterns
        assert I is not None and P is not None, (
            "num_informative_nodes (I) and num_informative_patterns (P) "
            "must be provided for mode='informative_nodes'."
        )
        assert 1 <= I < K, f"I={I} must satisfy 1 <= I < K={K}."
        assert 1 <= P < n_classes, f"P={P} must satisfy 1 <= P < n_classes={n_classes}."

        # ── Randomly select I informative nodes out of K ──────────────
        rng_sel = np.random.default_rng(seed + 999)
        informative_nodes = rng_sel.choice(K, size=I, replace=False)
        informative_nodes.sort()

        class_ids            = np.arange(n_classes)
        informative_classes  = class_ids[:P]
        rest_classes         = class_ids[P:]

        print(f"[split | informative_nodes]  K={K}, n_classes={n_classes}, I={I}, P={P}")
        print(f"  Randomly selected informative nodes: {informative_nodes.tolist()}")
        print(f"  Exclusive classes {informative_classes.tolist()} -> only informative "
              f"nodes  (dir_level_I={dir_level_I})")
        print(f"  Remaining classes {rest_classes.tolist()} -> ALL {K} nodes "
              f"(dir_level={alpha})")

        node_indices = [[] for _ in range(K)]

        # ── Step 1: exclusive P classes -> ONLY the I informative nodes ──
        rng_I = np.random.default_rng(seed)
        if I > 0 and P > 0:
            proportions_I = rng_I.dirichlet(alpha=np.repeat(dir_level_I, I), size=P)

            for local_c_idx, global_c_id in enumerate(informative_classes):
                c         = classes[global_c_id].item()
                c_indices = class_indices[c].copy()
                rng_I.shuffle(c_indices)
                n_samples = len(c_indices)

                samples_per_node = (proportions_I[local_c_idx] * n_samples).astype(int)
                diff = n_samples - samples_per_node.sum()
                for i in range(diff):
                    samples_per_node[i % I] += 1

                start = 0
                for local_node_idx, node_id in enumerate(informative_nodes):
                    count = samples_per_node[local_node_idx]
                    if count > 0:
                        node_indices[node_id].extend(c_indices[start:start + count])
                        start += count

        # ── Step 2: remaining (C-P) classes -> ALL K nodes (incl. informative) ──
        rng_rest = np.random.default_rng(seed + 500)
        if K > 0 and len(rest_classes) > 0:
            proportions_rest = rng_rest.dirichlet(
                alpha=np.repeat(alpha, K), size=len(rest_classes)
            )

            for local_c_idx, global_c_id in enumerate(rest_classes):
                c         = classes[global_c_id].item()
                c_indices = class_indices[c].copy()
                rng_rest.shuffle(c_indices)
                n_samples = len(c_indices)

                samples_per_node = (proportions_rest[local_c_idx] * n_samples).astype(int)
                diff = n_samples - samples_per_node.sum()
                for i in range(diff):
                    samples_per_node[i % K] += 1

                start = 0
                for node_id in range(K):
                    count = samples_per_node[node_id]
                    if count > 0:
                        node_indices[node_id].extend(c_indices[start:start + count])
                        start += count

        node_indices = [np.array(sorted(idx)) for idx in node_indices]

        data_chunks = _make_data_chunks(node_indices)
        test_data   = (dataset_test.data, dataset_test.targets)
        return data_chunks, test_data


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
        ## Caveman(3, 7)
        # G = nx.connected_caveman_graph(3, 7)
        ## Caveman(2, 7)
        G = nx.connected_caveman_graph(2, 7)
        ## Caveman(2, 5)
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
def compute_stochastic_grad(model, loader, criterion, device):
    model.zero_grad()
    X_batch, y_batch = next(iter(loader))
    X_batch = X_batch.unsqueeze(1).to(device)   # [B,28,28] → [B,1,28,28]
    y_batch = y_batch.to(device)
    with torch.enable_grad():
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


# ─────────────────────────────────────────────────────────────────
# Helper: midpoint model z_{i,j} = (y_i^t + y_j^t) / 2
#         and its gradients w.r.t. f_i^{tr} and f_j^{tr}
# ─────────────────────────────────────────────────────────────────
def compute_midpoint_grads(model_i, model_j, loader_i, loader_j,
                           criterion, model_factory, device):
    z_ij = model_factory().to(device)

    with torch.no_grad():
        for (_, p_i), (_, p_j), (_, p_z) in zip(
            model_i.named_parameters(),
            model_j.named_parameters(),
            z_ij.named_parameters(),
        ):
            p_z.copy_((p_i + p_j) / 2.0)

    # ∇_{y} f_i^{tr}(z_{i,j} ; ζ_i^t)
    with torch.enable_grad():
        z_ij.zero_grad()
        X_i, y_i = next(iter(loader_i))
        X_i = X_i.unsqueeze(1).to(device)       # [B,28,28] → [B,1,28,28]
        y_i = y_i.to(device)
        criterion(z_ij(X_i), y_i).backward()
        grad_i_mid = [p.grad.detach().clone() for p in z_ij.parameters()]

    # ∇_{y} f_j^{tr}(z_{i,j} ; ζ_j^t)
    with torch.enable_grad():
        z_ij.zero_grad()
        X_j, y_j = next(iter(loader_j))
        X_j = X_j.unsqueeze(1).to(device)       # [B,28,28] → [B,1,28,28]
        y_j = y_j.to(device)
        criterion(z_ij(X_j), y_j).backward()
        grad_j_mid = [p.grad.detach().clone() for p in z_ij.parameters()]

    return grad_i_mid, grad_j_mid


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


# ─────────────────────────────────────────────────────────────────
# Helper: pairwise disagreement
#   Σ_{i<j} x_{i,j} ||y_i - y_j||_2^2
# ─────────────────────────────────────────────────────────────────
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
