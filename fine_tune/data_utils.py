import os
import sys

import torch
import numpy as np
from config import *
import pandas as pd
from data_processing import *
import scipy.sparse as sp

_FT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FT_ROOT not in sys.path:
    sys.path.insert(0, _FT_ROOT)
from paths import (
    DOWNSTREAM_DATA_DIR,
    DYNAMIC_EDGE_FEATURE_DIR,
    DYNAMIC_FEATURES_DIR,
    PROCESSED_DIR,
    STATIC_DATA_DIR,
)
from graph_data import (
    expected_num_nodes,
    load_static_edge_index,
    validate_graph_alignment,
)

def get_checkpoint_path(epoch, args):
    """
    Gets the model checkpoint path.
    Args:
        epoch: The current epoch number.
        args: Arguments.
    Returns:
        The path to the checkpoint file.
    """
    return f"{args.node_checkpoint_dir}/{args.prefix}_{args.dataset}_{epoch}_node_classification.pth"

def get_adj_tensor(args, num_nodes=None, device=None):
    if num_nodes is None:
        num_nodes = expected_num_nodes(args.dataset)
    edge_index = load_static_edge_index(args.dataset)
    validate_graph_alignment(args.dataset, num_nodes, edge_index)
    adj = edge_index_to_sparse(edge_index, num_nodes=num_nodes)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    if device is not None:
        adj = adj.to(device)
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Converts a SciPy sparse matrix to a PyTorch sparse tensor.
    Args:
        sparse_mx: A scipy.sparse matrix.
    Returns:
        A torch.sparse.FloatTensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def edge_index_to_sparse(edge_index, num_nodes=None, edge_weight=None):
    """
    Converts an edge_index in PyTorch Geometric format to a scipy.sparse.csr_matrix.
    Args:
        edge_index: torch.Tensor of shape [2, num_edges].
        num_nodes: Optional, the number of nodes in the graph. Inferred if None.
        edge_weight: Optional, torch.Tensor of shape [num_edges]. Defaults to 1 for all edges if None.
    Returns:
        A scipy.sparse.csr_matrix representing the adjacency matrix.
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    row, col = edge_index
    if num_nodes is None:
        num_nodes = max(edge_index.max() + 1, edge_index.shape[1])
    if edge_weight is None:
        data = np.ones(edge_index.shape[1])
    else:
        if isinstance(edge_weight, torch.Tensor):
            data = edge_weight.cpu().numpy()
        else:
            data = edge_weight
    adj_matrix = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj_matrix

def get_raw_feat(dataset_name, args=None):
    node_npy = os.path.join(PROCESSED_DIR, f"ml_{dataset_name}_node.npy")
    if os.path.isfile(node_npy):
        return torch.from_numpy(np.load(node_npy)).float()
    feat_pt = os.path.join(DYNAMIC_FEATURES_DIR, f"{dataset_name}.pt")
    raw_data = torch.load(feat_pt)
    return raw_data["x"] if isinstance(raw_data, dict) else raw_data.x

def get_d_data(args, dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
    if dataset_name == 'genre':
        csv_path = os.path.join(
            DOWNSTREAM_DATA_DIR,
            dataset_name,
            f"ds_{dataset_name}_{args.genre_class}.csv",
        )
    else:
        csv_path = os.path.join(
            DOWNSTREAM_DATA_DIR, dataset_name, f"ds_{dataset_name}.csv"
        )
    graph_df = pd.read_csv(csv_path)
    edge_features = np.load(os.path.join(PROCESSED_DIR, f"ml_{dataset_name}.npy"))
    node_feat_tensor = get_raw_feat(dataset_name)
    node_features = node_feat_tensor.numpy()
    if randomize_features:
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1])
    val_time, test_time = list(np.quantile(graph_df.ts, [0.10, 0.20]))
    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)
    test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time
    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])
        edge_contains_new_val_node_mask = np.array([(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array([(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    else:
        edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], labels[new_node_val_mask])
    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions, full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(test_data.n_interactions, test_data.n_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))
    return node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data

def create_adj_list(data):
    """
    Creates an adjacency list.
    Args:
        data: A data object.
    Returns:
        The adjacency list.
    """
    sources = data['sources'].numpy()
    destinations = data['destinations'].numpy()
    edge_idxs = data['edge_idxs'].numpy()
    timestamps = data['timestamps'].numpy()
    max_node_idx = max(np.max(sources), np.max(destinations)) + 1
    adj_list = [[] for _ in range(max_node_idx)]
    for i in range(len(sources)):
        adj_list[sources[i]].append((destinations[i], edge_idxs[i], timestamps[i]))
        adj_list[destinations[i]].append((sources[i], edge_idxs[i], timestamps[i]))
    return adj_list

def split_data_by_time(data, val_ratio=0.7, test_ratio=0.8):
    """
    Splits the dataset chronologically.
    Args:
        data: The data object.
        val_ratio: The timestamp quantile to start the validation set (default 0.7).
        test_ratio: The timestamp quantile to start the test set (default 0.8).
    Returns:
        train_data: The training set.
        val_data: The validation set.
        test_data: The test set.
    """
    sources = data['sources']
    destinations = data['destinations']
    timestamps = data['timestamps']
    edge_idxs = data['edge_idxs']
    timestamps_np = timestamps.numpy()
    val_time = np.quantile(timestamps_np, val_ratio)
    test_time = np.quantile(timestamps_np, test_ratio)
    train_mask = timestamps <= val_time
    val_mask = (timestamps > val_time) & (timestamps <= test_time)
    test_mask = timestamps > test_time
    train_data = {
        'sources': sources[train_mask],
        'destinations': destinations[train_mask],
        'timestamps': timestamps[train_mask],
        'edge_idxs': edge_idxs[train_mask]
    }
    val_data = {
        'sources': sources[val_mask],
        'destinations': destinations[val_mask],
        'timestamps': timestamps[val_mask],
        'edge_idxs': edge_idxs[val_mask]
    }
    test_data = {
        'sources': sources[test_mask],
        'destinations': destinations[test_mask],
        'timestamps': timestamps[test_mask],
        'edge_idxs': edge_idxs[test_mask]
    }
    print(f"Dataset split statistics:")
    print(f"Total samples: {len(sources)}")
    print(f"Training samples: {len(train_data['sources'])}, Ratio: {len(train_data['sources'])/len(sources):.2f}")
    print(f"Validation samples: {len(val_data['sources'])}, Ratio: {len(val_data['sources'])/len(sources):.2f}")
    print(f"Test samples: {len(test_data['sources'])}, Ratio: {len(test_data['sources'])/len(sources):.2f}")
    return train_data, val_data, test_data

def load_dataset(dataset_name, args):
    """
    Loads a dataset and splits it chronologically.
    Args:
        dataset_name: The name of the dataset.
        args: Arguments.
    Returns:
        node_features, edge_features, full_data, train_data, val_data, test_data, adj_list
    """
    data = torch.load(os.path.join(DYNAMIC_FEATURES_DIR, f"{dataset_name}.pt"))
    node_features = data["x"] if isinstance(data, dict) else data.x
    full_data = torch.load(os.path.join(args.normal_time_dir, f"{dataset_name}.pt"))
    edge_features = torch.load(os.path.join(DYNAMIC_EDGE_FEATURE_DIR, f"{dataset_name}.pt"))
    edge_features = torch.tensor(edge_features)
    train_data, val_data, test_data = split_data_by_time(full_data)
    adj_list = create_adj_list(full_data)
    return node_features, edge_features, full_data, train_data, val_data, test_data, adj_list

if __name__ == "__main__":
    args = get_args()
    node_features, edge_features, full_data, train_data, val_data, test_data, adj_list = load_dataset("lastfm", args)
    print(adj_list[0])