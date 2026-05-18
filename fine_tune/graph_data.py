"""Aligned static-graph loading for fine-tuning."""
import os
import sys
import warnings

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths import GRAPH_STRUCTURE_DIR, PROCESSED_DIR, STATIC_DATA_DIR


def expected_num_nodes(dataset_name: str) -> int:
    npy_path = os.path.join(PROCESSED_DIR, f"ml_{dataset_name}_node.npy")
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Missing node features: {npy_path}")
    return int(np.load(npy_path, mmap_mode="r").shape[0])


def load_static_edge_index(dataset_name: str) -> np.ndarray:
    gs_path = os.path.join(GRAPH_STRUCTURE_DIR, f"{dataset_name}_edge_index.npy")
    if os.path.isfile(gs_path):
        return np.load(gs_path)
    pt_path = os.path.join(STATIC_DATA_DIR, f"{dataset_name}.pt")
    data = torch.load(pt_path, map_location="cpu")
    edge_index = data.edge_index if hasattr(data, "edge_index") else data["edge_index"]
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    return edge_index


def validate_graph_alignment(dataset_name: str, num_nodes: int, edge_index: np.ndarray) -> None:
    max_id = int(edge_index.max())
    if max_id >= num_nodes:
        raise ValueError(
            f"[{dataset_name}] edge_index max id {max_id} >= num_nodes {num_nodes}."
        )
    if int(edge_index.min()) < 0:
        raise ValueError(f"[{dataset_name}] negative node id in edge_index.")

    pt_path = os.path.join(STATIC_DATA_DIR, f"{dataset_name}.pt")
    if os.path.isfile(pt_path):
        d = torch.load(pt_path, map_location="cpu")
        static_n = int(d.num_nodes) if hasattr(d, "num_nodes") else d.x.shape[0]
        if os.path.isfile(
            os.path.join(GRAPH_STRUCTURE_DIR, f"{dataset_name}_edge_index.npy")
        ) and static_n < num_nodes:
            warnings.warn(
                f"[{dataset_name}] static/data/{dataset_name}.pt has {static_n} nodes "
                f"but graph expects {num_nodes}; using graph_structure for GCN adjacency.",
                stacklevel=2,
            )
