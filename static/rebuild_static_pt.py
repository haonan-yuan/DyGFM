#!/usr/bin/env python3
"""Rebuild static/data/{dataset}.pt from graph_structure + processed node features."""
import argparse
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data as PyGData

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paths import GRAPH_STRUCTURE_DIR, PROCESSED_DIR, STATIC_DATA_DIR


def rebuild(dataset: str, out_dir: str = STATIC_DATA_DIR) -> str:
    node_path = os.path.join(PROCESSED_DIR, f"ml_{dataset}_node.npy")
    edge_path = os.path.join(GRAPH_STRUCTURE_DIR, f"{dataset}_edge_index.npy")
    if not os.path.isfile(node_path):
        raise FileNotFoundError(node_path)
    if not os.path.isfile(edge_path):
        raise FileNotFoundError(edge_path)

    x = torch.from_numpy(np.load(node_path)).float()
    edge_index = torch.from_numpy(np.load(edge_path)).long()
    num_nodes = x.shape[0]
    if edge_index.max().item() >= num_nodes:
        raise ValueError(
            f"{dataset}: edge max {edge_index.max().item()} >= num_nodes {num_nodes}"
        )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}.pt")
    torch.save(PyGData(x=x, edge_index=edge_index, num_nodes=num_nodes), out_path)
    print(f"Saved {out_path}: nodes={num_nodes}, edges={edge_index.shape[1]}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="One dataset or omit for all four")
    args = parser.parse_args()
    from paths import DATASETS

    names = [args.dataset] if args.dataset else list(DATASETS)
    for name in names:
        rebuild(name)


if __name__ == "__main__":
    main()
