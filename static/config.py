# File: config.py
import argparse
from datetime import datetime
import os
import random
import sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths import STATIC_DATA_DIR, STATIC_DIR, STATIC_SAVE_MODEL_DIR, PROCESSED_DIR
# lastfm mooc reddit wikipedia
def get_pretrain_datasets(args):
    """Get pretraining dataset name list"""
    data_name = args.dataset
    datasets = []
    
    if data_name == 'genre':
        # Citeseer, Pubmed, P-home, Wiki-CS
        datasets = ['mooc', 'reddit', 'wikipedia']
    elif data_name == 'mooc':
        # Cora, Pubmed, P-home, Wiki-CS
        datasets = ['genre', 'reddit', 'wikipedia']
    elif data_name == 'reddit':
        # Cora, Citeseer, P-home, Wiki-CS
        datasets = ['genre', 'mooc', 'wikipedia']
    elif data_name == 'wikipedia':
        # Cora, Citeseer, Pubmed, P-home, Wiki-CS
        datasets = ['genre', 'mooc', 'reddit']
    elif data_name == 'all':
        datasets = ['genre', 'mooc', 'reddit', 'wikipedia']
    return datasets

def set_seed(seed):
    """Set random seed to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_num_nodes(args):
    """Get number of nodes without loading full PyG graph when possible."""
    metadata_path = os.path.join(args.data_dir, f"{args.dataset}_metadata.pkl")
    if os.path.isfile(metadata_path):
        import pickle
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        if "num_nodes" in metadata:
            return metadata["num_nodes"]
    npy_path = os.path.join(PROCESSED_DIR, f"ml_{args.dataset}_node.npy")
    if os.path.isfile(npy_path):
        return int(np.load(npy_path, mmap_mode="r").shape[0])
    data = torch.load(args.data_path)
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        return data.num_nodes
    return data.x.shape[0]

def get_args():
    parser = argparse.ArgumentParser("pre_train")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Basic parameters
    parser.add_argument("--dataset", type=str, default="genre", help="Dataset name")
    parser.add_argument("--seed", type=int, default=39, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--unify_dim", type=int, default=64, help="Unified feature dimension")
    
    # Model parameters
    parser.add_argument("--hid_units", type=int, default=256, help="GCN hidden layer dimension")
    parser.add_argument("--out_channels", type=int, default=64, help="Output embedding dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GCN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--l2_coef", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--nb_epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=200, help="Early stopping patience")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluation interval steps")

    # Enable caching mechanism
    parser.add_argument("--use_cache", default=True, type=bool, help="Whether to enable caching mechanism")
    # Negative sampling parameters
    parser.add_argument("--neg_samples", type=int, default=50, help="Number of negative samples per positive sample")

    # Multi-dataset joint pretraining parameters
    parser.add_argument("--kl_weight", type=float, default=0, help="Weight of KL divergence loss")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Static graph data dir (default: <DyGFM>/static/data)",
    )

    args = parser.parse_args()

    args.base_dir = STATIC_DIR
    if args.data_dir is None:
        args.data_dir = STATIC_DATA_DIR
    args.data_path = os.path.join(args.data_dir, f"{args.dataset}.pt")
    args.cache_path = os.path.join(args.data_dir, "cache")

    save_dir = STATIC_SAVE_MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{args.dataset}_{current_time}.pt"
    args.save_path = os.path.join(save_dir, save_name)
    
    set_seed(args.seed)
    if args.dataset != 'all':
        args.num_nodes = get_num_nodes(args)

    if args.use_cache:
        os.makedirs(args.cache_path, exist_ok=True)
        # Combined matrix path
        args.all_adj_path = f"{args.cache_path}/all_adj_{args.dataset}.pt"
        # Combined feature matrix path
        args.all_features_path = f"{args.cache_path}/all_features_{args.dataset}.pt"
    return args