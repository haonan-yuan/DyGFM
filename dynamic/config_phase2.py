# File: config_phase2.py
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
from paths import (
    DYNAMIC_CHECKPOINTS_DIR,
    DYNAMIC_DATA_DIR,
    DYNAMIC_DIR,
    DYNAMIC_FEATURES_DIR,
    DYNAMIC_NORMAL_TIME_DIR,
    DYNAMIC_SAVE_MODEL_DIR,
)

# --- Common utility functions (same as in config.py for phase 1) ---

def get_pretrain_datasets(args):
    """Get pretraining dataset name list"""
    data_name = args.dataset
    datasets = []
    if data_name == 'genre':
        datasets = ['mooc', 'reddit', 'wikipedia']
    elif data_name == 'mooc':
        datasets = ['genre', 'reddit', 'wikipedia']
    elif data_name == 'reddit':
        datasets = ['genre', 'mooc', 'wikipedia']
    elif data_name == 'wikipedia':
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

# --- Phase 2 specific parameter configuration function ---

def get_args_phase2():
    """
    Configure parameters for independent phase 2 Adapter fine-tuning
    """
    parser = argparse.ArgumentParser("train_adapter_phase2")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    parser.add_argument("--dataset", type=str, default="genre", 
                        help="Main dataset name (must match the name used in phase 1 training)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")

    # --- Model structure parameters (must strictly match phase 1 model) ---
    parser.add_argument("--num_layers", type=int, default=1, help="Number of TGAT layers")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--attn_mode", type=str, default="prod", help="Attention mode")
    parser.add_argument("--node_feat_dim", type=int, default=172, help="Original node feature dimension")
    parser.add_argument("--edge_feat_dim", type=int, default=172, help="Original edge feature dimension")
    parser.add_argument("--time_dim", type=int, default=172, help="Time encoding dimension")
    parser.add_argument("--unify_dim", type=int, default=64, help="Unified node embedding dimension (embedding_dim)")
    
    # --- Phase 2 fine-tuning parameters ---
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adapter fine-tuning")
    parser.add_argument("--l2_coef", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # --- Parameters for debugging ---
    parser.add_argument("--epochs_per_domain", type=int, default=50, help="Fine-tuning epochs per domain")
    parser.add_argument("--alternating_cycles", type=int, default=2, help="Number of alternating fine-tuning cycles")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluation interval (epochs)")
    parser.add_argument("--num_neighbors", type=int, default=10, help="Number of neighbors to sample")
    
    parser.add_argument(
        "--phase1_model_path",
        type=str,
        default=None,
        help="Phase-1 checkpoint (default: dynamic/checkpoints/overall_best_model_{dataset}.pt)",
    )

    args = parser.parse_args()

    args.base_dir = DYNAMIC_DIR
    args.data_dir = DYNAMIC_DATA_DIR
    args.normal_time_dir = DYNAMIC_NORMAL_TIME_DIR
    args.features_dir = DYNAMIC_FEATURES_DIR
    if args.phase1_model_path is None:
        args.phase1_model_path = os.path.join(
            DYNAMIC_CHECKPOINTS_DIR, f"overall_best_model_{args.dataset}.pt"
        )

    save_dir = DYNAMIC_SAVE_MODEL_DIR
    os.makedirs(DYNAMIC_CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"tgat_{args.dataset}_{current_time}_phase2.pt"
    args.save_path = os.path.join(save_dir, save_name)
    
    set_seed(args.seed)
    return args