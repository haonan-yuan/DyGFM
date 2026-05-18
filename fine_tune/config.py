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
    DOWNSTREAM_DATA_DIR,
    DYNAMIC_DATA_DIR,
    DYNAMIC_EDGE_FEATURE_DIR,
    DYNAMIC_FEATURES_DIR,
    DYNAMIC_NORMAL_TIME_DIR,
    FINE_TUNE_CHECKPOINTS_DIR,
    FINE_TUNE_DIR,
    FINE_TUNE_SAVE_MODEL_DIR,
    GRAPH_STRUCTURE_DIR,
    PROCESSED_DIR,
    SENTENC_BRANCH_DIR,
    STATIC_DATA_DIR,
    STATIC_DIR,
    TIME_BRANCH_DIR,
)

# =================================================================================
# == Common utility functions (shared)
# =================================================================================

def get_pretrain_datasets(args):
    """
    Get pretraining dataset name list.
    Returns other datasets for pretraining based on the specified downstream task dataset.
    """
    data_name = args.dataset
    # genre mooc reddit wikipedia
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
    """
    Set global random seed to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    """
    Unified parameter configuration function.
    """
    parser = argparse.ArgumentParser("Fine_tune")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Base configuration
    parser.add_argument("--dataset", type=str, default="genre", help="Target downstream dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    
    # TGAT model parameters (don't change)
    parser.add_argument("--num_layers", type=int, default=1, help="Number of TGAT layers")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--attn_mode", type=str, default="prod", help="Attention mode ('prod' or 'map')")
    
    # Dimension parameters
    parser.add_argument("--node_feat_dim", type=int, default=172, help="Original node feature dimension")
    parser.add_argument("--edge_feat_dim", type=int, default=172, help="Original edge feature dimension")
    parser.add_argument("--time_dim", type=int, default=172, help="Time encoding dimension")
    parser.add_argument("--unify_dim", type=int, default=64, help="Unified node embedding dimension")
    parser.add_argument("--token_dim", type=int, default=64, help="Token dimension")
    parser.add_argument("--condition_dim", type=int, default=65, help="Condition dimension")
    parser.add_argument("--projection_input_dim", type=int, default=300, help="Projection input dimension")
    parser.add_argument("--projection_output_dim", type=int, default=172, help="Projection output dimension")

    # Training parameters
    parser.add_argument("--num_neighbors", type=int, default=10, help="Number of neighbors to sample")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluation interval")

    # Learning rates and L2 regularization coefficients for different models
    parser.add_argument("--projection_lr", type=float, default=1e-3, help="Projection learning rate")
    parser.add_argument("--timer_lr", type=float, default=1e-3, help="Timer learning rate")
    parser.add_argument("--timer_l2_coef", type=float, default=0.0, help="Timer weight decay (L2 regularization)")
    parser.add_argument("--moe_lr", type=float, default=1e-3, help="MoE learning rate")
    parser.add_argument("--moe_l2_coef", type=float, default=0.0, help="MoE weight decay (L2 regularization)")
    parser.add_argument("--prompt_generator_lr", type=float, default=1e-3, help="Prompt Generator learning rate")
    parser.add_argument("--prompt_generator_l2_coef", type=float, default=0.0, help="Prompt Generator weight decay")
    parser.add_argument("--projection_l2_coef", type=float, default=0.0, help="Projection weight decay")
    parser.add_argument("--classifier_edge_lr", type=float, default=1e-5, help="Edge Classifier learning rate")
    parser.add_argument("--classifier_edge_l2_coef", type=float, default=0.0, help="Edge Classifier weight decay")
    parser.add_argument("--proj_lr", type=float, default=1e-5, help="Proj learning rate")
    parser.add_argument("--proj_l2_coef", type=float, default=0.0, help="Proj weight decay")
    # Static model parameters
    parser.add_argument("--hid_units", type=int, default=256, help="GCN hidden layer dimension")
    parser.add_argument("--out_channels", type=int, default=64, help="Output embedding dimension")
    parser.add_argument("--num_static_layers", type=int, default=2, help="Number of GCN layers")
    parser.add_argument("--static_dropout", type=float, default=0.2, help="Dropout rate for static model")

    # Branch weights
    parser.add_argument("--branch_weight_static", type=float, default=0.5, help="Static branch divergence weight")

    # MoE loss weights
    parser.add_argument("--kl_weight", type=float, default=1e-6, help="KL divergence weight")
    parser.add_argument("--entropy_weight", type=float, default=1e-6, help="Entropy weight")
    
    # Feature weights
    parser.add_argument("--node_feature_static_weight", type=float, default=0.5, help="Static node feature weight")

    # Architecture parameters
    parser.add_argument("--bottle_neck_mlp_dim", type=int, default=32, help="Bottleneck MLP hidden dimension")

    # Transductive/inductive settings
    parser.add_argument('--different_new_nodes', type=bool, default=True, help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--new_node', type=bool, default=True, help='Enable new node modeling')

    # Neighbor sampling parameters
    parser.add_argument('--uniform', action='store_true', help='Use uniform sampling from temporal neighbors')

    # Experiment settings
    parser.add_argument('--num_runs', type=int, default=1, help='Number of experimental runs')

    # Loss weights
    parser.add_argument('--moe_loss_weight', type=float, default=5e-5, help='MoE loss weight')

    # Validation and testing parameters
    parser.add_argument("--val_batch_size", type=int, default=100, help="Validation batch size")
    parser.add_argument("--test_batch_size", type=int, default=100, help="Test batch size")
    parser.add_argument("--val_freq", type=int, default=5, help="Validation frequency")
    parser.add_argument("--prefix", type=str, default="hello", help="File prefix for saved results")

    # Few-shot learning parameters
    parser.add_argument("--task_num", type=int, default=1, help="Number of tasks")
    parser.add_argument("--train_shot_num", type=int, default=3, help="Number of training shots")
    parser.add_argument("--val_shot_num", type=int, default=3, help="Number of validation shots")
    parser.add_argument("--test_shot_num", type=int, default=100, help="Number of test shots")
    parser.add_argument("--name", type=str, default="", help="Run name prefix")
    parser.add_argument("--classifier_lr", type=float, default=1e-5, help="Node classifier learning rate")
    parser.add_argument("--classifier_l2_coef", type=float, default=0.0, help="Node classifier weight decay")

    # Genre-specific parameters
    parser.add_argument("--genre_class", type=int, default=1, help="Genre class specification")

    # Model selection
    parser.add_argument("--use_all", type=bool, default=False, help="Whether to use the 'all' base model")

    # Embedding weights
    parser.add_argument("--prompt_weight", type=float, default=0.0, help="Prompt weight")
    parser.add_argument("--routing_token_weight", type=float, default=0.0, help="Routing token weight")
    args = parser.parse_args()

    args.fine_tune_dir = FINE_TUNE_DIR
    args.fine_tuning_dir = FINE_TUNE_DIR  # alias used in some scripts
    args.static_dir = STATIC_DIR
    args.dynamic_dir = os.path.dirname(DYNAMIC_DATA_DIR)

    args.static_token_dim = 172
    args.dynamic_token_dim = 64

    args.static_token_dir = os.path.join(
        SENTENC_BRANCH_DIR, "embeddings", args.dataset
    )
    args.dynamic_token_dir = os.path.join(
        TIME_BRANCH_DIR, "embeddings", args.dataset
    )
    args.static_model_path = os.path.join(
        SENTENC_BRANCH_DIR, "saved_model", f"{args.dataset}.pt"
    )
    args.dynamic_model_path = os.path.join(
        TIME_BRANCH_DIR, "saved_model", f"{args.dataset}.pt"
    )

    os.makedirs(FINE_TUNE_SAVE_MODEL_DIR, exist_ok=True)
    os.makedirs(FINE_TUNE_CHECKPOINTS_DIR, exist_ok=True)
    save_name = f"tgat_{args.dataset}_{current_time}.pt"
    args.save_path = os.path.join(FINE_TUNE_SAVE_MODEL_DIR, save_name)

    args.processed_dir = PROCESSED_DIR
    args.downstream_data_dir = DOWNSTREAM_DATA_DIR
    args.graph_structure_dir = GRAPH_STRUCTURE_DIR
    args.dynamic_data_dir = DYNAMIC_DATA_DIR
    args.dynamic_normal_time_dir = DYNAMIC_NORMAL_TIME_DIR
    args.dynamic_features_dir = DYNAMIC_FEATURES_DIR
    args.features_dir = DYNAMIC_FEATURES_DIR
    args.normal_time_dir = DYNAMIC_NORMAL_TIME_DIR
    args.edge_feature_dir = DYNAMIC_EDGE_FEATURE_DIR

    args.node_checkpoint_dir = os.path.join(FINE_TUNE_CHECKPOINTS_DIR, "node")
    args.edge_checkpoint_dir = os.path.join(FINE_TUNE_CHECKPOINTS_DIR, "edge")
    os.makedirs(args.node_checkpoint_dir, exist_ok=True)
    os.makedirs(args.edge_checkpoint_dir, exist_ok=True)

    args.static_data_dir = STATIC_DATA_DIR

    # Set device
    device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device_string)
    
    # Calculate number of domains
    args.num_domains = len(get_pretrain_datasets(args))
    
    # Set random seed
    set_seed(args.seed)
    return args