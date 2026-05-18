# Load all datasets, default to selecting one dataset and loading all remaining datasets
import torch
import torch.optim as optim
import numpy as np
import time
import os
from config import *
from data_utils import *
from DyGFM_static_models import JointContrastiveModel

def train(model, features, adj, neg_samples, optimizer, args):
    """
    Train the model for one epoch
    
    Args:
        model: The model to train
        features: Node features
        adj: Adjacency matrix
        neg_samples: Negative samples for contrastive learning
        optimizer: Optimizer
        args: Arguments
        
    Returns:
        dict: Loss values
    """
    model.train()
    
    # Convert features and adjacency matrix to PyTorch tensors and move to specified device
    features_tensor = torch.tensor(features, dtype=torch.float).to(args.device)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj)).to(args.device)
    neg_samples_tensor = torch.tensor(neg_samples, dtype=torch.long).to(args.device)
    
    # Wrap features and adjacency matrix into lists (expected input format for model)
    features_list = [features_tensor]
    adj_list = [adj_tensor]
    
    # Forward pass
    optimizer.zero_grad()
    contrastive_loss, kl_loss = model(features_list, adj_list, neg_samples_tensor)
    
    # Calculate total loss
    total_loss = contrastive_loss + args.kl_weight * kl_loss
    
    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()
    
    return {
        'contrastive_loss': contrastive_loss.item(),
        'kl_loss': kl_loss.item(),
        'total_loss': total_loss.item()
    }

def main(args):
    # Set device
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {args.device}")
    
    datasets = []
    datasets_names = get_pretrain_datasets(args)
    # If cache mechanism is used and cache files exist
    if args.use_cache and os.path.exists(args.all_adj_path) and os.path.exists(args.all_features_path):
        adj = torch.load(args.all_adj_path)
        features = torch.load(args.all_features_path)
    else:
        for dataset in datasets_names:
            datasets.append(torch.load(f"{args.data_dir}/{dataset}.pt"))
        
        # Convert dataset edge_index to scipy.sparse format
        sparse_adjs = []
        features_list = []
        
        for data in datasets:
            # Get number of nodes and features
            num_nodes = data.num_nodes
            # Convert edge_index to scipy.sparse format (without normalization)
            sparse_adj = edge_index_to_sparse(data.edge_index, num_nodes=num_nodes)
            sparse_adjs.append(sparse_adj)
            # Collect node features
            features_list.append(data.x)
        
        # Now sparse_adjs contains sparse adjacency matrices for all datasets
        print(f"Loaded and converted adjacency matrices for {len(sparse_adjs)} datasets to scipy.sparse format")
        
        # Combine adjacency matrices and feature matrices
        adj = combine_dataset(*sparse_adjs)
        features = combine_features(*features_list)
        
        # Save cache
        if args.use_cache:
            torch.save(adj, args.all_adj_path)
            torch.save(features, args.all_features_path)
    
    print(f"all_adj.shape: {adj.shape}")
    print(f"features.shape: {features.shape}")
    
    # Perform negative sampling
    neg_samples = prompt_pretrain_sample(adj, args.neg_samples)
    print(f"neg_samples.shape: {neg_samples.shape}")
    
    # If training epochs is 0, exit
    if args.nb_epochs <= 0:
        print("Training epochs set to 0, skipping training process")
        return
    
    # Initialize model
    in_channels = features.shape[1]  # Input feature dimension
    model = JointContrastiveModel(
        in_channels=in_channels,
        hidden_channels=args.hid_units,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    print(f"Model initialized, input dimension: {in_channels}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(args.nb_epochs):
        # Train one epoch
        loss_dict = train(model, features, adj, neg_samples, optimizer, args)
        train_loss = loss_dict['total_loss']
        # Early stopping based on training loss
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), args.save_path)
            if (epoch + 1) % args.eval_steps == 0:
                print(f"Epoch {epoch+1}: Saved best model, loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break
    
    total_time = time.time() - start_time
    print(f"Training complete, total time: {total_time:.2f} seconds, best loss: {best_loss:.4f}")
    print(f"Best model saved to: {args.save_path}")
    
    # Load best model for subsequent applications
    model.load_state_dict(torch.load(args.save_path))
    model.eval()

if __name__ == '__main__':
    args = get_args()
    main(args)