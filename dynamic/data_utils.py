import torch
import numpy as np
from config import *

def create_adj_list(data):
    """
    Create adjacency list
    
    Args:
        data: Data object
        
    Returns:
        Adjacency list
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
    Split dataset by time order
    
    Args:
        data: Data object
        val_ratio: Time ratio for validation set start (default 0.7, i.e., first 70% for training)
        test_ratio: Time ratio for test set start (default 0.8, i.e., 70%-80% for validation, 80%-100% for testing)
        
    Returns:
        train_data: Training set
        val_data: Validation set
        test_data: Test set
    """
    sources = data['sources']
    destinations = data['destinations']
    timestamps = data['timestamps']
    edge_idxs = data['edge_idxs']
    
    # Get timestamp quantiles
    timestamps_np = timestamps.numpy()
    val_time = np.quantile(timestamps_np, val_ratio)
    test_time = np.quantile(timestamps_np, test_ratio)
    
    # Create train, validation and test masks
    train_mask = timestamps <= val_time
    val_mask = (timestamps > val_time) & (timestamps <= test_time)
    test_mask = timestamps > test_time
    
    # Split dataset
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
    print(f"Training samples: {len(train_data['sources'])}, ratio: {len(train_data['sources'])/len(sources):.2f}")
    print(f"Validation samples: {len(val_data['sources'])}, ratio: {len(val_data['sources'])/len(sources):.2f}")
    print(f"Test samples: {len(test_data['sources'])}, ratio: {len(test_data['sources'])/len(sources):.2f}")
    
    return train_data, val_data, test_data

def load_dataset(dataset_name, args):
    """
    Load dataset and split by time order
    
    Args:
        dataset_name: Dataset name
        args: Arguments
        
    Returns:
        node_features: Node features
        edge_features: Edge features
        full_data: Complete data
        train_data: Training set
        val_data: Validation set
        test_data: Test set
        adj_list: Adjacency list
    """
    data = torch.load(f"{args.features_dir}/{dataset_name}.pt")
    original_node_features = data['x']

    # Load dynamic graph data
    full_data = torch.load(f"{args.normal_time_dir}/{dataset_name}.pt")
    
    # Create edge features (if not available, use zero features)
    edge_features = torch.load(f"{args.data_dir}/edge_feature/{dataset_name}.pt")
    # Convert to torch.tensor
    edge_features = torch.tensor(edge_features)

    zero_row = torch.zeros((1, original_node_features.shape[1]))
    corrected_node_features = torch.cat([zero_row, original_node_features], dim=0)
    # Check shape, should be [N+1, D]
    print(f"\n--- Dynamic data fix for '{dataset_name}' ---")
    print(f"  - Original feature matrix shape: {original_node_features.shape}")
    print(f"  - Corrected feature matrix shape: {corrected_node_features.shape}")
    print("-------------------------------------------\n")

    # Use corrected node_features for subsequent operations
    node_features = corrected_node_features

    # Split dataset by time order
    train_data, val_data, test_data = split_data_by_time(full_data)
    
    # Create adjacency list (using complete dataset)
    adj_list = create_adj_list(full_data)
    
    return node_features, edge_features, full_data, train_data, val_data, test_data, adj_list

if __name__ == "__main__":
    args = get_args()
    node_features, edge_features, full_data, train_data, val_data, test_data, adj_list = load_dataset("lastfm", args)
    
    # Print node_features, edge_features, dynamic_data, adj_list
    print(adj_list[0])