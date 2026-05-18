# Node embedding generation
import torch
import numpy as np
import os
from config import *
from data_utils import edge_index_to_sparse, normalize_adj, sparse_mx_to_torch_sparse_tensor
from DyGFM_static_models import JointContrastiveModel

def generate_embeddings(args, model_path):
    """
    Generate node embeddings using pretrained model
    
    Args:
        args: Configuration arguments
        model_path: Path to the pretrained model
    """
    pre_datasets = get_pretrain_datasets(args)
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    

    for dataset in pre_datasets:
        # Load target dataset
        print(f"Loading dataset: {dataset}")
        data = torch.load(f"{args.data_dir}/{dataset}.pt")
    
        # Prepare model inputs
        num_nodes = data.num_nodes
        sparse_adj = edge_index_to_sparse(data.edge_index, num_nodes=num_nodes)
        features = data.x
    
        # Convert to PyTorch tensors
        if isinstance(features, torch.Tensor):
            features_tensor = features.clone().detach().to(device).float()
        else:
            features_tensor = torch.tensor(features, dtype=torch.float).to(device)
        adj_tensor = sparse_mx_to_torch_sparse_tensor(normalize_adj(sparse_adj)).to(device)
        
        # Wrap features and adjacency matrix into lists (expected input format for model)
        features_list = [features_tensor]
        adj_list = [adj_tensor]
    
        # Initialize model
        in_channels = features.shape[1]
        model = JointContrastiveModel(
            in_channels=in_channels,
            hidden_channels=args.hid_units,
            out_channels=args.out_channels,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        
        # Load pretrained weights
        print(f"Loading pretrained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Generate embeddings
        print("Generating node embeddings...")
        with torch.no_grad():
            embeddings = model.get_embeddings(features_list, adj_list)
        
        print(f"Embedding shape: {embeddings.shape}")
        
        # Save embeddings
        save_dir = os.path.join(args.data_dir, f"embeddings/{args.dataset}")
        os.makedirs(save_dir, exist_ok=True)
        embedding_path = os.path.join(save_dir, f"{dataset}_embeddings.pt")
        torch.save(embeddings, embedding_path)
        print(f"Node embeddings saved to: {embedding_path}")

    return 

if __name__ == "__main__":
    args = get_args()
    # Set model path based on command line arguments
    parser = argparse.ArgumentParser("generate_embeddings")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the pretrained model checkpoint")
    embedding_args = parser.parse_args()
    
    generate_embeddings(args, embedding_args.model_path)
    print("Node embedding generation complete!")