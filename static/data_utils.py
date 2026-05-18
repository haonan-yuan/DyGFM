import torch
import numpy as np
import scipy.sparse as sp

def edge_index_to_sparse(edge_index, num_nodes=None, edge_weight=None):
    """
    Convert PyTorch Geometric format edge_index to scipy.sparse.csr_matrix
    
    Args:
        edge_index: torch.Tensor, shape [2, num_edges] edge indices
        num_nodes: int, optional, number of nodes in the graph. If None, inferred from edge_index
        edge_weight: torch.Tensor, optional, shape [num_edges] edge weights. If None, all edge weights set to 1
        
    Returns:
        scipy.sparse.csr_matrix: sparse adjacency matrix
    """
    # Ensure edge_index is a numpy array on CPU
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    
    # Get source and target nodes
    row, col = edge_index
    
    # If node count not provided, infer from edge_index
    if num_nodes is None:
        num_nodes = max(edge_index.max() + 1, edge_index.shape[1])
    
    # Handle edge weights
    if edge_weight is None:
        # If no edge weights provided, set all to 1
        data = np.ones(edge_index.shape[1])
    else:
        # Ensure edge_weight is numpy array
        if isinstance(edge_weight, torch.Tensor):
            data = edge_weight.cpu().numpy()
        else:
            data = edge_weight
    
    # Create sparse matrix (CSR format)
    adj_matrix = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    return adj_matrix

def sparse_to_edge_index(sparse_matrix):
    """
    Convert scipy.sparse matrix back to PyTorch Geometric format edge_index
    
    Args:
        sparse_matrix: scipy.sparse matrix, sparse adjacency matrix
        
    Returns:
        edge_index: torch.Tensor, shape [2, num_edges] edge indices
        edge_weight: torch.Tensor, shape [num_edges] edge weights
    """
    # Ensure matrix is in COO format to extract row and column indices
    coo = sparse_matrix.tocoo()
    
    # Extract row, column indices and values
    indices = np.vstack((coo.row, coo.col))
    edge_index = torch.tensor(indices, dtype=torch.long)
    edge_weight = torch.tensor(coo.data, dtype=torch.float)
    
    return edge_index, edge_weight

def normalize_adj(adj):
    """
    Calculate normalized adjacency matrix: A_hat = D^(-1/2) * A * D^(-1/2)
    
    Args:
        adj: scipy.sparse.csr_matrix, adjacency matrix
        
    Returns:
        scipy.sparse.csr_matrix: normalized adjacency matrix
    """
    # Add self-loops
    adj = adj + sp.eye(adj.shape[0])
    
    # Calculate inverse square root of degree matrix
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Calculate normalized adjacency matrix
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert scipy sparse matrix to torch sparse tensor
    
    Args:
        sparse_mx: scipy.sparse matrix
        
    Returns:
        torch.sparse.FloatTensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def combine_dataset(*args):
    """
    Combine multiple adjacency matrices into one large adjacency matrix
    
    Args:
        *args: Multiple scipy.sparse adjacency matrices
        
    Returns:
        scipy.sparse.csr_matrix: Combined adjacency matrix
    """
    for step, adj in enumerate(args):
        if step == 0:
            adj1 = adj.todense()
        else:
            adj2 = adj.todense()
            zeroadj = np.zeros((adj1.shape[0], adj2.shape[0]))
            tmpadj1 = np.column_stack((adj1, zeroadj))
            tmpadj2 = np.column_stack((zeroadj.T, adj2))
            adj1 = np.row_stack((tmpadj1, tmpadj2))
            
    adj = sp.csr_matrix(adj1)
    return adj

def combine_features(*features_list):
    """
    Combine multiple node feature matrices into one large feature matrix
    Ensure consistency with adjacency matrix combination order
    
    Args:
        features_list: List of feature matrices, each can be numpy array or torch tensor
        
    Returns:
        torch.Tensor: Combined feature matrix
    """
    combined_features = []
    
    for features in features_list:
        # If torch tensor, convert to numpy array
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        combined_features.append(features)
    
    # Vertically stack all feature matrices
    combined = np.vstack(combined_features)
    
    # Convert back to torch tensor
    return torch.tensor(combined, dtype=torch.float)

def prompt_pretrain_sample(adj, n):
    """
    Generate samples for pretraining with positive and negative examples
    
    Args:
        adj: Adjacency matrix in scipy.sparse format
        n: Number of negative samples per node
        
    Returns:
        numpy.ndarray: Sample indices for training
    """
    nodenum = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    res = np.zeros((nodenum, 1+n))
    whole = np.array(range(nodenum))
    for i in range(nodenum):
        nonzero_index_i_row = indices[indptr[i]:indptr[i+1]]
        zero_index_i_row = np.setdiff1d(whole, nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row) == 0:
            res[i][0] = i
        else:
            res[i][0] = nonzero_index_i_row[0]
        res[i][1:1+n] = zero_index_i_row[0:n]
    return res.astype(int)