import torch
import numpy as np
import time
import os
from pathlib import Path
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score, average_precision_score

from DyGFM_downstream_models.tgat import *
from config import *
from data_utils import *

import torch

def calculate_l2_distance(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    """
    Calculates the L2 (Euclidean) distance between two one-dimensional PyTorch tensors.

    Args:
        tensor_a (torch.Tensor): The first input tensor (vector).
        tensor_b (torch.Tensor): The second input tensor (vector), must have the same shape as tensor_a.

    Returns:
        torch.Tensor: A scalar tensor containing the L2 distance.
    """
    assert tensor_a.shape == tensor_b.shape, \
        f"Input tensors must have the same shape, but got {tensor_a.shape} and {tensor_b.shape}"
    distance = torch.linalg.norm(tensor_a - tensor_b, ord=2)
    return distance

def kl_gaussian(mu_p, cov_p, mu_q, cov_q, full_cov=False, eps=1e-8):
    """
    [New Function] Calculates the KL divergence D_KL(P || Q) between two Gaussian distributions P and Q.
    This version is enhanced for numerical stability to effectively prevent NaN values.
    """
    # --- Measure 1: Use double precision for all calculations for higher numerical range and precision ---
    mu_p, cov_p = mu_p.to(torch.float64), cov_p.to(torch.float64)
    mu_q, cov_q = mu_q.to(torch.float64), cov_q.to(torch.float64)
    d = mu_p.shape[0]

    if not full_cov:
        # Ensure inputs are variance vectors
        var_p = cov_p if cov_p.ndim == 1 else torch.diag(cov_p)
        var_q = cov_q if cov_q.ndim == 1 else torch.diag(cov_q)
        
        # --- Measure 2: Clamp all variances to ensure they are strictly positive ---
        var_p = torch.clamp(var_p, min=eps)
        var_q = torch.clamp(var_q, min=eps)
        
        kl = 0.5 * torch.sum(
            torch.log(var_q / var_p)
            + var_p / var_q
            + (mu_q - mu_p).pow(2) / var_q
            - 1
        )
    else: # Full covariance matrix branch
        # Regularize covariance matrices to ensure positive definiteness
        cov_p = cov_p + torch.eye(d, device=cov_p.device, dtype=cov_p.dtype) * eps
        cov_q = cov_q + torch.eye(d, device=cov_q.device, dtype=cov_q.dtype) * eps

        try:
            inv_q = torch.linalg.inv(cov_q)
            diff = mu_q - mu_p
            
            # Use cholesky decomposition for logdet, which is more numerically stable
            logdet_q = 2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(cov_q))))
            logdet_p = 2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(cov_p))))

            term1 = logdet_q - logdet_p
            term2 = torch.trace(inv_q @ cov_p)
            term3 = diff.T @ inv_q @ diff

            kl = 0.5 * (term1 - d + term2 + term3)
        except torch.linalg.LinAlgError:
            # If the matrix is still singular or not positive-definite, return a large value instead of NaN
            print("Warning: Covariance matrix is singular or not positive-definite. Returning a large KL value.")
            kl = torch.tensor(1e9, device=mu_p.device, dtype=torch.float64)
            
    # --- Measure 3: Convert the final result back to standard float32 to be compatible with the rest of the model ---
    return kl.to(torch.float32)

def fit_gaussian(data, full_cov=False, regularize=1e-6, to_numpy=False, dtype=torch.float32, device=None):
    was_numpy = False
    if isinstance(data, np.ndarray):
        was_numpy = True
        device = device or torch.device('cpu')
        data = torch.from_numpy(data).to(device=device, dtype=dtype)
    elif isinstance(data, torch.Tensor):
        if device is not None:
            data = data.to(device=device, dtype=dtype)
        else:
            data = data.to(dtype=dtype)
    else:
        raise TypeError("data must be a numpy.ndarray or torch.Tensor")
    
    if data.ndim != 2:
        raise ValueError("data must be 2D: (N, D)")
    N, D = data.shape
    if N < 2:
        raise ValueError("Need at least 2 samples to estimate covariance")
    
    mean = torch.mean(data, dim=0)
    
    if full_cov:
        X = data - mean.unsqueeze(0)
        cov = (X.t() @ X) / (N - 1)
        cov = cov + regularize * torch.eye(D, device=cov.device, dtype=cov.dtype)
    else:
        var = torch.sum((data - mean.unsqueeze(0))**2, dim=0) / (N - 1)
        cov = var + regularize
    
    if to_numpy:
        mean = mean.cpu().numpy()
        cov = cov.cpu().numpy()
    return mean, cov

class RandEdgeSampler:
    """
    Random Edge Sampler: for negative sampling.
    """
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = seed
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
    
    def sample(self, size):
        """
        Samples negative examples.
        
        Args:
            size: The sample size.
            
        Returns:
            Source nodes, destination nodes.
        """
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        
        return self.src_list[src_index], self.dst_list[dst_index]
