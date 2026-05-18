import torch
import torch.nn as nn
import torch.nn.functional as F
from DyGFM_downstream_models.utils import fit_gaussian, kl_gaussian, calculate_l2_distance

# All input combinations are handled within this forward method
class MoE(nn.Module):
    """
    A simple version of Mixture of Experts (MoE), containing only num_domains learnable weights.

    Args:
        num_domains (int): The number of domains, which determines the number of weights.

    Experts:
        - Semantic experts: For each domain, node features of length 64 for each node.
        - Temporal experts: Temporal embeddings obtained by applying a timer and adapter.

    Outputs:
        - A weighted combination of the input experts.
        - A loss value.
    """
    def __init__(self, num_domains, branch_weight_static):
        super(MoE, self).__init__()
        self.num_domains = num_domains
        # Initialize learnable weights
        self.weights = nn.Parameter(torch.ones(num_domains))

    def forward(self, semantic_tokens, temporal_tokens, time_tar_token, static_semantic_token, args):
        # Fit Gaussian distributions for source and target domains respectively.
        means = []
        covs = []

        alpha = F.softmax(self.weights, dim=0)
        # Add a small epsilon to prevent log(0) resulting in NaN
        eps = 1e-9
        entropy_loss = -torch.sum(alpha * torch.log(alpha + eps))

        for datasets in semantic_tokens:
            semantic_token = semantic_tokens[datasets]
            mean, cov = fit_gaussian(semantic_token)
            means.append(mean)
            covs.append(cov)

        tar_mean, tar_cov = fit_gaussian(static_semantic_token)

        kls_static = []
        for i in range(len(means)):
            kl = kl_gaussian(means[i], covs[i], tar_mean, tar_cov)
            kls_static.append(kl)
        
        kls_dynamic = []
        for datasets in temporal_tokens:
            temporal_token = temporal_tokens[datasets]
            kl = calculate_l2_distance(temporal_token, time_tar_token)
            kls_dynamic.append(kl)

        # Weighted sum of static and dynamic KL divergence using the weights
        kl_dynamic_weighted = torch.sum(torch.stack(kls_dynamic) * alpha)
        kl_static_weighted = torch.sum(torch.stack(kls_static) * alpha)
        
        divergence_loss = kl_dynamic_weighted * (1 - args.branch_weight_static) + kl_static_weighted * args.branch_weight_static
        total_loss = divergence_loss * args.kl_weight + args.entropy_weight * entropy_loss

        # Also, output a weighted sum of semantic and temporal tokens
        semantic_tokens_list = []
        for dataset in semantic_tokens:
            semantic_token_avg = semantic_tokens[dataset].mean(dim=0)
            semantic_tokens_list.append(semantic_token_avg)

        dynamic_tokens_list = []
        for dataset in temporal_tokens:
            dynamic_tokens_list.append(temporal_tokens[dataset])
        
        # Convert lists to tensors for broadcast multiplication
        semantic_tokens_tensor = torch.stack(semantic_tokens_list) # [num_domains, feat_dim]
        dynamic_tokens_tensor = torch.stack(dynamic_tokens_list)   # [num_domains, feat_dim]
        
        # Perform weighted sum using alpha weights
        weighted_semantic_token = (semantic_tokens_tensor * alpha.view(-1, 1)).sum(dim=0)
        weighted_dynamic_token = (dynamic_tokens_tensor * alpha.view(-1, 1)).sum(dim=0)
        
        return total_loss, divergence_loss, weighted_semantic_token, weighted_dynamic_token
