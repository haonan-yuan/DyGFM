import torch
import torch.nn as nn

class PromptGenerator(nn.Module):
    def __init__(self, semantic_dim: int, temporal_dim: int, hidden_dim: int, prompt_dim: int):
        super().__init__()
        self.semantic_mlp = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim)
        )
        self.temporal_mlp = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim)
        )

    def forward(self, semantic_condition: torch.Tensor, temporal_condition: torch.Tensor):
        # Generate prompts through respective MLPs
        p_sem = self.semantic_mlp(semantic_condition)
        p_temp = self.temporal_mlp(temporal_condition)
        return p_sem, p_temp
