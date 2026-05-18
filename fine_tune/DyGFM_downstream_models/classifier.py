import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(torch.nn.Module):
    """A simple multilayer perceptron classifier."""
    def __init__(self, input_dim=172, hidden_dim=128, output_dim=1, dropout_prob=0.5):
        super(MLPClassifier, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class EnhancedMLPClassifier(torch.nn.Module):
    """Enhanced multilayer perceptron classifier."""
    def __init__(self, input_dim=172, hidden_dim=256, output_dim=1, dropout_prob=0.3):
        super(EnhancedMLPClassifier, self).__init__()
        self.network = torch.nn.Sequential(
            # First layer
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=dropout_prob),
            # Second layer
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=dropout_prob),
            # Third layer
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4),
            torch.nn.BatchNorm1d(hidden_dim // 4),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=dropout_prob),
            # Output layer
            torch.nn.Linear(hidden_dim // 4, output_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

class EdgeClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        """
        embed_dim: Dimension of a single node embedding.
        hidden_dim: Dimension of the MLP hidden layer.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output raw logits
        )

    def forward(self, node_emb_u, node_emb_v):
        """
        node_emb_u, node_emb_v: [batch_size, embed_dim]
        Returns: Raw logits of shape [batch_size].
        """
        x = torch.cat([node_emb_u, node_emb_v], dim=-1)  # Concatenate
        logits = self.mlp(x).squeeze(-1)
        return logits

class Proj(nn.Module):
    def __init__(self, embed_dim=172, hidden_dim=128, output_dim=64):
        """
        embed_dim: Input embedding dimension, default 172.
        hidden_dim: Intermediate hidden layer dimension, default 128.
        output_dim: Output embedding dimension, default 64.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: [batch_size, embed_dim]
        Returns: [batch_size, output_dim]
        """
        return self.mlp(x)

class EdgeClassifierV3(nn.Module):
    def __init__(self, embed_dim, hidden_dim1=256, hidden_dim2=128, dropout_p=0.5):
        """
        embed_dim: Dimension of a single node embedding.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            # Input layer
            nn.Linear(embed_dim * 4, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),  # BatchNorm
            nn.ReLU(),
            nn.Dropout(p=dropout_p),      # Dropout
            # Hidden layer
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            # Output layer
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, node_emb_u, node_emb_v):
        hadamard_prod = node_emb_u * node_emb_v
        abs_diff = torch.abs(node_emb_u - node_emb_v)
        all_features = torch.cat([node_emb_u, node_emb_v, hadamard_prod, abs_diff], dim=-1)
        logits = self.mlp(all_features).squeeze(-1)
        return logits
