import torch
import torch.nn as nn
import math

class Timer(nn.Module):
    """
    Timer Model: Single-domain time encoder.
    Encodes timestamps and converts them into temporal embeddings.
    It uses sine/cosine positional encoding (similar to Transformers)
    and then an MLP for dimensional alignment.
    """

    def __init__(self, time_dim, hidden_dim=None):
        """
        Initializes the Timer model.
        
        Args:
            time_dim: The final dimension of the output temporal embedding.
            hidden_dim: The dimension of the hidden layer in the MLP, defaults to 2 * time_dim.
        """
        super(Timer, self).__init__()
        self.time_dim = time_dim
        
        if hidden_dim is None:
            hidden_dim = time_dim * 2
            
        # Initialize MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, time_dim)
        )
        
    def time_encoding(self, timestamps, freq_scale=10000):
        """
        Encodes timestamps using sine/cosine encoding.
        
        Args:
            timestamps: A tensor of timestamps with shape [batch_size], normalized to [0,1].
            freq_scale: The frequency scaling factor.
            
        Returns:
            A temporal encoding tensor of shape [batch_size, time_dim].
        """
        batch_size = timestamps.size(0)
        
        # Expand timestamp dimensions for broadcasting
        timestamps = timestamps.unsqueeze(1)
        
        # Calculate positions at different frequencies
        half_dim = self.time_dim // 2
        positions = torch.arange(0, half_dim, device=timestamps.device).float()
        # Calculate frequency factors
        inv_freq = 1.0 / (freq_scale ** (positions / half_dim))
        
        # Calculate sine and cosine values
        sinusoid_inp = timestamps * inv_freq.unsqueeze(0)
        sin_encodings = torch.sin(sinusoid_inp)
        cos_encodings = torch.cos(sinusoid_inp)
        
        # Interleave sine and cosine encodings
        encodings = torch.zeros(batch_size, self.time_dim, device=timestamps.device)
        encodings[:, 0::2] = sin_encodings
        encodings[:, 1::2] = cos_encodings
        
        return encodings
    
    def forward(self, timestamps):
        """
        Forward propagation function.
        
        Args:
            timestamps: A tensor of timestamps with shape [batch_size], should be normalized to [0,1].
            
        Returns:
            A temporal embedding tensor of shape [batch_size, time_dim].
        """
        # Get time encoding
        time_enc = self.time_encoding(timestamps)
        
        # Process time encoding with MLP
        time_embeddings = self.time_mlp(time_enc)
        
        return time_embeddings
