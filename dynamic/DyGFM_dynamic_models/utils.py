import torch
import numpy as np
import time
import os
from pathlib import Path
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score, average_precision_score

from DyGFM_dynamic_models.tgat import *
from config import *
from data_utils import *

class EarlyStopMonitor:
    """
    Early stopping monitor: Used to monitor model performance and stop training when performance no longer improves
    """
    def __init__(self, max_round=5, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.best_epoch = -1
        self.best_value = -float('inf') if higher_better else float('inf')

    def early_stop_check(self, curr_value, epoch):
        """
        Check if early stopping should be triggered
        
        Args:
            curr_value: Current performance value
            epoch: Current epoch number
            
        Returns:
            Whether training should be stopped
        """
        is_better = (self.higher_better and curr_value > self.best_value + self.tolerance) or \
                    (not self.higher_better and curr_value < self.best_value - self.tolerance)
        
        if is_better:
            self.best_value = curr_value
            self.num_round = 0
            self.best_epoch = epoch
            return False
        else:
            self.num_round += 1
            return self.num_round >= self.max_round

class RandEdgeSampler:
    """
    Random edge sampler: Used for negative sampling
    """
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = seed
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
    
    def sample(self, size):
        """
        Sample negative examples
        
        Args:
            size: Sample size
            
        Returns:
            Source nodes, destination nodes
        """
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        
        return self.src_list[src_index], self.dst_list[dst_index]