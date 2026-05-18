# Dynamic Graph Pre-training Module (DyGFM)

## Module Introduction

This module is the **DyGFM** dynamic-graph pre-training stage: a pre-training framework based on Temporal Graph Attention Network (TGAT), specifically designed for processing dynamic graph data. This module adopts a two-phase training strategy and implements multi-dataset joint pre-training through a decoupled adapter architecture, providing high-quality node and edge representations for downstream tasks.

## File Structure

- `config.py`: Configuration and parameter parsing for the first phase pre-training
- `config_phase2.py`: Configuration and parameter parsing for the second phase adapter fine-tuning
- `data_utils.py`: Data processing and graph operation tools
- `pretrain.py`: Main script for the first phase pre-training
- `pretrain_phase2.py`: Main script for the second phase adapter fine-tuning
- `DyGFM_dynamic_models/`: Model-related files (TGAT, timer, utils)
  - `tgat.py`: Temporal Graph Attention Network implementation
  - `timer.py`: Time encoder implementation
  - `utils.py`: Model utility functions

## Key Features

- TGAT-based dynamic graph encoder architecture
- Two-phase training strategy:
  - First phase: Train shared backbone network, freeze adapter parameters
  - Second phase: Freeze backbone network, fine-tune adapters for specific datasets
- Decoupled adapter architecture, supporting joint pre-training across multiple datasets
- Time encoder for capturing temporal information
- Negative sampling strategy for optimizing contrastive learning

## Parameter Configuration

### First Phase Parameters (config.py)

#### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | str | "genre" | Main dataset name |
| `--seed` | int | 39 | Random seed |
| `--gpu` | int | 0 | GPU ID to use |

#### Model Dimension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--node_feat_dim` | int | 172 | Original node feature dimension |
| `--edge_feat_dim` | int | 172 | Original edge feature dimension |
| `--time_dim` | int | 172 | Time encoding dimension |
| `--unify_dim` | int | 64 | Unified node embedding dimension |

#### Model Structure Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_layers` | int | 1 | Number of TGAT layers |
| `--num_heads` | int | 2 | Number of attention heads |
| `--dropout` | float | 0.1 | Dropout rate |
| `--attn_mode` | str | "prod" | Attention mode |

#### Training Control Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch_size` | int | 2000 | Batch size |
| `--lr` | float | 0.0001 | Learning rate |
| `--l2_coef` | float | 0.0 | Weight decay coefficient |
| `--patience` | int | 50 | Early stopping patience value |
| `--eval_steps` | int | 10 | Evaluation interval (epochs) |
| `--freeze_adapter` | bool | False | Whether to freeze adapter parameters |
| `--epochs_per_domain` | int | 30 | Training epochs per domain |
| `--alternating_cycles` | int | 2 | Number of alternating training cycles |
| `--num_neighbors` | int | 5 | Number of neighbors to sample |

### Second Phase Parameters (config_phase2.py)

The second phase parameters are mostly the same as the first phase, but with the following key differences:

- The second phase focuses on fine-tuning adapters for specific datasets
- Requires loading the pre-trained model from the first phase
- Freezes backbone network parameters, only training adapters

## Example Execution Commands

### First Phase Pre-training (Freeze Adapters, Train Backbone Network)

```bash
python pretrain.py --dataset genre --gpu 0 --freeze_adapter --epochs_per_domain 30 --alternating_cycles 2
```

### Second Phase Pre-training (Freeze Backbone Network, Fine-tune Adapters)

```bash
python pretrain_phase2.py --dataset genre --gpu 0 --phase1_model_path ./save_model/tgat_genre_YYYY-MM-DD_HH-MM-SS_phase1.pt --epochs 100
```

### Pre-training with Custom Parameters

```bash
python pretrain.py --dataset mooc --gpu 0 --node_feat_dim 256 --edge_feat_dim 256 --time_dim 256 --unify_dim 128 --num_layers 2 --num_heads 4 --batch_size 4000 --lr 0.0005
```

## Training Process

1. **First Phase**:
   - Load multiple datasets
   - Initialize TGAT model and adapters
   - Freeze adapter parameters
   - Alternately train backbone network on different datasets
   - Save model

2. **Second Phase**:
   - Load pre-trained model from the first phase
   - Freeze backbone network parameters
   - Fine-tune adapters for specific datasets
   - Save final model

## Data Format

The dynamic graph pre-training module expects data to be stored in the following format:
- Node feature matrix
- Edge feature matrix
- Timestamp information
- Edge list (source node, target node, timestamp)

## Output

After pre-training is complete, the model will be saved in the `save_model` directory:
- First phase model: `tgat_{dataset}_{timestamp}_phase1.pt`
- Second phase model: `tgat_{dataset}_{timestamp}_phase2.pt`