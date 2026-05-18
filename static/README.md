# Static Graph Pre-training Module (DyGFM)

This module is the **DyGFM** static-graph pre-training stage. It contains code for pre-training and node embedding generation for static graph data.

## File Structure

- `config.py`: Configuration and parameter parsing
- `data_utils.py`: Data processing and graph operation tools
- `DyGFM_static_models.py`: GCN-based contrastive learning model implementation
- `pretrain.py`: Main script for model pre-training
- `embedding.py`: Script for generating node embeddings using pre-trained models

## Module Introduction

The static graph pre-training module is a pre-training framework based on Graph Neural Networks (GNN), primarily used for processing static graph data. This module generates high-quality node embedding representations through joint contrastive learning across multiple datasets, which can be used for downstream tasks such as node classification and edge classification.

## Key Features

- GCN-based graph encoder architecture
- Support for joint pre-training across multiple datasets
- Unsupervised training using contrastive learning methods
- Introduction of Gaussian encoders to enhance representation capabilities
- Support for KL divergence constraints to maintain consistency of representations across different datasets

## Parameter Configuration

The main configuration parameters are in the `config.py` file and can be set via command line arguments:

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | str | "genre" | Dataset name, options: genre, mooc, reddit, wikipedia, all |
| `--seed` | int | 39 | Random seed for result reproducibility |
| `--gpu` | int | 0 | GPU ID to use |
| `--unify_dim` | int | 64 | Unified feature dimension |

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--hid_units` | int | 256 | GCN hidden layer dimension |
| `--out_channels` | int | 64 | Output embedding dimension |
| `--num_layers` | int | 2 | Number of GCN layers |
| `--dropout` | float | 0.2 | Dropout rate |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--lr` | float | 0.00001 | Learning rate |
| `--l2_coef` | float | 0.0 | Weight decay coefficient |
| `--nb_epochs` | int | 10000 | Number of training epochs |
| `--patience` | int | 200 | Early stopping patience value |
| `--eval_steps` | int | 10 | Evaluation interval steps |
| `--use_cache` | bool | True | Whether to enable caching mechanism |
| `--neg_samples` | int | 50 | Number of negative samples per positive sample |
| `--kl_weight` | float | 0 | KL divergence loss weight |

### Path Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--base_dir` | str | "./" | Base directory path |
| `--data_dir` | str | "./data" | Data directory path |

## Model Architecture

This module uses the following main components:

1. **GCN Layers**: For feature extraction from graph structured data
2. **Gaussian Encoder**: Encodes GCN output as Gaussian distributions to enhance representation capabilities
3. **Contrastive Learning Loss**: Learns node representations through positive and negative sample comparison
4. **KL Divergence Constraint**: Maintains representation consistency during joint training across multiple datasets

## Example Execution Commands

### Single Dataset Pre-training

```bash
python pretrain.py --dataset genre --gpu 0 --lr 0.00001 --nb_epochs 5000 --patience 200
```

### Multi-Dataset Joint Pre-training

```bash
python pretrain.py --dataset all --gpu 0 --lr 0.00001 --nb_epochs 5000 --kl_weight 0.1
```

### Pre-training with Custom Parameters

```bash
python pretrain.py --dataset mooc --gpu 0 --hid_units 512 --out_channels 128 --num_layers 3 --dropout 0.3 --lr 0.0001 --nb_epochs 3000
```

### Generating Node Embeddings

```bash
python embedding.py --dataset genre --model_path ./save_model/genre_2023-01-01_12-00-00.pt --gpu 0
```

## Data Format

The pre-training module expects data to be stored in PyTorch Geometric's Data object format, containing the following attributes:
- `x`: Node feature matrix
- `edge_index`: Edge indices
- `num_nodes`: Number of nodes

## Output

After pre-training is complete, the model will be saved in the `save_model` directory with the filename format `{dataset}_{timestamp}.pt`.
