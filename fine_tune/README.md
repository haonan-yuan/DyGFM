# Model Fine-tuning Module (DyGFM)

## Module Introduction

This module is the **DyGFM** fine-tuning stage: it applies pre-trained models (static or dynamic) to downstream tasks. This module supports two main task types: node classification and edge classification, and provides specialized fine-tuning methods for specific datasets (such as Genre).

## File Structure

- `config.py`: Unified configuration and parameter parsing
- `data_utils.py`: Data processing and loading tools
- `data_processing.py`: Data preprocessing functions
- `fine_tune.py`: General fine-tuning main script
- `fine_tune_node.py`: Node classification task fine-tuning script
- `fine_tune_genre.py`: Genre dataset edge classification specialized fine-tuning script
- `reload_test_node.py`: Node classification model testing script, used to evaluate pre-trained model performance on time-series tasks
- `utils.py`: General utility functions
- `utils_for_genre.py`: Genre dataset specialized utility functions
- `DyGFM_downstream_models/`: Model-related files
  - `classifier.py`: Classifier models
  - `moe.py`: Mixture of Experts model
  - `prompt_gen.py`: Prompt generator
  - `static_model.py`: Static model adapter
  - `tgat.py`: Temporal Graph Attention Network
  - `timer.py`: Time encoder
  - `utils.py`: Model utility functions

## Key Features

- Support for fine-tuning both static and dynamic pre-trained models
- Provides two task types: node classification and edge classification
- Mixture of Experts (MoE) model integrates advantages of different pre-trained models
- Prompt generator enhances model generalization capability
- Specialized optimization for the Genre dataset

## Parameter Configuration

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | str | "genre" | Target downstream dataset name |
| `--seed` | int | 42 | Random seed |
| `--gpu` | int | 0 | GPU ID to use |

### Directory Path Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--base_dir` | str | ".." | Project base directory |
| `--fine_tuning_dir` | str | None | Fine-tuning directory |
| `--dynamic_dir` | str | None | Dynamic model directory |
| `--static_dir` | str | None | Static model directory |

### Model Dimension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--node_feat_dim` | int | 172 | Original node feature dimension |
| `--edge_feat_dim` | int | 172 | Original edge feature dimension |
| `--time_dim` | int | 172 | Time encoding dimension |
| `--unify_dim` | int | 64 | Unified node embedding dimension |
| `--token_dim` | int | 64 | Token dimension |
| `--condition_dim` | int | 65 | Condition dimension |
| `--projection_input_dim` | int | 300 | Projection input dimension |
| `--projection_output_dim` | int | 172 | Projection output dimension |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_neighbors` | int | 10 | Number of neighbors to sample |
| `--epochs` | int | 10000 | Number of training epochs |
| `--batch_size` | int | 1000 | Batch size |
| `--patience` | int | 100 | Early stopping patience value |
| `--eval_steps` | int | 10 | Evaluation interval |

### Learning Rate and Regularization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--projection_lr` | float | 1e-3 | Projection learning rate |
| `--timer_lr` | float | 1e-3 | Time encoder learning rate |
| `--moe_lr` | float | 1e-3 | Mixture of Experts model learning rate |
| `--prompt_generator_lr` | float | 1e-3 | Prompt generator learning rate |
| `--classifier_edge_lr` | float | 1e-5 | Edge classifier learning rate |
| `--proj_lr` | float | 1e-5 | Projection learning rate |

## Example Execution Commands

### General Fine-tuning

```bash
python fine_tune.py --dataset genre --gpu 0 --epochs 1000 --batch_size 2000 --patience 50
```

### Node Classification Task Fine-tuning

```bash
python fine_tune_node.py --dataset mooc --gpu 0 --epochs 500 --batch_size 1000 --patience 30
```

### Node Classification Model Testing

```bash
python reload_test_node.py --dataset mooc --gpu 0 --task_num 5 --different_new_nodes 0
```

### Genre Dataset Edge Classification Fine-tuning

```bash
python fine_tune_genre.py --gpu 0 --epochs 300 --batch_size 1500 --patience 20
```

### Fine-tuning with Custom Parameters

```bash
python fine_tune.py --dataset reddit --gpu 0 --node_feat_dim 256 --edge_feat_dim 256 --time_dim 256 --unify_dim 128 --batch_size 3000 --moe_lr 5e-4 --projection_lr 5e-4
```

## Fine-tuning Process

1. **Data Preparation**:
   - Load target dataset
   - Load pre-trained models (static and/or dynamic)
   - Prepare training, validation, and test data

2. **Model Initialization**:
   - Initialize classifier
   - Initialize Mixture of Experts model (MoE)
   - Initialize prompt generator (if needed)

3. **Training Process**:
   - Batch training
   - Regular evaluation
   - Early stopping mechanism
   - Save best model

4. **Evaluation**:
   - Evaluate performance on test set
   - Calculate relevant metrics (accuracy, AUC, AP, etc.)

## Output

After fine-tuning is complete, the model will be saved in the specified directory with the filename format:
- General fine-tuning: `{dataset}_fine_tuned_{timestamp}.pt`
- Node classification: `{dataset}_node_fine_tuned_{timestamp}.pt`
- Genre edge classification: `genre_edge_fine_tuned_{timestamp}.pt`