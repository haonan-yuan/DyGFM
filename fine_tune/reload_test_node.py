import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import math
import time
import pickle
import os
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from config import *
from data_utils import *
from utils import *
from DyGFM_downstream_models.tgat import *
from DyGFM_downstream_models.static_model import *
from DyGFM_downstream_models.timer import *
from DyGFM_downstream_models.moe import *
from DyGFM_downstream_models.prompt_gen import *
from DyGFM_downstream_models.classifier import *

def eval_edge_prediction(model_features, negative_edge_sampler, data, device, batch_size=100, eval_mode='full', shot_size=10, classifier=None):
    if len(data.sources) == 0: return 0.0, 0.5

    # Ensure negative sampling is reproducible for each evaluation
    if hasattr(negative_edge_sampler, 'seed') and negative_edge_sampler.seed is not None:
        negative_edge_sampler.reset_random_state()
        
    aps, aucs = [], []
    with torch.no_grad():
        if eval_mode == 'full':
            # --- Full evaluation logic ---
            num_batch = math.ceil(len(data.sources) / batch_size)
            for i in range(num_batch):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(data.sources))
                
                sources_batch = data.sources[start_idx:end_idx]
                destinations_batch = data.destinations[start_idx:end_idx]
                size = len(sources_batch)
                if size == 0: continue
                
                _, negatives_batch = negative_edge_sampler.sample(size)
                
                pos_prob, neg_prob = compute_edge_probabilities(
                    sources_batch, destinations_batch, negatives_batch, model_features, device, classifier)
                
                y_true = np.concatenate([np.ones(size), np.zeros(size)])
                y_pred = torch.cat([pos_prob, neg_prob]).cpu().numpy()
                
                aps.append(average_precision_score(y_true, y_pred))
                aucs.append(roc_auc_score(y_true, y_pred))

        elif eval_mode == 'few_shot':
            # --- Few-shot evaluation logic ---
            if len(data.sources) < shot_size:
                shot_size = len(data.sources)

            if shot_size > 0:
                indices = np.random.choice(len(data.sources), shot_size, replace=False)
                
                sources_batch = data.sources[indices]
                destinations_batch = data.destinations[indices]
                size = len(sources_batch)

                _, negatives_batch = negative_edge_sampler.sample(size)

                pos_prob, neg_prob = compute_edge_probabilities(
                    sources_batch, destinations_batch, negatives_batch, model_features, device, classifier)
                
                y_true = np.concatenate([np.ones(size), np.zeros(size)])
                y_pred = torch.cat([pos_prob, neg_prob]).cpu().numpy()

                aps.append(average_precision_score(y_true, y_pred))
                aucs.append(roc_auc_score(y_true, y_pred))
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

    return np.mean(aps) if aps else 0.0, np.mean(aucs) if aucs else 0.5

def get_task_time_set(args, full_data):
    label_flag = 0
    task_start_time = 0
    task_end_time = 0

    for i in range(len(full_data.labels)):
        if full_data.labels[i]:
            label_flag += 1
        if label_flag == 20 and task_start_time == 0:
            task_start_time = full_data.timestamps[i]
        if label_flag == 24:
            task_end_time = full_data.timestamps[i]
            break
    
    task_start = full_data.timestamps >= task_start_time
    task_end = full_data.timestamps <= task_end_time
    task_time_p = task_start * task_end
    task_time_pool = full_data.timestamps[task_time_p]
    test_indices = (1 - task_time_p) > 0
    test_indices = (1 - task_start) > 0
    task_time_set = random.sample(set(task_time_pool), args.task_num)
    return task_time_set, test_indices

def split_dataset(task_time_set, full_data, task, test_indices):
    time_stamp = task_time_set[task]
    print(f"Predicting task {task}, time_stamp={time_stamp}")
    ts_flag = (full_data.timestamps <= time_stamp)
    index = np.where(full_data.timestamps == time_stamp)[0][0]
    print(f"Found timestamp index: {index}, total data: {len(full_data.sources)}")
    
    ts_label_flag_1 = (ts_flag) * (full_data.labels)
    ts_label_flag_1 = ts_label_flag_1[0:index+1]
    print(f"Data truncated to timestamp: {len(ts_label_flag_1)}")
    
    # Count initial label distribution
    unique, counts = np.unique(ts_label_flag_1, return_counts=True)
    print(f"Initial label distribution: {dict(zip(unique, counts))}")
    
    record = {}
    filtered_count = 0
    for i in range(len(ts_label_flag_1)-1, -1, -1):
        if full_data.sources[i] in record:
            ts_label_flag_1[i] = -1
            filtered_count += 1
        else:
            record[full_data.sources[i]] = 1
    
    print(f"After filtering duplicate source nodes, removed {filtered_count} entries")
    
    # Count filtered label distribution
    unique, counts = np.unique(ts_label_flag_1, return_counts=True)
    print(f"Filtered label distribution: {dict(zip(unique, counts))}")
    
    # Calculate available positive and negative samples
    positive_samples = len(set(np.where(ts_label_flag_1 == 1)[0]))
    negative_samples = len(set(np.where(ts_label_flag_1 == 0)[0]))
    print(f"Available positive samples: {positive_samples}")
    print(f"Available negative samples: {negative_samples}")
    
    num_indices = 10
    
    # Ensure sampling count doesn't exceed available samples
    train_pos_samples = min(num_indices, positive_samples)
    train_neg_samples = min(num_indices*5, negative_samples)
    
    print(f"Will sample {train_pos_samples} positive and {train_neg_samples} negative samples")
    
    if positive_samples >= num_indices:
        train_indices_1 = random.sample(set(np.where(ts_label_flag_1 == 1)[0]), train_pos_samples)
    else:
        print(f"Warning: Not enough positive samples! Need {num_indices}, have {positive_samples}")
        train_indices_1 = list(set(np.where(ts_label_flag_1 == 1)[0]))
    
    if negative_samples >= num_indices*5:
        train_indices_0 = random.sample(set(np.where((ts_label_flag_1 == 0))[0]), train_neg_samples)
    else:
        print(f"Warning: Not enough negative samples! Need {num_indices*5}, have {negative_samples}")
        train_indices_0 = list(set(np.where((ts_label_flag_1 == 0))[0]))
        
    # Mark used samples
    ts_label_flag_1[train_indices_1], ts_label_flag_1[train_indices_0] = -1, -2
    
    # Count remaining samples available for validation
    remaining_pos = len(set(np.where(ts_label_flag_1 == 1)[0]))
    remaining_neg = len(set(np.where(ts_label_flag_1 == 0)[0]))
    print(f"Remaining positive samples for validation: {remaining_pos}")
    print(f"Remaining negative samples for validation: {remaining_neg}")
    
    # Ensure validation sampling count doesn't exceed available samples
    val_pos_samples = min(num_indices, remaining_pos)
    val_neg_samples = min(num_indices*5, remaining_neg)
    
    print(f"Will sample {val_pos_samples} positive and {val_neg_samples} negative samples for validation")
    
    if remaining_pos >= num_indices:
        val_indices_1 = random.sample(set(np.where(ts_label_flag_1 == 1)[0]), val_pos_samples)
    else:
        print(f"Warning: Not enough validation positive samples! Need {num_indices}, have {remaining_pos}")
        val_indices_1 = list(set(np.where(ts_label_flag_1 == 1)[0]))
    
    if remaining_neg >= num_indices*5:
        val_indices_0 = random.sample(set(np.where((ts_label_flag_1 == 0))[0]), val_neg_samples)
    else:
        print(f"Warning: Not enough validation negative samples! Need {num_indices*5}, have {remaining_neg}")
        val_indices_0 = list(set(np.where((ts_label_flag_1 == 0))[0]))
        
    train_indices = train_indices_1 + train_indices_0
    val_indices = val_indices_1 + val_indices_0

    train_data = Data(full_data.sources[train_indices], full_data.destinations[train_indices], full_data.timestamps[train_indices],
                    full_data.edge_idxs[train_indices], full_data.labels[train_indices])
    val_data = Data(full_data.sources[val_indices], full_data.destinations[val_indices], full_data.timestamps[val_indices],
                    full_data.edge_idxs[val_indices], full_data.labels[val_indices])
    test_data = Data(full_data.sources[test_indices], full_data.destinations[test_indices], full_data.timestamps[test_indices],
                    full_data.edge_idxs[test_indices], full_data.labels[test_indices])

    return train_data, val_data, test_data

def test(args, node_features, test_data, classifier, device):
    classifier.eval()
    with torch.no_grad():
        # Process test set in batches
        test_size = len(test_data.sources)
        batch_size = 1000
        num_test_batches = (test_size + batch_size - 1) // batch_size
        
        all_pred_scores = []
        all_labels = []
        
        for batch_idx in range(num_test_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, test_size)
            
            batch_sources = test_data.sources[start_idx:end_idx]
            batch_labels = test_data.labels[start_idx:end_idx]
            
            # Make predictions on test set
            source_embeddings = node_features[batch_sources]
            batch_labels_torch = torch.from_numpy(batch_labels).float().to(device)
            pred = classifier(source_embeddings).squeeze()
            
            # Collect prediction results
            pred_score = torch.sigmoid(pred).cpu().numpy()
            all_pred_scores.extend(pred_score)
            all_labels.extend(batch_labels)

        # Calculate test metrics
        all_pred_scores = np.array(all_pred_scores)
        all_labels = np.array(all_labels)
        
        # AUC
        test_auc = roc_auc_score(all_labels, all_pred_scores)
        
        return test_auc

def main(args):
    pre_datasets = get_pretrain_datasets(args)
    device = args.device
    info_pth = os.path.join(
        args.fine_tune_dir, "save_model", "node", f"all_{args.dataset}_{args.seed}_0.pt"
    )
    info = torch.load(info_pth, map_location=device)
    _, _, full_data, _, _, _, _, _ = get_d_data(args, args.dataset, args.different_new_nodes)
    
    node_features = info['feat']['x'].to(device)
    task_time_set, test_indices = get_task_time_set(args, full_data)
    all_tasks_results = []

    for task in tqdm(range(args.task_num)):
        _, _, test_data = split_dataset(task_time_set, full_data, task, test_indices)
        classifier = EnhancedMLPClassifier().to(device)
        print("\n--- Final Evaluation on Test Set ---")

        classifier.load_state_dict(info['classifier'])
        test_results = test(args, node_features, test_data, classifier, device)
        print(f"Test AUC: {test_results:.4f}")
        all_tasks_results.append(test_results)

    print(f"All Tasks AUC: {np.mean(all_tasks_results):.4f}")
    print(f"All Tasks AUC Std: {np.std(all_tasks_results):.4f}")

def main_un_all(args):
    pre_datasets = get_pretrain_datasets(args)
    device = args.device
    info_pth = os.path.join(
        args.fine_tune_dir, "save_model", "node", f"{args.dataset}_{args.seed}_0.pt"
    )
    info = torch.load(info_pth, map_location=device)

    _, _, full_data, _, _, _, _, _ = get_d_data(args, args.dataset, args.different_new_nodes)
    
    node_features = info['feat']['x'].to(device)
    print(f"Node feature shape (main_un_all): {node_features.shape}")
    
    task_time_set, test_indices = get_task_time_set(args, full_data)
    all_tasks_results = []

    for task in tqdm(range(args.task_num)):
        _, _, test_data = split_dataset(task_time_set, full_data, task, test_indices)
        classifier = EnhancedMLPClassifier().to(device)
        print("\n--- Final Evaluation on Test Set ---")

        classifier.load_state_dict(info['classifier'])
        test_results = test(args, node_features, test_data, classifier, device)
        print(f"Test AUC: {test_results:.4f}")
        all_tasks_results.append(test_results)

    print(f"All Tasks AUC: {np.mean(all_tasks_results):.4f}")
    print(f"All Tasks AUC Std: {np.std(all_tasks_results):.4f}")

if __name__ == "__main__":
    args = get_args()
    print("Running with arguments:")
    print(args)
    main(args)
    main_un_all(args)
