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
from DyGFM_dynamic_models.utils import *
import swanlab

def train_epoch(model, optimizer, data, node_features, edge_features, ngh_finder, domain_id, batch_size, num_neighbors, neg_sampler, device):
    model.train()
    sources, destinations, timestamps = data['sources'].numpy(), data['destinations'].numpy(), data['timestamps'].numpy()
    n_samples = len(sources)
    effective_batch_size = min(n_samples, batch_size)
    rand_idx = np.random.choice(n_samples, size=effective_batch_size, replace=False)
    src_batch, dst_batch, ts_batch = sources[rand_idx], destinations[rand_idx], timestamps[rand_idx]
    size = len(src_batch)
    _, neg_dst_batch = neg_sampler.sample(size)
    pos_score = model(src_batch, dst_batch, ts_batch, ngh_finder, domain_id, 
                      node_features, edge_features, num_neighbors)
    neg_score = model(src_batch, neg_dst_batch, ts_batch, ngh_finder, domain_id, 
                      node_features, edge_features, num_neighbors)
    criterion = BCEWithLogitsLoss()
    loss = criterion(torch.ones_like(pos_score), pos_score) + criterion(torch.zeros_like(neg_score), neg_score)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, node_features, edge_features, ngh_finder, domain_id, batch_size, num_neighbors, neg_sampler, device):
    model.eval()
    criterion = BCEWithLogitsLoss()
    total_loss = 0
    with torch.no_grad():
        sources, destinations, timestamps = data['sources'].numpy(), data['destinations'].numpy(), data['timestamps'].numpy()
        n_samples = len(sources)
        n_batches = (n_samples - 1) // batch_size + 1
        y_pred, y_true = [], []
        for i in range(n_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            src_batch, dst_batch, ts_batch = sources[start:end], destinations[start:end], timestamps[start:end]
            size = len(src_batch)
            if size == 0: continue
            _, neg_dst_batch = neg_sampler.sample(size)
            pos_score = model(src_batch, dst_batch, ts_batch, ngh_finder, domain_id, 
                              node_features, edge_features, num_neighbors)
            neg_score = model(src_batch, neg_dst_batch, ts_batch, ngh_finder, domain_id, 
                              node_features, edge_features, num_neighbors)
            batch_loss = criterion(torch.ones_like(pos_score), pos_score) + criterion(torch.zeros_like(neg_score), neg_score)
            total_loss += batch_loss.item() * size
            y_pred.extend(pos_score.sigmoid().cpu().numpy())
            y_pred.extend(neg_score.sigmoid().cpu().numpy())
            y_true.extend(np.ones(len(pos_score)))
            y_true.extend(np.zeros(len(neg_score)))
        avg_loss = total_loss / n_samples if n_samples > 0 else 0
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        return avg_loss, auc, ap


def main():
    args = get_args()
    print(args)
    swanlab.init(project="iclr_pretrain", workspace="aboutime", config=vars(args))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    
    # --- 1. Data loading ---
    pretrain_dataset_names = get_pretrain_datasets(args)
    print(f"Pretraining datasets: {pretrain_dataset_names}")
    datasets = []
    for i, dataset_name in enumerate(pretrain_dataset_names):
        print(f"Loading dataset: {dataset_name}")
        node_features, edge_features, _, train_data, val_data, test_data, adj_list = load_dataset(dataset_name, args)
        ngh_finder = NeighborFinder(adj_list, uniform=False)
        train_neg_sampler = RandEdgeSampler(train_data['sources'].numpy(), train_data['destinations'].numpy())
        val_neg_sampler = RandEdgeSampler(val_data['sources'].numpy(), val_data['destinations'].numpy(), seed=0)
        test_neg_sampler = RandEdgeSampler(test_data['sources'].numpy(), test_data['destinations'].numpy(), seed=1)
        datasets.append({
            'name': dataset_name, 'node_features': node_features.float().to(device), 'edge_features': edge_features.float().to(device),
            'train_data': train_data, 'val_data': val_data, 'test_data': test_data, 'ngh_finder': ngh_finder,
            'neg_sampler': train_neg_sampler, 'val_neg_sampler': val_neg_sampler, 'test_neg_sampler': test_neg_sampler, 'domain_id': i
        })

    # --- 2. Model and optimizer initialization ---
    print("Creating decoupled TGAT model")
    model = TGAT(
        node_feat_dim=args.node_feat_dim, edge_feat_dim=args.edge_feat_dim,
        time_dim=args.time_dim, embedding_dim=args.unify_dim,
        num_layers=args.num_layers, n_head=args.num_heads, drop_out=args.dropout,
        attn_mode=args.attn_mode, num_domains=len(datasets)
    ).to(device)
    
    if args.freeze_adapter:
        print("Freezing Adapter parameters, training only backbone network in phase 1...")
        for name, param in model.named_parameters():
            if 'domain_adapters' in name:
                param.requires_grad = False
    else:
        print("Training all parameters (backbone + Adapters) in phase 1...")

    print("Creating optimizer for trainable parameters...")
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                     lr=args.lr, weight_decay=args.l2_coef)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 1 trainable parameter count: {trainable_params}")
    
    checkpoints_dir = os.path.join(args.base_dir, "checkpoints")
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    
    # --- 3. Initialize global best model record ---
    overall_best_val_loss = float('inf')
    overall_best_model_path = os.path.join(checkpoints_dir, f'overall_best_model_{args.dataset}.pt')

    # --- 4. Alternating training loop ---
    print("Starting alternating training")
    for cycle in range(args.alternating_cycles):
        print(f"Alternating training cycle {cycle + 1}/{args.alternating_cycles}")
        for dataset in datasets:
            domain_id, node_features, edge_features = dataset['domain_id'], dataset['node_features'], dataset['edge_features']
            print(f"--- Training domain {domain_id}: {dataset['name']} ---")
            
            early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
            temp_best_model_path = os.path.join(checkpoints_dir, f'temp_best_{dataset["name"]}_{args.dataset}.pt')
            has_saved_temp_best = False

            for epoch in range(args.epochs_per_domain):
                train_loss = train_epoch(
                    model, optimizer, dataset['train_data'], node_features, edge_features,
                    dataset['ngh_finder'], domain_id, args.batch_size, args.num_neighbors,
                    dataset['neg_sampler'], device
                )
                if (epoch + 1) % args.eval_steps == 0:
                    val_loss, val_auc, val_ap = evaluate(
                        model, dataset['val_data'], node_features, edge_features,
                        dataset['ngh_finder'], domain_id, args.batch_size,
                        args.num_neighbors, dataset['val_neg_sampler'], device
                    )
                    print(f"Cycle {cycle+1}, Domain {domain_id}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")
                    if val_loss < early_stopper.best_value:
                        torch.save(model.state_dict(), temp_best_model_path)
                        has_saved_temp_best = True
                        print(f"  (Saved new domain-best model, Val Loss: {val_loss:.4f})")
                    if early_stopper.early_stop_check(val_loss, epoch + 1):
                        print(f"  (Early stopping triggered)")
                        break
                    if val_loss < overall_best_val_loss:
                        overall_best_val_loss = val_loss
                        torch.save(model.state_dict(), overall_best_model_path)
                        print(f"  ****** New global best model! Val Loss: {val_loss:.4f} ******")

            if has_saved_temp_best:
                print(f"Domain {domain_id} training complete. Reverting to domain-best state")
                model.load_state_dict(torch.load(temp_best_model_path, map_location=device))
            else:
                print(f"Domain {domain_id} training complete. (No improvement, no reversion)")
    
    # --- 5. Final model saving and testing ---
    print("\n--- Training complete ---")
    final_model_path = args.save_path.replace('.pt', '_final.pt')
    torch.save({'model_state_dict': model.state_dict(), 'args': args}, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    if os.path.exists(overall_best_model_path):
        print(f"Global best model saved to: {overall_best_model_path}")
        print("\n--- Starting final testing (using global best model) ---")
        model.load_state_dict(torch.load(overall_best_model_path, map_location=device))
        for dataset in datasets:
            node_features, edge_features = dataset['node_features'], dataset['edge_features']
            print(f"\n--- Testing on dataset '{dataset['name']}' ---")
            final_loss, final_auc, final_ap = evaluate(
                model, dataset['test_data'], node_features, edge_features,
                dataset['ngh_finder'], dataset['domain_id'], args.batch_size,
                args.num_neighbors, dataset['test_neg_sampler'], device
            )
            print(f"  Test Loss: {final_loss:.4f}, AUC: {final_auc:.4f}, AP: {final_ap:.4f}")
    else:
        print("Global best model not found, skipping final testing.")

if __name__ == "__main__":
    main()