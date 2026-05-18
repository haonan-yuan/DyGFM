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
from DyGFM_downstream_models.classifier import EdgeClassifierV3
from utils_for_genre import StratifiedEdgeSampler, TemporalStratifiedEdgeSampler

def compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch, model_features, device, edge_classifier):
    """Calculates probabilities for positive and negative edge samples."""
    source_embeds = model_features[sources_batch]
    dest_embeds = model_features[destinations_batch]
    neg_embeds = model_features[negatives_batch]
    pos_scores = edge_classifier(source_embeds, dest_embeds)
    neg_scores = edge_classifier(source_embeds, neg_embeds)
    return torch.sigmoid(pos_scores), torch.sigmoid(neg_scores)

def get_target_dynamic_embeddings(node_features, edge_features, src_idx_l, target_idx_l, cut_time_l, ngh_finder, timer, args, device, model):
    if len(src_idx_l) == 0:
        return torch.zeros((node_features.shape[0], args.unify_dim), device=device)
    batch_size = 1000
    num_edges = len(src_idx_l)
    num_batches = (num_edges + batch_size - 1) // batch_size
    all_src_embeds, all_dst_embeds = [], []
    print(f'node_features.shape{node_features.shape}')
    for batch_idx in range(num_batches):
        start_idx, end_idx = batch_idx * batch_size, min((batch_idx + 1) * batch_size, num_edges)
        batch_sources, batch_destinations, batch_timestamps = src_idx_l[start_idx:end_idx], target_idx_l[start_idx:end_idx], cut_time_l[start_idx:end_idx]
        src_embed, dst_embed = model.embed_without_adapter_with_new_timer(
            batch_sources, batch_destinations, batch_timestamps, ngh_finder, 
            node_features, edge_features, num_neighbors=args.num_neighbors, timer=timer)
        all_src_embeds.append(src_embed)
        all_dst_embeds.append(dst_embed)
    src_embeddings, dst_embeddings = torch.cat(all_src_embeds, dim=0), torch.cat(all_dst_embeds, dim=0)
    num_nodes = node_features.shape[0]
    embedding_dim = src_embeddings.shape[1]
    new_embeddings = torch.zeros((num_nodes, embedding_dim), device=device)
    id_count = torch.zeros(num_nodes, device=device)
    src_idx_l_tensor = torch.tensor(src_idx_l, device=device)
    target_idx_l_tensor = torch.tensor(target_idx_l, device=device)
    new_embeddings.index_add_(0, src_idx_l_tensor, src_embeddings)
    new_embeddings.index_add_(0, target_idx_l_tensor, dst_embeddings)
    id_count.index_add_(0, src_idx_l_tensor, torch.ones_like(src_idx_l_tensor, dtype=torch.float))
    id_count.index_add_(0, target_idx_l_tensor, torch.ones_like(target_idx_l_tensor, dtype=torch.float))
    id_count[id_count == 0] = 1
    new_embeddings /= id_count.unsqueeze(1)
    return new_embeddings

def get_target_dynamic_token(node_features, edge_features, src_idx_l, target_idx_l, cut_time_l, ngh_finder, timer, args, device, model):
    if len(src_idx_l) == 0:
        return torch.zeros(args.unify_dim, device=device)
    avg_embeddings = get_target_dynamic_embeddings(node_features, edge_features, src_idx_l, target_idx_l, cut_time_l, ngh_finder, timer, args, device, model)
    involved_nodes = torch.unique(torch.cat([torch.tensor(src_idx_l, device=device), torch.tensor(target_idx_l, device=device)]))
    time_features = torch.mean(avg_embeddings[involved_nodes], dim=0)
    return time_features

def svd_dim_reduction(tensor, unify_dim):
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
    reduced = torch.matmul(tensor, Vh.T[:, :unify_dim])
    return reduced

def get_target_static_token(node_features, unify_dim, device):
    static_tokens = svd_dim_reduction(node_features, unify_dim)
    return static_tokens

def generate_final_node_features(args, node_features, edge_features,
                                 source_nodes, dest_nodes, timestamps,
                                 ngh_finder, models, tokens, device):
    prompt_generator, timer, moe = models['prompt_generator'], models['timer'], models['moe']
    projection_head, static_model, dynamic_model = models['projection_head'], models['static_model'], models['dynamic_model']
    static_tokens_data, dynamic_tokens_data, static_semantic_token = tokens['static_tokens'], tokens['dynamic_tokens'], tokens['static_semantic_token']
    time_tar_token = get_target_dynamic_token(node_features, edge_features, source_nodes, dest_nodes, timestamps, ngh_finder, timer, args, device, dynamic_model)
    moe_loss, kl, _, _ = moe(static_tokens_data, dynamic_tokens_data, time_tar_token, static_semantic_token, args)
    static_tar_token = static_semantic_token.sum(dim=0) / static_semantic_token.shape[0]
    kl_as_vector_1d = kl.view(1)
    input_semantic = torch.cat([static_tar_token, kl_as_vector_1d], dim=0)
    input_temporal = torch.cat([time_tar_token, kl_as_vector_1d], dim=0)
    semantic_prompt, temporal_prompt = prompt_generator(input_semantic, input_temporal)
    static_tar_token_exp = static_tar_token.unsqueeze(0).repeat(node_features.size(0), 1)
    semantic_prompt_exp = semantic_prompt.unsqueeze(0).repeat(node_features.size(0), 1)
    temporal_prompt_exp = temporal_prompt.unsqueeze(0).repeat(node_features.size(0), 1)
    time_tar_token_exp = time_tar_token.unsqueeze(0).repeat(node_features.size(0), 1)
    static_node_features_input = torch.cat([node_features, static_tar_token_exp, semantic_prompt_exp], dim=1)
    dynamic_node_features_input = torch.cat([node_features, time_tar_token_exp, temporal_prompt_exp], dim=1)
    static_node_features = projection_head(static_node_features_input)
    dynamic_node_features = projection_head(dynamic_node_features_input)
    adj = get_adj_tensor(args, num_nodes=node_features.shape[0], device=device)
    static_node_features_processed = static_model.get_embeddings([static_node_features], [adj])
    static_node_features = torch.cat([torch.zeros(1, static_node_features_processed.shape[1], device=device), static_node_features_processed], dim=0)
    dynamic_node_features = torch.cat([torch.zeros(1, dynamic_node_features.shape[1], device=device), dynamic_node_features], dim=0)
    dynamic_node_features = get_target_dynamic_embeddings(dynamic_node_features, edge_features, source_nodes, dest_nodes, timestamps, ngh_finder, timer, args, device, dynamic_model)
    new_node_features = static_node_features * args.node_feature_static_weight + dynamic_node_features * (1 - args.node_feature_static_weight)
    return new_node_features, moe_loss

def train_one_epoch(args, train_data, train_rand_sampler, train_ngh_finder, models, tokens, optimizers, criterion, device):
    for model in models.values():
        if isinstance(model, torch.nn.Module) and model not in [models['static_model'], models['dynamic_model']]:
            model.train()
    node_features, edge_features = tokens['node_features'], tokens['edge_features']
    shot_size = 30
    if len(train_data.sources) < shot_size:
        print(f"Warning: Not enough training samples ({len(train_data.sources)}) for the requested shot size ({shot_size}). Using all available samples.")
        shot_size = len(train_data.sources)
    if shot_size == 0:
        return 0, 0, 0
    try:
        sources_batch, destinations_batch, timestamps_batch, actual_size = train_pos_sampler.sample(shot_size)
        size = actual_size
        if size == 0:
            print("Warning: No samples could be obtained. Skipping this batch.")
            return 0, 0, 0
    except RuntimeError as e:
        print(f"Sampling error: {str(e)}")
        print("Skipping this batch due to sampling error.")
        return 0, 0, 0
    for optimizer in optimizers: optimizer.zero_grad()
    new_node_features, moe_loss = generate_final_node_features(
        args, node_features, edge_features, sources_batch, destinations_batch, timestamps_batch,
        train_ngh_finder, models, tokens, device)
    try:
        _, negatives_batch, neg_size = train_rand_sampler.sample(size)
    except RuntimeError as e:
        print(f"Negative sampling error: {str(e)}")
        print("Skipping this batch due to negative sampling error.")
        return 0, 0, 0
    size = min(size, neg_size)
    if size == 0:
        print("Warning: No valid samples after ensuring consistency. Skipping this batch.")
        return 0, 0, 0
    sources_batch = sources_batch[:size]
    destinations_batch = destinations_batch[:size]
    negatives_batch = negatives_batch[:size]
    pos_label = torch.ones(size, dtype=torch.float, device=device)
    neg_label = torch.zeros(size, dtype=torch.float, device=device)
    pos_prob, neg_prob = compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch, new_node_features, device, models['edge_classifier'])
    print(f"pos_prob:{pos_prob}")
    print(f"neg_prob:{neg_prob}")
    classification_loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
    loss = moe_loss * args.moe_loss_weight + classification_loss
    loss.backward()
    for optimizer in optimizers: optimizer.step()
    return loss.item(), moe_loss.item(), classification_loss.item()

def evaluate(args, eval_data, negative_sampler, full_ngh_finder, models, tokens, device, eval_type='Validation', pos_sampler=None):
    """Selects evaluation strategy (full or few-shot) based on eval_type."""
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.eval()
    node_features, edge_features = tokens['node_features'], tokens['edge_features']
    with torch.no_grad():
        if eval_type in ['Test', 'Inductive']:
            num_feature_samples = min(len(eval_data.sources), 1000)
            if num_feature_samples == 0: return 0.0, 0.5
            feature_indices = np.random.choice(len(eval_data.sources), num_feature_samples, replace=False)
            source_nodes, dest_nodes, timestamps = eval_data.sources[feature_indices], eval_data.destinations[feature_indices], eval_data.timestamps[feature_indices]
            new_node_features, _ = generate_final_node_features(
                args, node_features, edge_features, source_nodes, dest_nodes, timestamps,
                full_ngh_finder, models, tokens, device)
            ap, auc = eval_edge_prediction(
                model_features=new_node_features, 
                negative_edge_sampler=negative_sampler,
                data=eval_data, 
                batch_size=args.val_batch_size, 
                device=device,
                eval_mode='full'
            )
        else:
            shot_size = 30
            if len(eval_data.sources) < shot_size:
                shot_size = len(eval_data.sources)
            if shot_size == 0: return 0.0, 0.5
            eval_indices = np.random.choice(len(eval_data.sources), shot_size, replace=False)
            source_nodes, dest_nodes, timestamps = eval_data.sources[eval_indices], eval_data.destinations[eval_indices], eval_data.timestamps[eval_indices]
            new_node_features, _ = generate_final_node_features(
                args, node_features, edge_features, source_nodes, dest_nodes, timestamps,
                full_ngh_finder, models, tokens, device)
            ap, auc = eval_edge_prediction(
                model_features=new_node_features,
                negative_edge_sampler=negative_sampler,
                data=eval_data,
                device=device,
                eval_mode='few_shot',
                shot_size=shot_size
            )
    print(f"   >>> {eval_type} AP: {ap:.4f}, AUC: {auc:.4f}")
    return ap, auc

def eval_edge_prediction(model_features, negative_edge_sampler, data, device, batch_size=100, eval_mode='full', shot_size=10, classifier=None, pos_sampler=None):
    """
    Performs evaluation based on the specified mode.
    'full': Iterates over the entire dataset.
    'few_shot': Computes metrics on a small random sample.
    """
    if len(data.sources) == 0: return 0.0, 0.5
    if hasattr(negative_edge_sampler, 'seed') and negative_edge_sampler.seed is not None:
        negative_edge_sampler.reset_random_state()
    aps, aucs = [], []
    with torch.no_grad():
        if eval_mode == 'full':
            num_batch = 10
            for i in range(num_batch):
                try:
                    sources_batch, destinations_batch, _, actual_size = pos_sampler.sample(batch_size)
                    size = actual_size
                    if size == 0: continue
                except RuntimeError as e:
                    print(f"Sampling error in evaluation: {str(e)}")
                    continue
                try:
                    _, negatives_batch, neg_size = negative_edge_sampler.sample(size)
                except RuntimeError as e:
                    print(f"Negative sampling error in evaluation: {str(e)}")
                    continue
                size = min(size, neg_size)
                if size == 0: continue
                sources_batch = sources_batch[:size]
                destinations_batch = destinations_batch[:size]
                negatives_batch = negatives_batch[:size]
                pos_prob, neg_prob = compute_edge_probabilities(
                    sources_batch, destinations_batch, negatives_batch, model_features, device, models['edge_classifier'])
                y_true = np.concatenate([np.ones(size), np.zeros(size)])
                y_pred = torch.cat([pos_prob, neg_prob]).cpu().numpy()
                aps.append(average_precision_score(y_true, y_pred))
                aucs.append(roc_auc_score(y_true, y_pred))
        elif eval_mode == 'few_shot':
            try:
                sources_batch, destinations_batch, _, actual_size = pos_sampler.sample(shot_size)
                size = actual_size
                if size == 0: return 0.0, 0.5
            except RuntimeError as e:
                print(f"Sampling error in few-shot evaluation: {str(e)}")
                return 0.0, 0.5
            try:
                _, negatives_batch, neg_size = negative_edge_sampler.sample(size)
            except RuntimeError as e:
                print(f"Negative sampling error in few-shot evaluation: {str(e)}")
                return 0.0, 0.5
            size = min(size, neg_size)
            if size == 0: return 0.0, 0.5
            sources_batch = sources_batch[:size]
            destinations_batch = destinations_batch[:size]
            negatives_batch = negatives_batch[:size]
            pos_prob, neg_prob = compute_edge_probabilities(
                sources_batch, destinations_batch, negatives_batch, model_features, device, models['edge_classifier'])
            y_true = np.concatenate([np.ones(size), np.zeros(size)])
            y_pred = torch.cat([pos_prob, neg_prob]).cpu().numpy()
            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")
    return np.mean(aps) if aps else 0.0, np.mean(aucs) if aucs else 0.5

def main(args):
    pre_datasets = get_pretrain_datasets(args)
    device = args.device
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_d_data(
        args, args.dataset, args.different_new_nodes
    )
    node_features = torch.from_numpy(node_features).float().to(device)
    edge_features = torch.from_numpy(edge_features).float().to(device)
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
    train_rand_sampler = StratifiedEdgeSampler(train_data.sources, train_data.destinations, num_classes=5, seed=args.seed)
    val_rand_sampler = StratifiedEdgeSampler(full_data.sources, full_data.destinations, num_classes=5, seed=args.seed)
    nn_val_rand_sampler = StratifiedEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, num_classes=5, seed=args.seed)
    test_rand_sampler = StratifiedEdgeSampler(full_data.sources, full_data.destinations, num_classes=5, seed=args.seed)
    nn_test_rand_sampler = StratifiedEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, num_classes=5, seed=args.seed)
    train_pos_sampler = TemporalStratifiedEdgeSampler(train_data.sources, train_data.destinations, train_data.timestamps, num_dst_classes=5, num_time_bins=5, seed=args.seed)
    val_pos_sampler = TemporalStratifiedEdgeSampler(val_data.sources, val_data.destinations, val_data.timestamps, num_dst_classes=5, num_time_bins=5, seed=args.seed)
    nn_val_pos_sampler = TemporalStratifiedEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, new_node_val_data.timestamps, num_dst_classes=5, num_time_bins=5, seed=args.seed)
    test_pos_sampler = TemporalStratifiedEdgeSampler(test_data.sources, test_data.destinations, test_data.timestamps, num_dst_classes=5, num_time_bins=5, seed=args.seed)
    nn_test_pos_sampler = TemporalStratifiedEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, new_node_test_data.timestamps, num_dst_classes=5, num_time_bins=5, seed=args.seed)
    dynamic_model = TGAT(node_feat_dim=args.node_feat_dim, edge_feat_dim=args.edge_feat_dim, time_dim=args.time_dim, embedding_dim=args.unify_dim, num_layers=args.num_layers, n_head=args.num_heads, drop_out=args.dropout, attn_mode=args.attn_mode, num_domains=len(pre_datasets)).to(device)
    dynamic_model.load_state_dict(torch.load(args.dynamic_model_path, map_location=device))
    dynamic_model.eval()
    for param in dynamic_model.parameters():
        param.requires_grad = False
    static_model = JointContrastiveModel(in_channels=args.node_feat_dim, hidden_channels=args.hid_units, out_channels=args.out_channels, num_layers=args.num_static_layers, dropout=args.static_dropout).to(device)
    static_model.load_state_dict(torch.load(args.static_model_path, map_location=device))
    static_model.eval()
    for param in static_model.parameters():
        param.requires_grad = False
    static_tokens_data = {ds: torch.load(os.path.join(args.static_token_dir, f"{ds}_embeddings.pt"), map_location=device) for ds in pre_datasets}
    dynamic_tokens_data = {ds: (lambda t: t.sum(dim=0) / t.shape[0])(torch.load(os.path.join(args.dynamic_token_dir, f"{ds}_embeddings.pt"), map_location=device)) for ds in pre_datasets}
    static_semantic_token = get_target_static_token(node_features, args.unify_dim, device)
    all_runs_results = []
    Path("./results/").mkdir(parents=True, exist_ok=True)
    for run in range(args.num_runs):
        print(f"\n{'='*25} Starting Run {run + 1}/{args.num_runs} {'='*25}")
        timer = Timer(args.time_dim).to(device)
        moe = MoE(len(pre_datasets), args.branch_weight_static).to(device)
        projection_head = ProjectionHead(args.projection_input_dim, args.projection_output_dim).to(device)
        prompt_generator = PromptGenerator(args.condition_dim, args.condition_dim, args.bottle_neck_mlp_dim, args.unify_dim).to(device)
        edge_classifier = EdgeClassifierV3(args.unify_dim, hidden_dim1=256, hidden_dim2=128, dropout_p=0.5).to(device)
        optimizers = [
            torch.optim.Adam(prompt_generator.parameters(), lr=args.prompt_generator_lr, weight_decay=args.prompt_generator_l2_coef),
            torch.optim.Adam(timer.parameters(), lr=args.timer_lr, weight_decay=args.timer_l2_coef),
            torch.optim.Adam(moe.parameters(), lr=args.moe_lr, weight_decay=args.moe_l2_coef),
            torch.optim.Adam(projection_head.parameters(), lr=args.projection_lr, weight_decay=args.projection_l2_coef),
            torch.optim.Adam(edge_classifier.parameters(), lr=args.classifier_edge_lr, weight_decay=args.classifier_edge_l2_coef)
        ]
        models = {'prompt_generator': prompt_generator, 'timer': timer, 'moe': moe, 'projection_head': projection_head, 'static_model': static_model, 'dynamic_model': dynamic_model, 'edge_classifier': edge_classifier}
        tokens = {'node_features': node_features, 'edge_features': edge_features, 'static_tokens': static_tokens_data, 'dynamic_tokens': dynamic_tokens_data, 'static_semantic_token': static_semantic_token}
        criterion = torch.nn.BCELoss()
        early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)
        for epoch in range(args.epochs):
            print(f"\n------ Epoch {epoch + 1}/{args.epochs} ------")
            loss, moe_loss, cls_loss = train_one_epoch(args, train_data, train_rand_sampler, train_ngh_finder, models, tokens, optimizers, criterion, device)
            print(f"Epoch {epoch + 1}: Loss={loss:.4f} (MOE={moe_loss:.4f}, CLS={cls_loss:.4f})")
            if (epoch + 1) % args.val_freq == 0:
                print("--- Evaluating on Validation Set ---")
                val_ap, _ = evaluate(args, val_data, val_rand_sampler, full_ngh_finder, models, tokens, device, eval_type='Transductive', pos_sampler=val_pos_sampler)
                nn_val_ap, _ = evaluate(args, new_node_val_data, nn_val_rand_sampler, full_ngh_finder, models, tokens, device, eval_type='Inductive', pos_sampler=nn_val_pos_sampler)
                if early_stopper.early_stop_check(val_ap):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        print("\n--- Final Evaluation on Test Set ---")
        test_results = evaluate(args, test_data, test_rand_sampler, full_ngh_finder, models, tokens, device, eval_type='Test', pos_sampler=test_pos_sampler)
        nn_test_results = evaluate(args, new_node_test_data, nn_test_rand_sampler, full_ngh_finder, models, tokens, device, eval_type='Test', pos_sampler=nn_test_pos_sampler)
        run_summary = {
            "test_ap": test_results[0], "test_auc": test_results[1],
            "nn_test_ap": nn_test_results[0], "nn_test_auc": nn_test_results[1]
        }
        all_runs_results.append(run_summary)
        run_results_path = f"./results/{args.dataset}_{args.prefix}_run_{run}.pkl"
        with open(run_results_path, "wb") as f:
            pickle.dump(run_summary, f)
        print(f"Run {run + 1} results saved to {run_results_path}")
    if not all_runs_results:
        print("No runs were completed. Exiting.")
        return
    avg_results = {
        "avg_test_ap": np.mean([res['test_ap'] for res in all_runs_results]),
        "std_test_ap": np.std([res['test_ap'] for res in all_runs_results]),
        "avg_test_auc": np.mean([res['test_auc'] for res in all_runs_results]),
        "std_test_auc": np.std([res['test_auc'] for res in all_runs_results]),
        "avg_nn_test_ap": np.mean([res['nn_test_ap'] for res in all_runs_results]),
        "std_nn_test_ap": np.std([res['nn_test_ap'] for res in all_runs_results]),
        "avg_nn_test_auc": np.mean([res['nn_test_auc'] for res in all_runs_results]),
        "std_nn_test_auc": np.std([res['nn_test_auc'] for res in all_runs_results]),
    }
    print(f"\n{'='*25} Final Average Results over {args.num_runs} Runs {'='*25}")
    print(f"Transductive AP:   {avg_results['avg_test_ap']:.4f} ± {avg_results['std_test_ap']:.4f}")
    print(f"Transductive AUC:  {avg_results['avg_test_auc']:.4f} ± {avg_results['std_test_auc']:.4f}")
    print(f"Inductive AP:      {avg_results['avg_nn_test_ap']:.4f} ± {avg_results['std_nn_test_ap']:.4f}")
    print(f"Inductive AUC:     {avg_results['avg_nn_test_auc']:.4f} ± {avg_results['std_nn_test_auc']:.4f}")
    summary_path = f"./results/{args.dataset}_{args.prefix}_final_summary.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(avg_results, f)
    print(f"\nFinal summary saved to {summary_path}")

if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    print("Running with arguments:")
    print(args)
    main(args)