from data_utils import *
from graph_data import expected_num_nodes, load_static_edge_index, validate_graph_alignment
from utils import *
from data_processing import *
from DyGFM_downstream_models.tgat import *
from DyGFM_downstream_models.static_model import *
from DyGFM_downstream_models.timer import *
from DyGFM_downstream_models.moe import *
from DyGFM_downstream_models.prompt_gen import *
from DyGFM_downstream_models.classifier import MLPClassifier
import time
import os
import math
import numpy as np
import torch
import pickle
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import swanlab


def sample_labeled_indices(labels, pos_k, neg_k):
    pos_indices = [i for i, y in enumerate(labels) if y == 1]
    neg_indices = [i for i, y in enumerate(labels) if y == 0]
    pos_k = min(pos_k, len(pos_indices))
    neg_k = min(neg_k, len(neg_indices))
    indices = []
    if pos_k > 0:
        indices += random.sample(pos_indices, pos_k)
    if neg_k > 0:
        indices += random.sample(neg_indices, neg_k)
    return indices


def get_target_dynamic_embeddings(node_features, edge_features, src_idx_l, target_idx_l, cut_time_l, ngh_finder, timer, args, device, model):
    batch_size = 1000
    num_edges = len(src_idx_l)
    num_batches = (num_edges + batch_size - 1) // batch_size
    print(f"Total {num_edges} edges, processing in {num_batches} batches")
    all_src_embeds = []
    all_dst_embeds = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_edges)
        batch_sources = src_idx_l[start_idx:end_idx]
        batch_destinations = target_idx_l[start_idx:end_idx]
        batch_timestamps = cut_time_l[start_idx:end_idx]
        print(f"Processing batch {batch_idx+1}/{num_batches}, num_edges: {len(batch_sources)}")
        src_embed, dst_embed = model.embed_without_adapter_with_new_timer(
            batch_sources, batch_destinations, batch_timestamps,
            ngh_finder,
            node_features, edge_features,
            num_neighbors=args.num_neighbors,
            timer=timer
        )
        all_src_embeds.append(src_embed)
        all_dst_embeds.append(dst_embed)
    src_embeddings = torch.cat(all_src_embeds, dim=0)
    dst_embeddings = torch.cat(all_dst_embeds, dim=0)
    embeddings = (src_embeddings, dst_embeddings)
    num_nodes = node_features.shape[0]
    embedding_dim = src_embeddings.shape[1]
    id_count = np.zeros(num_nodes)
    new_embeddings = torch.zeros((num_nodes, embedding_dim), device=device)
    for idx, i in enumerate(src_idx_l):
        id_count[i] += 1
        new_embeddings[i] += src_embeddings[idx]
    for idx, i in enumerate(target_idx_l):
        id_count[i] += 1
        new_embeddings[i] += dst_embeddings[idx]
    for i in range(1, num_nodes):
        if id_count[i] > 0:
            new_embeddings[i] /= id_count[i]
    print(f"New node embeddings shape: {new_embeddings.shape}")
    return new_embeddings

def get_target_dynamic_token(node_features, edge_features, src_idx_l, target_idx_l, cut_time_l, ngh_finder, timer, args, device, model):
    batch_size = 1000
    num_edges = len(src_idx_l)
    num_batches = (num_edges + batch_size - 1) // batch_size
    print(f"Total {num_edges} edges, processing in {num_batches} batches")
    all_src_embeds = []
    all_dst_embeds = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_edges)
        batch_sources = src_idx_l[start_idx:end_idx]
        batch_destinations = target_idx_l[start_idx:end_idx]
        batch_timestamps = cut_time_l[start_idx:end_idx]
        with torch.no_grad():
            src_embed, dst_embed = model.embed_without_adapter_with_new_timer(
                batch_sources, batch_destinations, batch_timestamps,
                ngh_finder,
                node_features, edge_features,
                num_neighbors=args.num_neighbors,
                timer=timer
            )
            all_src_embeds.append(src_embed)
            all_dst_embeds.append(dst_embed)
    src_embeddings = torch.cat(all_src_embeds, dim=0)
    dst_embeddings = torch.cat(all_dst_embeds, dim=0)
    embeddings = (src_embeddings, dst_embeddings)
    num_nodes = node_features.shape[0]
    embedding_dim = src_embeddings.shape[1]
    id_count = np.zeros(num_nodes)
    new_embeddings = torch.zeros((num_nodes, embedding_dim), device=device)
    for idx, i in enumerate(src_idx_l):
        id_count[i] += 1
        new_embeddings[i] += src_embeddings[idx]
    for idx, i in enumerate(target_idx_l):
        id_count[i] += 1
        new_embeddings[i] += dst_embeddings[idx]
    for i in range(1, num_nodes):
        if id_count[i] > 0:
            new_embeddings[i] /= id_count[i]
    time_features = torch.mean(new_embeddings, dim=0)
    print(f"Temporal feature vector shape: {time_features.shape}")
    return time_features

def split_dataset(task_time_set, full_data, task, test_indices):
    time_stamp = task_time_set[task]
    ts_flag = (full_data.timestamps <= time_stamp)
    index = np.where(full_data.timestamps == time_stamp)[0][0]
    ts_label_flag_1 = (ts_flag) * (full_data.labels)
    ts_label_flag_1 = ts_label_flag_1[0:index+1]
    record = {}
    for i in range(len(ts_label_flag_1)-1, -1, -1):
        if full_data.sources[i] in record:
            ts_label_flag_1[i] = -1
        else:
            record[full_data.sources[i]] = 1
    num_indices = 10
    pos_pool = set(np.where(ts_label_flag_1 == 1)[0])
    neg_pool = set(np.where(ts_label_flag_1 == 0)[0])
    train_indices_1 = random.sample(pos_pool, min(num_indices, len(pos_pool)))
    train_indices_0 = random.sample(neg_pool, min(num_indices * 5, len(neg_pool)))
    ts_label_flag_1[train_indices_1], ts_label_flag_1[train_indices_0] = -1, -2
    pos_pool = set(np.where(ts_label_flag_1 == 1)[0])
    neg_pool = set(np.where(ts_label_flag_1 == 0)[0])
    val_indices_1 = random.sample(pos_pool, min(num_indices, len(pos_pool)))
    val_indices_0 = random.sample(neg_pool, min(num_indices * 5, len(neg_pool)))
    train_indices = train_indices_1 + train_indices_0
    val_indices = val_indices_1 + val_indices_0
    train_data = Data(full_data.sources[train_indices], full_data.destinations[train_indices], full_data.timestamps[train_indices],
                      full_data.edge_idxs[train_indices], full_data.labels[train_indices])
    val_data = Data(full_data.sources[val_indices], full_data.destinations[val_indices], full_data.timestamps[val_indices],
                      full_data.edge_idxs[val_indices], full_data.labels[val_indices])
    test_data = Data(full_data.sources[test_indices], full_data.destinations[test_indices], full_data.timestamps[test_indices],
                      full_data.edge_idxs[test_indices], full_data.labels[test_indices])
    return train_data, val_data, test_data

def get_target_static_token(node_features, unify_dim):
    static_tokens = svd_dim_reduction(node_features, unify_dim)
    print(f"Static feature vector shape: {static_tokens.shape}")
    return static_tokens

def svd_dim_reduction(tensor, unify_dim):
    """Reduces the dimension of a [N, D] tensor to [N, unify_dim] using SVD."""
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
    reduced = torch.matmul(tensor, Vh.T[:, :unify_dim])
    return reduced

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

def get_dynamic_model_and_static_model(args):
    dynamic_model = TGAT(
        node_feat_dim=args.node_feat_dim,
        edge_feat_dim=args.edge_feat_dim,
        time_dim=args.time_dim,
        embedding_dim=args.unify_dim,
        num_layers=args.num_layers,
        n_head=args.num_heads,
        drop_out=args.dropout,
        attn_mode=args.attn_mode,
        num_domains=args.num_domains
    ).to(args.device)
    dynamic_model.load_state_dict(torch.load(args.dynamic_model_path))
    dynamic_model.eval()
    static_model = JointContrastiveModel(
        in_channels=args.node_feat_dim,
        hidden_channels=args.hid_units,
        out_channels=args.out_channels,
        num_layers=args.num_static_layers,
        dropout=args.static_dropout
    ).to(args.device)
    static_model.load_state_dict(torch.load(args.static_model_path))
    static_model.eval()
    for param in dynamic_model.parameters():
        param.requires_grad = False
    for param in static_model.parameters():
        param.requires_grad = False
    return dynamic_model, static_model

def get_static_tokens_and_dynamic_tokens(args, pre_datasets):
    static_token_dir = args.static_token_dir
    static_tokens = {}
    dynamic_token_dir = args.dynamic_token_dir
    dynamic_tokens = {}
    for dataset in pre_datasets:
        per_tokens = torch.load(os.path.join(static_token_dir, f"{dataset}_embeddings.pt")).to(args.device)
        static_tokens[dataset] = per_tokens
        per_tokens = torch.load(os.path.join(dynamic_token_dir, f"{dataset}_embeddings.pt")).to(args.device)
        dynamic_tokens[dataset] = per_tokens.sum(dim=0) / per_tokens.shape[0]
    return static_tokens, dynamic_tokens

def get_node_feat(node_features, edge_features, sources_batch, destinations_batch, timestamps_batch, train_ngh_finder, timer, args, dynamic_model, static_model, prompt_generator, moe, projection_head, static_tokens, dynamic_tokens, static_semantic_token):
    device = args.device
    time_tar_token = get_target_dynamic_token(
        node_features,
        edge_features,
        sources_batch,
        destinations_batch,
        timestamps_batch,
        train_ngh_finder,
        timer,
        args,
        device,
        dynamic_model,
    )
    moe_loss, kl, semantic_token, dynamic_token = moe(
        static_tokens, dynamic_tokens, time_tar_token, static_semantic_token, args
    )
    static_tar_token = static_semantic_token.sum(dim=0) / static_semantic_token.shape[0]
    kl_as_vector_1d = kl.view(1)
    input_semantic = torch.cat([static_tar_token, kl_as_vector_1d], dim=0)
    input_temporal = torch.cat([time_tar_token, kl_as_vector_1d], dim=0)
    semantic_prompt, temporal_prompt = prompt_generator(input_semantic, input_temporal)
    static_tar_token = static_tar_token.unsqueeze(0).repeat(node_features.size(0), 1)
    semantic_prompt = semantic_prompt.unsqueeze(0).repeat(node_features.size(0), 1)
    temporal_prompt = temporal_prompt.unsqueeze(0).repeat(node_features.size(0), 1)
    time_tar_token = time_tar_token.unsqueeze(0).repeat(node_features.size(0), 1)
    static_node_features = torch.cat([node_features, static_tar_token, semantic_prompt], dim=1)
    dynamic_node_features = torch.cat([node_features, time_tar_token, temporal_prompt], dim=1)
    static_node_features = projection_head(static_node_features)
    dynamic_node_features = projection_head(dynamic_node_features)
    adj = get_adj_tensor(args, num_nodes=node_features.shape[0], device=device)
    feature_list = [static_node_features]
    adj_list = [adj]
    static_node_features = static_model.get_embeddings(feature_list, adj_list)
    dynamic_node_features = get_target_dynamic_embeddings(
        dynamic_node_features,
        edge_features,
        sources_batch,
        destinations_batch,
        timestamps_batch,
        train_ngh_finder,
        timer,
        args,
        device,
        dynamic_model,
    )
    new_node_features = (
        static_node_features * args.node_feature_static_weight
        + dynamic_node_features * (1 - args.node_feature_static_weight)
    )
    return new_node_features, moe_loss

def train_one_epoch(train_args):
    full_data = train_args['full_data']
    node_features = train_args['node_features']
    edge_features = train_args['edge_features']
    train_data = train_args['train_data']
    train_ngh_finder = train_args['train_ngh_finder']
    timer = train_args['timer']
    args = train_args['args']
    device = train_args['device']
    dynamic_model = train_args['dynamic_model']
    static_model = train_args['static_model']
    prompt_generator = train_args['prompt_generator']
    moe = train_args['moe']
    projection_head = train_args['projection_head']
    classifier = train_args['classifier']
    prompt_generator_optimizer = train_args['prompt_generator_optimizer']
    timer_optimizer = train_args['timer_optimizer']
    moe_optimizer = train_args['moe_optimizer']
    projection_head_optimizer = train_args['projection_head_optimizer']
    classifier_optimizer = train_args['classifier_optimizer']
    static_tokens = train_args['static_tokens']
    dynamic_tokens = train_args['dynamic_tokens']
    static_semantic_token = train_args['static_semantic_token']
    criterion = train_args['criterion']
    prompt_generator_optimizer.zero_grad()
    timer_optimizer.zero_grad()
    moe_optimizer.zero_grad()
    projection_head_optimizer.zero_grad()
    classifier_optimizer.zero_grad()
    prompt_generator.train()
    timer.train()
    moe.train()
    projection_head.train()
    classifier.train()
    indices = sample_labeled_indices(
        train_data.labels, args.train_shot_num, args.train_shot_num * 5
    )
    sources_batch = train_data.sources[indices]
    destinations_batch = train_data.destinations[indices]
    timestamps_batch = train_data.timestamps[indices]
    edge_idxs_batch = full_data.edge_idxs[indices]
    labels_batch = train_data.labels[indices]
    new_node_features, moe_loss = get_node_feat(node_features, edge_features, sources_batch, destinations_batch, timestamps_batch, train_ngh_finder, timer, args, dynamic_model, static_model, prompt_generator, moe, projection_head, static_tokens, dynamic_tokens, static_semantic_token)
    source_embeddings = new_node_features[sources_batch]
    labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
    pred = classifier(source_embeddings).squeeze()
    classifier_loss = criterion(pred, labels_batch_torch)
    total_loss = classifier_loss + args.moe_loss_weight * moe_loss
    total_loss.backward()
    prompt_generator_optimizer.step()
    timer_optimizer.step()
    moe_optimizer.step()
    projection_head_optimizer.step()
    classifier_optimizer.step()
    return total_loss.item()

def evaluate_one_epoch(evaluate_args):
    full_data = evaluate_args['full_data']
    node_features = evaluate_args['node_features']
    edge_features = evaluate_args['edge_features']
    val_data = evaluate_args['val_data']
    val_ngh_finder = evaluate_args['val_ngh_finder']
    timer = evaluate_args['timer']
    args = evaluate_args['args']
    device = evaluate_args['device']
    dynamic_model = evaluate_args['dynamic_model']
    static_model = evaluate_args['static_model']
    prompt_generator = evaluate_args['prompt_generator']
    moe = evaluate_args['moe']
    projection_head = evaluate_args['projection_head']
    classifier = evaluate_args['classifier']
    static_tokens = evaluate_args['static_tokens']
    dynamic_tokens = evaluate_args['dynamic_tokens']
    static_semantic_token = evaluate_args['static_semantic_token']
    criterion = evaluate_args['criterion']
    classifier.eval()
    prompt_generator.eval()
    timer.eval()
    moe.eval()
    projection_head.eval()
    with torch.no_grad():
        indices = sample_labeled_indices(
            val_data.labels, args.val_shot_num, args.val_shot_num * 5
        )
        if len(indices) == 0:
            return 0.5
        sources_batch = val_data.sources[indices]
        destinations_batch = val_data.destinations[indices]
        timestamps_batch = val_data.timestamps[indices]
        labels_batch = val_data.labels[indices]
        new_node_features, _ = get_node_feat(node_features, edge_features, sources_batch, destinations_batch, timestamps_batch, val_ngh_finder, timer, args, dynamic_model, static_model, prompt_generator, moe, projection_head, static_tokens, dynamic_tokens, static_semantic_token)
        source_embeddings = new_node_features[sources_batch]
        labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
        pred = classifier(source_embeddings).squeeze()
        pred_score = torch.sigmoid(pred).cpu().numpy()
        labels_batch_np = labels_batch_torch.cpu().numpy()
        if len(np.unique(labels_batch_np)) < 2:
            return 0.5
        val_auc = roc_auc_score(labels_batch_np, pred_score)
    return val_auc

def test(test_args):
    """Evaluates the model on the test set."""
    timer = test_args['timer']
    moe = test_args['moe']
    projection_head = test_args['projection_head']
    prompt_generator = test_args['prompt_generator']
    classifier = test_args['classifier']
    static_tokens = test_args['static_tokens']
    dynamic_tokens = test_args['dynamic_tokens']
    static_semantic_token = test_args['static_semantic_token']
    test_data = test_args['test_data']
    test_ngh_finder = test_args['test_ngh_finder']
    node_features = test_args['node_features']
    edge_features = test_args['edge_features']
    args = test_args['args']
    device = test_args['device']
    dynamic_model = test_args['dynamic_model']
    static_model = test_args['static_model']
    classifier.eval()
    prompt_generator.eval()
    timer.eval()
    moe.eval()
    projection_head.eval()
    with torch.no_grad():
        test_size = len(test_data.sources)
        batch_size = 1000
        num_test_batches = (test_size + batch_size - 1) // batch_size
        all_pred_scores = []
        all_labels = []
        for batch_idx in range(num_test_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, test_size)
            batch_sources = test_data.sources[start_idx:end_idx]
            batch_destinations = test_data.destinations[start_idx:end_idx]
            batch_timestamps = test_data.timestamps[start_idx:end_idx]
            batch_edge_idxs = test_data.edge_idxs[start_idx:end_idx]
            batch_labels = test_data.labels[start_idx:end_idx]
            new_node_features, _ = get_node_feat(
                node_features, edge_features,
                batch_sources, batch_destinations, batch_timestamps,
                test_ngh_finder,
                timer, args, dynamic_model, static_model,
                prompt_generator, moe, projection_head,
                static_tokens, dynamic_tokens, static_semantic_token
            )
            source_embeddings = new_node_features[batch_sources]
            batch_labels_torch = torch.from_numpy(batch_labels).float().to(device)
            pred = classifier(source_embeddings).squeeze()
            pred_score = torch.sigmoid(pred).cpu().numpy()
            all_pred_scores.extend(pred_score)
            all_labels.extend(batch_labels)
        all_pred_scores = np.array(all_pred_scores)
        all_labels = np.array(all_labels)
        test_auc = roc_auc_score(all_labels, all_pred_scores)
        pred_labels = (all_pred_scores > 0.5).astype(int)
        test_acc = np.mean(pred_labels == all_labels)
        from sklearn.metrics import f1_score
        test_f1 = f1_score(all_labels, pred_labels)
        return test_auc, test_acc, test_f1

def main(args):
    pre_datasets = get_pretrain_datasets(args)
    device = args.device
    node_features, edge_features, full_data, _, _, _, _, _ = get_d_data(
        args, args.dataset, args.different_new_nodes
    )
    if isinstance(node_features, torch.Tensor):
        node_features = node_features.float().to(device)
    else:
        node_features = torch.from_numpy(node_features).float().to(device)
    edge_features = torch.from_numpy(edge_features).float().to(device)

    num_nodes = expected_num_nodes(args.dataset)
    if node_features.shape[0] != num_nodes:
        raise ValueError(
            f"node_features rows {node_features.shape[0]} != expected {num_nodes}"
        )
    max_node_id = int(max(full_data.sources.max(), full_data.destinations.max()))
    if max_node_id >= num_nodes:
        raise ValueError(f"max node id {max_node_id} >= num_nodes {num_nodes}")
    validate_graph_alignment(
        args.dataset, num_nodes, load_static_edge_index(args.dataset)
    )
    task_time_set, test_indices = get_task_time_set(args, full_data)
    total_auc = []
    total_acc = []
    total_f1 = []
    for task in tqdm(range(args.task_num)):
        train_data, val_data, test_data = split_dataset(task_time_set, full_data, task, test_indices)
        train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
        val_ngh_finder = get_neighbor_finder(val_data, args.uniform)
        test_ngh_finder = get_neighbor_finder(test_data, args.uniform)
        run_auc,run_acc,run_f1 = [],[],[]
        dynamic_model, static_model = get_dynamic_model_and_static_model(args)
        criterion = torch.nn.BCEWithLogitsLoss()
        static_semantic_token = get_target_static_token(node_features, args.unify_dim)
        static_tokens, dynamic_tokens = get_static_tokens_and_dynamic_tokens(args, pre_datasets)
        new_nodes_val_aps = []
        val_aps = []
        epoch_times = []
        train_losses = []
        early_stopper = EarlyStopMonitor(max_round=args.patience)
        for i in range(args.num_runs):
            timer = Timer(args.time_dim).to(device)
            moe = MoE(len(pre_datasets), args.branch_weight_static).to(device)
            projection_head = ProjectionHead(args.projection_input_dim, args.projection_output_dim).to(device)
            prompt_generator = PromptGenerator(args.condition_dim, args.condition_dim, args.bottle_neck_mlp_dim, args.unify_dim).to(device)
            classifier = MLPClassifier(args.unify_dim, args.unify_dim // 2).to(device)
            prompt_generator_optimizer = torch.optim.Adam(prompt_generator.parameters(), lr=args.prompt_generator_lr, weight_decay=args.prompt_generator_l2_coef)
            timer_optimizer = torch.optim.Adam(timer.parameters(), lr=args.timer_lr, weight_decay=args.timer_l2_coef)
            moe_optimizer = torch.optim.Adam(moe.parameters(), lr=args.moe_lr, weight_decay=args.moe_l2_coef)
            projection_head_optimizer = torch.optim.Adam(projection_head.parameters(), lr=args.projection_lr, weight_decay=args.projection_l2_coef)
            classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_l2_coef)
            for epoch in range(args.epochs):
                start_epoch = time.time()
                train_args = {
                    'full_data': full_data,
                    'node_features': node_features,
                    'edge_features': edge_features,
                    'train_data': train_data,
                    'train_ngh_finder': train_ngh_finder,
                    'timer': timer,
                    'args': args,
                    'device': device,
                    'dynamic_model': dynamic_model,
                    'static_model': static_model,
                    'moe': moe,
                    'projection_head': projection_head,
                    'prompt_generator': prompt_generator,
                    'classifier': classifier,
                    'prompt_generator_optimizer': prompt_generator_optimizer,
                    'timer_optimizer': timer_optimizer,
                    'moe_optimizer': moe_optimizer,
                    'projection_head_optimizer': projection_head_optimizer,
                    'classifier_optimizer': classifier_optimizer,
                    'static_tokens': static_tokens,
                    'dynamic_tokens': dynamic_tokens,
                    'static_semantic_token': static_semantic_token,
                    'criterion': criterion
                }
                total_loss = train_one_epoch(train_args)
                print(f"Epoch {epoch+1}/{args.epochs}: Loss: {total_loss:.4f}")
                train_losses.append(total_loss)
                evaluate_args = {
                    'full_data': full_data,
                    'node_features': node_features,
                    'edge_features': edge_features,
                    'val_data': val_data,
                    'val_ngh_finder': val_ngh_finder,
                    'timer': timer,
                    'args': args,
                    'device': device,
                    'dynamic_model': dynamic_model,
                    'static_model': static_model,
                    'prompt_generator': prompt_generator,
                    'moe': moe,
                    'projection_head': projection_head,
                    'classifier': classifier,
                    'static_tokens': static_tokens,
                    'dynamic_tokens': dynamic_tokens,
                    'static_semantic_token': static_semantic_token,
                    'criterion': criterion
                }
                val_auc = evaluate_one_epoch(evaluate_args)
                print(f"Val AUC: {val_auc:.4f}")
                val_aps.append(val_auc)
                should_stop, _ = early_stopper.early_stop_check(val_auc, classifier)
                if should_stop:
                    print(f'Early stopping at epoch {epoch}')
                    break
                torch.save({
                    'timer': timer.state_dict(),
                    'moe': moe.state_dict(),
                    'projection_head': projection_head.state_dict(),
                    'prompt_generator': prompt_generator.state_dict(),
                    'classifier': classifier.state_dict()
                }, get_checkpoint_path(epoch, args))
                epoch_time = time.time() - start_epoch
                epoch_times.append(epoch_time)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs}: Loss: {total_loss:.4f}, Val AUC: {val_auc:.4f}, Time: {epoch_time:.2f}s")
            best_epoch = early_stopper.best_epoch
            best_checkpoint = torch.load(get_checkpoint_path(best_epoch, args))
            timer.load_state_dict(best_checkpoint['timer'])
            moe.load_state_dict(best_checkpoint['moe'])
            projection_head.load_state_dict(best_checkpoint['projection_head'])
            prompt_generator.load_state_dict(best_checkpoint['prompt_generator'])
            classifier.load_state_dict(best_checkpoint['classifier'])
            test_args = {
                'timer': timer,
                'moe': moe,
                'projection_head': projection_head,
                'prompt_generator': prompt_generator,
                'classifier': classifier,
                'dynamic_model': dynamic_model,
                'static_model': static_model,
                'node_features': node_features,
                'edge_features': edge_features,
                'test_data': test_data,
                'test_ngh_finder': test_ngh_finder,
                'static_tokens': static_tokens,
                'dynamic_tokens': dynamic_tokens,
                'static_semantic_token': static_semantic_token,
                'args': args,
                'device': device
            }
            test_auc, test_acc, test_f1 = test(test_args)
            print(f"Run {i+1}/{args.num_runs} - Test AUC: {test_auc:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
            run_auc.append(test_auc)
            run_acc.append(test_acc)
            run_f1.append(test_f1)
            results_path = f"results/{args.prefix}_node_classification_{i}.pkl" if i > 0 else f"results/{args.prefix}_node_classification.pkl"
            Path("results/").mkdir(parents=True, exist_ok=True)
            pickle.dump({
                "val_aps": val_aps,
                "test_ap": test_auc,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "train_losses": train_losses,
                "epoch_times": epoch_times,
                "new_nodes_val_aps": new_nodes_val_aps,
                "new_node_test_ap": 0,
            }, open(results_path, "wb"))
        total_auc.append(sum(run_auc)/args.num_runs)
        total_acc.append(sum(run_acc)/args.num_runs)
        total_f1.append(sum(run_f1)/args.num_runs)
        print(f"Task {task} - Avg AUC: {sum(run_auc)/args.num_runs:.4f}, Avg Acc: {sum(run_acc)/args.num_runs:.4f}, Avg F1: {sum(run_f1)/args.num_runs:.4f}")
    folder_path = "./"
    np.savetxt(f"{folder_path}/{args.name}_total_mean_auc.txt", [sum(total_auc)/args.task_num], fmt='%s')
    np.savetxt(f"{folder_path}/{args.name}_total_mean_acc.txt", [sum(total_acc)/args.task_num], fmt='%s')
    np.savetxt(f"{folder_path}/{args.name}_total_mean_f1.txt", [sum(total_f1)/args.task_num], fmt='%s')
    print(f"Final results - Avg AUC: {sum(total_auc)/args.task_num:.4f}, Avg Acc: {sum(total_acc)/args.task_num:.4f}, Avg F1: {sum(total_f1)/args.task_num:.4f}")

if __name__ == "__main__":
    args = get_args()
    main(args)