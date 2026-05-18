import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from DyGFM_dynamic_models.timer import Timer

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.Tanh()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class MapBasedMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        q = torch.unsqueeze(q, dim=2)
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3])

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        k = torch.unsqueeze(k, dim=1)
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3])

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)

        # Map based Attention
        q_k = torch.cat([q, k], dim=3)
        attn = self.weight_map(q_k).squeeze(dim=3)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn

class AttnModel(torch.nn.Module):
    def __init__(self, node_embedding_dim, edge_feat_dim, time_dim, output_dim,
                 n_head=2, drop_out=0.1, attn_mode='prod'):
        super(AttnModel, self).__init__()
        self.model_dim = node_embedding_dim + edge_feat_dim + time_dim
        self.merger = MergeLayer(self.model_dim, node_embedding_dim,
                                 node_embedding_dim, output_dim)
        assert(self.model_dim % n_head == 0)
        self.multi_head_target = MultiHeadAttention(n_head,
                                     d_model=self.model_dim,
                                     d_k=self.model_dim // n_head,
                                     d_v=self.model_dim // n_head,
                                     dropout=drop_out)

    def forward(self, src_embed, src_t_embed, neighbor_embed, neighbor_t_embed, neighbor_e_feat, mask):
        src_ext = torch.unsqueeze(src_embed, dim=1)
        edge_placeholder = torch.zeros(src_ext.shape[0], 1, neighbor_e_feat.shape[2], device=src_ext.device)
        q = torch.cat([src_ext, edge_placeholder, torch.unsqueeze(src_t_embed, dim=1)], dim=2)
        k = torch.cat([neighbor_embed, neighbor_e_feat, neighbor_t_embed], dim=2)
        mask = torch.unsqueeze(mask, dim=2).permute([0, 2, 1])
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)
        output = output.squeeze(1)
        output = self.merger(output, src_embed)
        return output, attn

class NeighborFinder:
    """
    Neighbor finder for temporal graphs.
    """
    def __init__(self, adj_list, uniform=False):
        """
        Initializes the neighbor finder.

        Args:
            adj_list: Adjacency list.
            uniform: Whether to use uniform sampling.
        """
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.off_set_l = off_set_l
        self.uniform = uniform

    def init_off_set(self, adj_list):
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])
            off_set_l.append(len(n_idx_l))

        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before(self, src_idx, cut_time):
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l

        if src_idx >= len(off_set_l) - 1 or src_idx < 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_e_idx, neighbors_ts

        left = 0
        right = len(neighbors_idx) - 1

        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right+1], neighbors_e_idx[:right+1], neighbors_ts[:right+1]
        else:
            return neighbors_idx[:left+1], neighbors_e_idx[:left+1], neighbors_ts[:left+1]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Get temporal neighbors.

        Args:
            src_idx_l: List of source node indices.
            cut_time_l: List of cut-off times.
            num_neighbors: Number of neighbors to sample.

        Returns:
            out_ngh_node_batch: Batch of neighbor nodes.
            out_ngh_eidx_batch: Batch of neighbor edge indices.
            out_ngh_t_batch: Batch of neighbor timestamps.
        """
        assert(len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if self.uniform:
                    # Uniform sampling. If not enough neighbors, sample with replacement.
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

                    # Resort based on time.
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    # If not enough neighbors, pad at the end.
                    n_neighbors = min(len(ngh_idx), num_neighbors)
                    out_ngh_node_batch[i, num_neighbors - n_neighbors:] = ngh_idx[:n_neighbors]
                    out_ngh_t_batch[i, num_neighbors - n_neighbors:] = ngh_ts[:n_neighbors]
                    out_ngh_eidx_batch[i, num_neighbors - n_neighbors:] = ngh_eidx[:n_neighbors]

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch


class TGAT(torch.nn.Module):
    """
    Temporal Graph Attention Network (TGAT) - Decoupled Version.
    """
    def __init__(self,
                 node_feat_dim: int,
                 edge_feat_dim: int,
                 time_dim: int,
                 embedding_dim: int,
                 num_layers=2, n_head=4, drop_out=0.1,
                 attn_mode='prod', num_domains=1):
        super(TGAT, self).__init__()

        self.num_layers = num_layers
        self.num_domains = num_domains
        self.embedding_dim = embedding_dim

        self.node_feat_proj = nn.Linear(node_feat_dim, embedding_dim)

        self.attn_model_list = nn.ModuleList([
            AttnModel(node_embedding_dim=embedding_dim,
                      edge_feat_dim=edge_feat_dim,
                      time_dim=time_dim,
                      output_dim=embedding_dim,
                      n_head=n_head,
                      drop_out=drop_out,
                      attn_mode=attn_mode)
            for _ in range(num_layers)
        ])

        self.affinity_score = MergeLayer(embedding_dim, embedding_dim, embedding_dim, 1)
        self.time_encoders = nn.ModuleList([Timer(time_dim=time_dim) for _ in range(num_domains)])
        adapter_dim = embedding_dim // 4
        self.domain_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, adapter_dim),
                nn.ReLU(),
                nn.Linear(adapter_dim, embedding_dim)
            ) for _ in range(num_domains)
        ])

    def embed_with_adapter(self, src_idx_l, target_idx_l, cut_time_l,
                ngh_finder, domain_id,
                node_features, edge_features,
                num_neighbors=20):
        """
        Computes embeddings for two nodes and applies a domain-specific adapter.
        """
        src_embed = self.tem_conv(src_idx_l, cut_time_l, ngh_finder, self.num_layers, domain_id,
                                  node_features, edge_features, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, ngh_finder, self.num_layers, domain_id,
                                     node_features, edge_features, num_neighbors)

        src_embed = self.domain_adapters[domain_id](src_embed)
        target_embed = self.domain_adapters[domain_id](target_embed)

        return src_embed, target_embed

    def embed_without_adapter_with_new_timer(self, src_idx_l, target_idx_l, cut_time_l, ngh_finder,
                node_features, edge_features, timer,
                num_neighbors=20):

        src_embed = self.tem_conv_with_new_timer(src_idx_l, cut_time_l, ngh_finder, self.num_layers,
                                  node_features, edge_features, timer, num_neighbors)
        target_embed = self.tem_conv_with_new_timer(target_idx_l, cut_time_l, ngh_finder, self.num_layers,
                                     node_features, edge_features, timer, num_neighbors)

        return src_embed, target_embed

    def forward(self, src_idx_l, target_idx_l, cut_time_l,
                ngh_finder, domain_id,
                node_features, edge_features,
                num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, ngh_finder, self.num_layers, domain_id,
                                  node_features, edge_features, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, ngh_finder, self.num_layers, domain_id,
                                     node_features, edge_features, num_neighbors)

        src_embed = self.domain_adapters[domain_id](src_embed)
        target_embed = self.domain_adapters[domain_id](target_embed)

        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score

    def tem_conv_with_new_timer(self, src_idx_l, cut_time_l, ngh_finder, curr_layers,
                 node_features, edge_features, timer,
                 num_neighbors=20):
        if curr_layers == 0:
            raw_node_feat = node_features[torch.from_numpy(src_idx_l).long()]
            return self.node_feat_proj(raw_node_feat)

        device = next(self.parameters()).device
        batch_size = len(src_idx_l)

        src_node_conv_feat = self.tem_conv_with_new_timer(
            src_idx_l, cut_time_l, ngh_finder, curr_layers - 1,
            node_features, edge_features, timer, num_neighbors
        )

        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = ngh_finder.get_temporal_neighbor(
            src_idx_l, cut_time_l, num_neighbors=num_neighbors
        )

        src_ngh_node_batch_flat = src_ngh_node_batch.flatten()
        src_ngh_t_batch_flat = src_ngh_t_batch.flatten()
        src_ngh_conv_feat = self.tem_conv_with_new_timer(
            src_ngh_node_batch_flat, src_ngh_t_batch_flat, ngh_finder, curr_layers - 1,
            node_features, edge_features, timer, num_neighbors
        )
        src_ngh_feat = src_ngh_conv_feat.view(batch_size, num_neighbors, -1)

        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        src_node_t_embed = timer(torch.zeros_like(cut_time_l_th))

        raw_edge_feat = edge_features[torch.from_numpy(src_ngh_eidx_batch).long()]

        time_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
        time_delta_th = torch.from_numpy(time_delta).float().to(device)

        original_shape = time_delta_th.shape
        time_delta_flat = time_delta_th.reshape(-1)
        src_ngh_t_embed_flat = timer(time_delta_flat)
        src_ngh_t_embed = src_ngh_t_embed_flat.view(original_shape[0], original_shape[1], -1)

        mask = (src_ngh_node_batch == 0)
        attn_model = self.attn_model_list[curr_layers - 1]

        output, _ = attn_model(
            src_node_conv_feat, src_node_t_embed,
            src_ngh_feat, src_ngh_t_embed, raw_edge_feat,
            torch.from_numpy(mask).to(device)
        )
        return output

    def tem_conv(self, src_idx_l, cut_time_l, ngh_finder, curr_layers, domain_id,
                 node_features, edge_features, num_neighbors=20):

        if curr_layers == 0:
            raw_node_feat = node_features[torch.from_numpy(src_idx_l).long()]
            return self.node_feat_proj(raw_node_feat)

        device = next(self.parameters()).device
        batch_size = len(src_idx_l)

        src_node_conv_feat = self.tem_conv(
            src_idx_l, cut_time_l, ngh_finder, curr_layers - 1, domain_id,
            node_features, edge_features, num_neighbors
        )

        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = ngh_finder.get_temporal_neighbor(
            src_idx_l, cut_time_l, num_neighbors=num_neighbors
        )

        src_ngh_node_batch_flat = src_ngh_node_batch.flatten()
        src_ngh_t_batch_flat = src_ngh_t_batch.flatten()
        src_ngh_conv_feat = self.tem_conv(
            src_ngh_node_batch_flat, src_ngh_t_batch_flat, ngh_finder, curr_layers - 1, domain_id,
            node_features, edge_features, num_neighbors
        )
        src_ngh_feat = src_ngh_conv_feat.view(batch_size, num_neighbors, -1)

        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        src_node_t_embed = self.time_encoders[domain_id](torch.zeros_like(cut_time_l_th))

        raw_edge_feat = edge_features[torch.from_numpy(src_ngh_eidx_batch).long()]

        time_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
        time_delta_th = torch.from_numpy(time_delta).float().to(device)

        original_shape = time_delta_th.shape
        time_delta_flat = time_delta_th.reshape(-1)
        src_ngh_t_embed_flat = self.time_encoders[domain_id](time_delta_flat)
        src_ngh_t_embed = src_ngh_t_embed_flat.view(original_shape[0], original_shape[1], -1)

        mask = (src_ngh_node_batch == 0)
        attn_model = self.attn_model_list[curr_layers - 1]

        output, _ = attn_model(
            src_node_conv_feat, src_node_t_embed,
            src_ngh_feat, src_ngh_t_embed, raw_edge_feat,
            torch.from_numpy(mask).to(device)
        )
        return output

    def compute_edge_probabilities(self, src_idx_l, target_idx_l, neg_idx_l,
                                   cut_time_l, ngh_finder, timer,
                                   node_features, edge_features,
                                   num_neighbors=20):
        """
        Computes the existence probabilities for positive and negative edges.
        This function is mainly used for training as it handles both positive and negative samples.

        Args:
            src_idx_l (np.ndarray): Source node ID array.
            target_idx_l (np.ndarray): Target node ID array (positive samples).
            neg_idx_l (np.ndarray): Negative sampled node ID array.
            cut_time_l (np.ndarray): Interaction timestamp array.
            ... (other arguments are consistent with the forward method)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Probabilities for positive and negative samples.
        """
        n_samples = len(src_idx_l)
        all_node_ids = np.concatenate([src_idx_l, target_idx_l, neg_idx_l])
        all_timestamps = np.concatenate([cut_time_l, cut_time_l, cut_time_l])

        # Compute temporal embeddings for all nodes at once
        all_embeddings = self.tem_conv_with_new_timer(all_node_ids, all_timestamps, ngh_finder, self.num_layers,
                                       node_features, edge_features, timer, num_neighbors)

        # Split the computed embeddings back into source, destination, and negative parts
        source_embedding = all_embeddings[:n_samples]
        destination_embedding = all_embeddings[n_samples: 2 * n_samples]
        negative_embedding = all_embeddings[2 * n_samples:]

        # Efficiently compute scores for positive and negative samples
        score = self.affinity_score(torch.cat([source_embedding, source_embedding], dim=0),
                                    torch.cat([destination_embedding, negative_embedding], dim=0)).squeeze(dim=-1)

        # Split the scores back into positive and negative scores
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        # Apply sigmoid activation to convert scores to probabilities
        return pos_score.sigmoid(), neg_score.sigmoid()
