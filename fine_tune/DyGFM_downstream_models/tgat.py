import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from DyGFM_downstream_models.timer import Timer

class ProjectionHead(nn.Module):
    """
    A Projection Head model used to map (project) the dimension
    of input embeddings to a specified target dimension.
    """
    def __init__(self, input_dim: int, output_dim: int = 172):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim), # First fully connected layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward propagation logic.
        
        Args:
            x (torch.Tensor): The input embedding tensor, shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: The output embedding tensor, shape (batch_size, output_dim).
        """
        return self.layers(x)

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

class TGAT(torch.nn.Module):
    """
    Temporal Graph Attention Network (TGAT) - Final Decoupled Version
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
        This function computes and outputs embeddings for two nodes using an adapter.
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
            src_idx_l, cut_time_l, n_neighbors=num_neighbors
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
            src_idx_l, cut_time_l, n_neighbors=num_neighbors
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
