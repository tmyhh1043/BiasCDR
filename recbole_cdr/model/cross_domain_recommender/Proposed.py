"""Stable two-stream LightGCN for RecBole-CDR.

Preference stream: pairwise ranking, and the only stream used at inference.
Bias stream: predicts popularity statistics from source/target training graphs.
Hybrid Selective-CL: unique overlapping users, detached risk weights.

IMPORTANT: popularity tendency is an observational proxy, not a causal effect.
"""
import csv
import os
from logging import getLogger

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


def _cfg(config, key, default):
    try:
        value = config[key]
        return default if value is None else value
    except (KeyError, TypeError):
        return default


def _load_source_user_feature(domain_config, item_id_map, interactions,
                              num_users, num_items, modality):
    """Aggregate precomputed item features using only SOURCE TRAIN interactions."""
    directory = domain_config.get('multimodal_feat_path')
    feat_name = domain_config.get(f'{modality}_feat_file')
    map_name = domain_config.get('item_map_file')
    if not all([directory, feat_name, map_name]):
        raise ValueError(f'Missing source {modality} configuration')

    feat_path = os.path.join(directory, feat_name)
    map_path = os.path.join(directory, map_name)
    raw = np.load(feat_path, mmap_mode='r')
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f'Expected 2D numeric features: {feat_path}')

    id_col = domain_config.get('item_feat_id_field', 'item_id')
    row_col = domain_config.get('item_feat_index_field', 'itemID')
    mapped_rows = np.full(num_items, -1, dtype=np.int64)
    with open(map_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not {id_col, row_col}.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing {id_col}/{row_col} in {map_path}')
        for record in reader:
            gid = item_id_map.get(record[id_col])
            if gid is None:
                continue
            gid = int(gid)
            rid = int(record[row_col])
            if not 0 < gid < num_items:
                continue
            if not 0 <= rid < raw.shape[0]:
                raise ValueError(f'Invalid feature row {rid} in {map_path}')
            if mapped_rows[gid] not in (-1, rid):
                raise ValueError(f'Conflicting mappings for {record[id_col]}')
            mapped_rows[gid] = rid

    # Batch conversion prevents keeping the entire float64 feature file on GPU.
    features = np.empty(raw.shape, dtype=np.float32)
    valid_feature_row = np.zeros(raw.shape[0], dtype=bool)
    for begin in range(0, raw.shape[0], 2048):
        end = min(begin + 2048, raw.shape[0])
        block = np.asarray(raw[begin:end], dtype=np.float32).copy()
        block[~np.isfinite(block)] = 0.0
        features[begin:end] = block
        valid_feature_row[begin:end] = np.any(block != 0, axis=1)

    coo = interactions.tocoo()
    u, i = coo.row, coo.col
    rows = mapped_rows[i]
    valid = (u > 0) & (i > 0) & (rows >= 0)
    valid[valid] &= valid_feature_row[rows[valid]]
    counts = np.bincount(u[valid], minlength=num_users).astype(np.float32)
    inc = sp.coo_matrix((np.ones(int(valid.sum()), dtype=np.float32),
                         (u[valid], rows[valid])),
                        shape=(num_users, raw.shape[0])).tocsr()
    profile = np.asarray(inc @ features, dtype=np.float32)
    profile /= np.maximum(counts[:, None], 1.0)
    profile[0] = 0
    return torch.from_numpy(profile), torch.from_numpy(counts > 0), {
        'mapped_items': int((mapped_rows >= 0).sum()),
        'users_with_features': int((counts > 0).sum()),
        'dim': int(raw.shape[1]),
    }


def _popularity_targets(matrix, num_users, num_items):
    """Standardized item popularity and mean consumed-popularity per user."""
    matrix = matrix.tocsr()
    degrees = np.asarray(matrix.sum(axis=0)).reshape(-1)
    log_degree = np.log1p(degrees).astype(np.float32)
    active = degrees > 0
    if active.any():
        mu = float(log_degree[active].mean())
        sigma = max(float(log_degree[active].std()), 1e-3)
    else:
        mu, sigma = 0., 1.
    item_z = (log_degree - mu) / sigma
    item_z[~active] = 0
    # Observed average popularity is a proxy, not an estimate of causal bias.
    counts = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float32)
    user_z = np.asarray(matrix @ item_z).reshape(-1) / np.maximum(counts, 1)
    user_z[counts == 0] = 0
    return (torch.tensor(item_z, dtype=torch.float32),
            torch.tensor(user_z, dtype=torch.float32),
            torch.tensor(counts > 0, dtype=torch.bool))


class Proposed(CrossDomainRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.device = config['device']
        self.embedding_size = int(config['embedding_size'])
        self.n_layers = int(config['n_layers'])
        self.reg_weight = float(config['reg_weight'])
        self.lambda_cl = float(_cfg(config, 'lambda_cl', 0.01))
        self.lambda_pop = float(_cfg(config, 'lambda_pop', 0.01))
        self.lambda_orth = float(_cfg(config, 'lambda_orth', 0.001))
        self.lambda_multi = float(_cfg(config, 'lambda_multi', 0.0))
        self.gamma = float(_cfg(config, 'gamma', 2.0))
        self.beta = float(_cfg(config, 'beta', 0.5))
        self.tau = float(_cfg(config, 'tau', 0.2))
        self.full_sort_chunk_size = int(_cfg(config, 'full_sort_chunk_size', 2048))
        if not 0 <= self.beta <= 1 or self.tau <= 0 or self.gamma < 0:
            raise ValueError('Require beta in [0,1], tau > 0, gamma >= 0')
        self.logger = getLogger(__name__)

        self.pref_u_emb_src = nn.Embedding(self.total_num_users, self.embedding_size)
        self.pref_i_emb_src = nn.Embedding(self.total_num_items, self.embedding_size)
        self.pref_u_emb_tgt = nn.Embedding(self.total_num_users, self.embedding_size)
        self.pref_i_emb_tgt = nn.Embedding(self.total_num_items, self.embedding_size)
        self.bias_u_emb_src = nn.Embedding(self.total_num_users, self.embedding_size)
        self.bias_i_emb_src = nn.Embedding(self.total_num_items, self.embedding_size)
        self.bias_u_emb_tgt = nn.Embedding(self.total_num_users, self.embedding_size)
        self.bias_i_emb_tgt = nn.Embedding(self.total_num_items, self.embedding_size)

        # Each head gives the corresponding bias embedding a direct objective.
        self.pop_item_head = nn.Linear(self.embedding_size, 1)
        self.pop_user_head = nn.Linear(self.embedding_size, 1)
        self.rec_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        src = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        tgt = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        self.register_buffer('source_norm_adj', self._normalized_adj(src), persistent=False)
        self.register_buffer('target_norm_adj', self._normalized_adj(tgt), persistent=False)
        for prefix, mat in [('src', src), ('tgt', tgt)]:
            item_z, user_z, valid = _popularity_targets(mat, self.total_num_users, self.total_num_items)
            self.register_buffer(f'{prefix}_item_pop', item_z, persistent=False)
            self.register_buffer(f'{prefix}_user_pop', user_z, persistent=False)
            self.register_buffer(f'{prefix}_user_valid', valid, persistent=False)

        self.has_mm = self.lambda_multi > 0 or (self.lambda_cl > 0 and self.beta < 1)
        self.source_text_projection = None
        self.source_visual_projection = None
        if self.has_mm:
            # Avoid silently making up or substituting random/missing modalities.
            source_config = config['source_domain']
            text, text_ok, text_stats = _load_source_user_feature(
                source_config, dataset.source_item_ID_remap_dict,
                src, self.total_num_users, self.total_num_items, 'text')
            visual, visual_ok, visual_stats = _load_source_user_feature(
                source_config, dataset.source_item_ID_remap_dict,
                src, self.total_num_users, self.total_num_items, 'visual')
            self.register_buffer('source_user_text_feature', text, persistent=False)
            self.register_buffer('source_user_visual_feature', visual, persistent=False)
            self.register_buffer('source_user_text_valid', text_ok, persistent=False)
            self.register_buffer('source_user_visual_valid', visual_ok, persistent=False)
            self.source_text_projection = nn.Linear(text.shape[1], self.embedding_size, bias=False)
            self.source_visual_projection = nn.Linear(visual.shape[1], self.embedding_size, bias=False)
            self.logger.info('Multimodal SOURCE train profiles: text=%s visual=%s', text_stats, visual_stats)

        # Initialize FIRST, then zero invalid/padding indices.
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            for emb in (self.pref_u_emb_src, self.bias_u_emb_src):
                emb.weight[self.overlapped_num_users:self.target_num_users].zero_()
                emb.weight[0].zero_()
            for emb in (self.pref_i_emb_src, self.bias_i_emb_src):
                emb.weight[self.overlapped_num_items:self.target_num_items].zero_()
                emb.weight[0].zero_()
            for emb in (self.pref_u_emb_tgt, self.bias_u_emb_tgt):
                emb.weight[self.target_num_users:].zero_()
                emb.weight[0].zero_()
            for emb in (self.pref_i_emb_tgt, self.bias_i_emb_tgt):
                emb.weight[self.target_num_items:].zero_()
                emb.weight[0].zero_()

        self.target_restore_pref_u_e = None
        self.target_restore_pref_i_e = None
        self.other_parameter_name = ['target_restore_pref_u_e', 'target_restore_pref_i_e']
        self.to(self.device)

    def _normalized_adj(self, mat):
        rows, cols = mat.row, mat.col
        n = self.total_num_users + self.total_num_items
        row = np.concatenate([rows, cols + self.total_num_users])
        col = np.concatenate([cols + self.total_num_users, rows])
        adj = sp.coo_matrix((np.ones(len(row), np.float32), (row, col)), shape=(n, n)).tocsr()
        adj.data[:] = 1
        degree = np.asarray(adj.sum(axis=1)).reshape(-1)
        inv = np.power(np.maximum(degree, 1), -0.5)
        inv[degree == 0] = 0
        adj = (sp.diags(inv) @ adj @ sp.diags(inv)).tocoo()
        indices = torch.tensor(np.stack([adj.row, adj.col]), dtype=torch.long)
        values = torch.tensor(adj.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    def _gcn(self, adjacency, user_emb, item_emb):
        embs = torch.cat([user_emb.weight, item_emb.weight], dim=0)
        result = embs
        for _ in range(self.n_layers):
            embs = torch.sparse.mm(adjacency, embs)
            result = result + embs
        result = result / (self.n_layers + 1)
        return torch.split(result, [self.total_num_users, self.total_num_items], dim=0)

    def forward(self):
        src = (self._gcn(self.source_norm_adj, self.pref_u_emb_src, self.pref_i_emb_src)
               + self._gcn(self.source_norm_adj, self.bias_u_emb_src, self.bias_i_emb_src))
        tgt = (self._gcn(self.target_norm_adj, self.pref_u_emb_tgt, self.pref_i_emb_tgt)
               + self._gcn(self.target_norm_adj, self.bias_u_emb_tgt, self.bias_i_emb_tgt))
        return src, tgt

    def _source_mm(self, users):
        text = self.source_user_text_feature[users]
        visual = self.source_user_visual_feature[users]
        text_ok = self.source_user_text_valid[users]
        visual_ok = self.source_user_visual_valid[users]
        t = self.source_text_projection(F.normalize(text, dim=-1))
        v = self.source_visual_projection(F.normalize(visual, dim=-1))
        count = (text_ok.to(t.dtype) + visual_ok.to(t.dtype)).clamp_min(1.)
        result = (t * text_ok[:, None] + v * visual_ok[:, None]) / count[:, None]
        return result, text_ok | visual_ok

    @staticmethod
    def _orthogonal(a, b):
        if a.numel() == 0:
            return a.sum() * 0.
        return F.cosine_similarity(a, b, dim=-1).square().mean()

    def calculate_loss(self, interaction):
        self.init_restore_e()
        (pu_s, pi_s, bu_s, bi_s), (pu_t, pi_t, bu_t, bi_t) = self.forward()
        su = interaction[self.SOURCE_USER_ID]
        si = interaction[self.SOURCE_ITEM_ID]
        sn = interaction[self.SOURCE_NEG_ITEM_ID]
        tu = interaction[self.TARGET_USER_ID]
        ti = interaction[self.TARGET_ITEM_ID]
        tn = interaction[self.TARGET_NEG_ITEM_ID]

        # Preference-only BPR: identical to prediction semantics.
        src_pos = (pu_s[su] * pi_s[si]).sum(-1)
        src_neg = (pu_s[su] * pi_s[sn]).sum(-1)
        tgt_pos = (pu_t[tu] * pi_t[ti]).sum(-1)
        tgt_neg = (pu_t[tu] * pi_t[tn]).sum(-1)
        loss_src = self.rec_loss(src_pos, src_neg)
        loss_tgt = self.rec_loss(tgt_pos, tgt_neg)
        loss_reg = self.reg_weight * (
            self.reg_loss(self.pref_u_emb_src(su), self.pref_i_emb_src(si), self.pref_i_emb_src(sn))
            + self.reg_loss(self.pref_u_emb_tgt(tu), self.pref_i_emb_tgt(ti), self.pref_i_emb_tgt(tn))
        )

        # Item and user bias embeddings each receive an explicit popularity objective.
        zero = loss_src.new_zeros(())
        loss_pop = zero
        if self.lambda_pop > 0:
            src_users = torch.unique(su[su > 0])
            tgt_users = torch.unique(tu[tu > 0])
            for item_e, user_e, item_ids, user_ids, prefix in [
                (bi_s, bu_s, torch.unique(si[si > 0]), src_users, 'src'),
                (bi_t, bu_t, torch.unique(ti[ti > 0]), tgt_users, 'tgt'),
            ]:
                if item_ids.numel():
                    pred = self.pop_item_head(item_e[item_ids]).squeeze(-1)
                    loss_pop = loss_pop + F.smooth_l1_loss(pred, getattr(self, f'{prefix}_item_pop')[item_ids])
                user_ids = user_ids[getattr(self, f'{prefix}_user_valid')[user_ids]]
                if user_ids.numel():
                    pred = self.pop_user_head(user_e[user_ids]).squeeze(-1)
                    loss_pop = loss_pop + F.smooth_l1_loss(pred, getattr(self, f'{prefix}_user_pop')[user_ids])

        loss_orth = zero
        if self.lambda_orth > 0:
            loss_orth = (self._orthogonal(pi_s[si], bi_s[si])
                         + self._orthogonal(pi_t[ti], bi_t[ti])
                         + self._orthogonal(pu_s[su], bu_s[su])
                         + self._orthogonal(pu_t[tu], bu_t[tu])) / 4.

        loss_multi = zero
        if self.lambda_multi > 0:
            mm, valid = self._source_mm(su)
            if valid.any():
                loss_multi = (1 - F.cosine_similarity(pu_s[su[valid]], mm[valid], dim=-1)).mean()

        # Unique overlap users avoid same-user false negatives in InfoNCE.
        loss_cl = zero
        if self.lambda_cl > 0:
            overlap = torch.unique(su[(su > 0) & (su < self.overlapped_num_users)])
            if overlap.numel() >= 2:
                source = F.normalize(pu_s[overlap], dim=-1)
                target = F.normalize(pu_t[overlap], dim=-1)
                # Risk proxy: SOURCE observed mean item popularity, not causal dependency.
                with torch.no_grad():
                    w_bias = torch.sigmoid(self.src_user_pop[overlap])
                    if self.beta < 1:
                        mm, mm_valid = self._source_mm(overlap)
                        w_sem = ((1 - F.cosine_similarity(pu_s[overlap], mm, dim=-1)) / 2).clamp(0, 1)
                        # Missing MM -> neutral; do not silently mark these users low risk.
                        w_sem = torch.where(mm_valid, w_sem, torch.full_like(w_sem, 0.5))
                    else:
                        w_sem = torch.zeros_like(w_bias)
                    alpha = (self.beta * w_bias + (1 - self.beta) * w_sem).clamp(0, 1)
                    weights = torch.exp(-self.gamma * alpha)
                logits = (source @ target.T) / self.tau
                labels = torch.arange(overlap.numel(), device=logits.device)
                per_user_loss = F.cross_entropy(logits, labels, reduction='none')
                loss_cl = (per_user_loss * weights).sum() / weights.sum().clamp_min(1e-8)

        # Constant-length tuple for RecBole-CDR trainer logging/summing.
        return (loss_src, loss_tgt, loss_reg,
                self.lambda_pop * loss_pop,
                self.lambda_orth * loss_orth,
                self.lambda_multi * loss_multi,
                self.lambda_cl * loss_cl)

    def init_restore_e(self):
        self.target_restore_pref_u_e = None
        self.target_restore_pref_i_e = None

    def get_restore_e(self):
        if self.target_restore_pref_u_e is None or self.target_restore_pref_i_e is None:
            _, (p_u, p_i, _, _) = self.forward()
            self.target_restore_pref_u_e = p_u
            self.target_restore_pref_i_e = p_i
        return self.target_restore_pref_u_e, self.target_restore_pref_i_e

    def predict(self, interaction):
        u = interaction[self.TARGET_USER_ID]
        i = interaction[self.TARGET_ITEM_ID]
        user_e, item_e = self.get_restore_e()
        return (user_e[u] * item_e[i]).sum(-1)

    def full_sort_predict(self, interaction):
        u = interaction[self.TARGET_USER_ID]
        user_e, item_e = self.get_restore_e()
        scores = []
        for start in range(0, self.target_num_items, self.full_sort_chunk_size):
            end = min(start + self.full_sort_chunk_size, self.target_num_items)
            scores.append(user_e[u] @ item_e[start:end].T)
        return torch.cat(scores, dim=1).reshape(-1)