import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class DGCDRBASE(CrossDomainRecommender):
    """Minimal end-to-end cross-domain GCN baseline."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = int(config["embedding_size"])
        self.n_layers = int(config["n_layers"])
        self.drop_rate = float(config["drop_rate"])
        self.connect_way = str(config["connect_way"]).lower()
        self.full_sort_chunk_size = int(config["full_sort_chunk_size"])
        if self.connect_way not in {"mean", "concat"}:
            raise ValueError("connect_way must be 'mean' or 'concat'.")

        self.source_user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size
        )
        self.source_item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size
        )
        self.target_user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size
        )
        self.target_item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size
        )

        self.dropout = nn.Dropout(self.drop_rate)
        self.bpr_loss = BPRLoss()

        source_matrix = dataset.inter_matrix(
            form="coo", value_field=None, domain="source"
        ).astype(np.float32)
        target_matrix = dataset.inter_matrix(
            form="coo", value_field=None, domain="target"
        ).astype(np.float32)
        self.register_buffer(
            "source_norm_adj",
            self._normalized_adjacency(source_matrix),
            persistent=False,
        )
        self.register_buffer(
            "target_norm_adj",
            self._normalized_adjacency(target_matrix),
            persistent=False,
        )

        self.apply(self._init_weights)
        with torch.no_grad():
            self.source_user_embedding.weight[
                self.overlapped_num_users : self.target_num_users
            ].zero_()
            self.source_item_embedding.weight[
                self.overlapped_num_items : self.target_num_items
            ].zero_()
            self.target_user_embedding.weight[self.target_num_users :].zero_()
            self.target_item_embedding.weight[self.target_num_items :].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.xavier_normal_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _normalized_adjacency(self, interaction_matrix):
        interaction = interaction_matrix.tocoo()
        user = interaction.row.astype(np.int64, copy=False)
        item = interaction.col.astype(np.int64, copy=False) + self.total_num_users
        rows = np.concatenate((user, item))
        cols = np.concatenate((item, user))
        values = np.ones(rows.size, dtype=np.float32)
        adjacency = sp.coo_matrix(
            (values, (rows, cols)),
            shape=(
                self.total_num_users + self.total_num_items,
                self.total_num_users + self.total_num_items,
            ),
        )
        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        inverse_sqrt = np.power(degree + 1e-7, -0.5).astype(np.float32)
        normalized = (sp.diags(inverse_sqrt) @ adjacency @ sp.diags(inverse_sqrt)).tocoo()
        indices = torch.from_numpy(
            np.vstack((normalized.row, normalized.col)).astype(np.int64)
        )
        return torch.sparse_coo_tensor(
            indices,
            torch.from_numpy(normalized.data.astype(np.float32, copy=False)),
            normalized.shape,
        ).coalesce()

    def forward(self):
        source = torch.cat(
            (self.source_user_embedding.weight, self.source_item_embedding.weight)
        )
        target = torch.cat(
            (self.target_user_embedding.weight, self.target_item_embedding.weight)
        )
        source_layers = [source]
        target_layers = [target]
        for _ in range(self.n_layers):
            source_side = torch.sparse.mm(self.source_norm_adj, source)
            target_side = torch.sparse.mm(self.target_norm_adj, target)
            source = self.dropout(source + source_side + source * source_side)
            target = self.dropout(target + target_side + target * target_side)
            source_layers.append(source)
            target_layers.append(target)

        if self.connect_way == "concat":
            source = torch.cat(source_layers, dim=1)
            target = torch.cat(target_layers, dim=1)
        else:
            source = torch.stack(source_layers, dim=1).mean(dim=1)
            target = torch.stack(target_layers, dim=1).mean(dim=1)

        source_user, source_item = torch.split(
            source, (self.total_num_users, self.total_num_items)
        )
        target_user, target_item = torch.split(
            target, (self.total_num_users, self.total_num_items)
        )

        return source_user, source_item, target_user, target_item

    def calculate_loss(self, interaction):
        source_user_all, source_item_all, target_user_all, target_item_all = (
            self.forward()
        )
        source_user = interaction[self.SOURCE_USER_ID]
        source_pos = interaction[self.SOURCE_ITEM_ID]
        source_neg = interaction[self.SOURCE_NEG_ITEM_ID]
        target_user = interaction[self.TARGET_USER_ID]
        target_pos = interaction[self.TARGET_ITEM_ID]
        target_neg = interaction[self.TARGET_NEG_ITEM_ID]

        source_pos_score = (
            source_user_all[source_user] * source_item_all[source_pos]
        ).sum(dim=1)
        source_neg_score = (
            source_user_all[source_user] * source_item_all[source_neg]
        ).sum(dim=1)
        target_pos_score = (
            target_user_all[target_user] * target_item_all[target_pos]
        ).sum(dim=1)
        target_neg_score = (
            target_user_all[target_user] * target_item_all[target_neg]
        ).sum(dim=1)

        return (
            self.bpr_loss(source_pos_score, source_neg_score)
            + self.bpr_loss(target_pos_score, target_neg_score)
        )

    def predict(self, interaction):
        _, _, target_user, target_item = self.forward()
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return (target_user[user] * target_item[item]).sum(dim=1)

    def full_sort_predict(self, interaction):
        _, _, target_user, target_item = self.forward()
        user = target_user[interaction[self.TARGET_USER_ID]]
        item = target_item[: self.target_num_items]
        scores = []
        for start in range(0, self.target_num_items, self.full_sort_chunk_size):
            scores.append(
                user @ item[start : start + self.full_sort_chunk_size].transpose(0, 1)
            )
        return torch.cat(scores, dim=1).reshape(-1)
