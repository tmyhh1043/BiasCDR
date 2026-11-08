"""RecBole-CDR implementation of P2M2-CDR.

The architecture follows the paper and the authors' released implementation:
LightGCN ID embeddings, user review and item text features, domain
disentanglement, Laplace obfuscation, domain-intra/domain-inter contrastive
losses, and an MLP click predictor.
"""

import csv
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.utils import InputType
from recbole_cdr.model.cross_domain_recommender.domain_disentanglement import (
    Domain_Disen,
)
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class _ClickPredictor(nn.Module):
    """Three-layer click predictor used in the released P2M2-CDR code."""

    def __init__(self, embedding_dim):
        super().__init__()
        hidden_dim = max(embedding_dim // 2, 1)
        self.linear_1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.norm_1 = nn.BatchNorm1d(embedding_dim)
        self.linear_2 = nn.Linear(embedding_dim, hidden_dim)
        self.norm_2 = nn.BatchNorm1d(hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, 1)

    def forward(self, user_embedding, item_embedding):
        features = torch.cat([user_embedding, item_embedding], dim=-1)
        features = self.norm_1(F.relu(self.linear_1(features)))
        features = self.norm_2(F.relu(self.linear_2(features)))
        return self.linear_3(features).squeeze(-1)


class P2M2CDR(CrossDomainRecommender):
    """Privacy-preserving multi-modal cross-domain recommender.

    Compared with the previous wrapper, training and evaluation now use the
    same LightGCN representation path.  Contrastive pairs are sampled from
    the actual overlapped-user IDs rather than pairing independently shuffled
    source and target rows.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        self.device = config["device"]
        self.embed_id_dim = int(config["embedding_size"])
        self.domain_disen_dim = int(config["domain_disen_dim"])
        self.n_layers = int(config["n_layers"])
        self.drop_rate = float(config["drop_rate"])
        self.ssl_temp = float(config["ssl_temp"])
        self.alpha = float(config["alpha"])
        self.reg_weight = float(self._config_get(config, "reg_weight", 0.0))
        self.noise_control = float(config["ldp_noise"])
        self.obf = bool(config["obf"])
        self.apply_noise_in_eval = bool(
            self._config_get(config, "apply_noise_in_eval", True)
        )
        self.disen_agg_way = str(config["agg_way"]).lower()
        self.ssl_user_batch_size = int(
            self._config_get(config, "ssl_user_batch_size", 512)
        )
        self.ssl_reduction = str(
            self._config_get(config, "ssl_reduction", "sum")
        ).lower()
        self.full_sort_chunk_size = int(
            self._config_get(config, "full_sort_chunk_size", 256)
        )
        self.strict_multimodal_loading = bool(
            self._config_get(config, "strict_multimodal_loading", True)
        )
        self.use_text_features = bool(
            self._config_get(config, "use_text_features", True)
        )
        self.use_review_features = bool(
            self._config_get(config, "use_review_features", True)
        )
        self.use_visual_features = bool(
            self._config_get(config, "use_visual_features", False)
        )
        self.text_embed_dim = int(
            self._config_get(config, "text_embed_dim", 384)
        )
        self.review_embed_dim = int(
            self._config_get(config, "review_embed_dim", 384)
        )
        self.visual_embed_dim = int(
            self._config_get(config, "visual_embed_dim", 4096)
        )
        self.phase = "BOTH"

        self._validate_configuration()

        self.source_user_embedding = nn.Embedding(
            self.total_num_users, self.embed_id_dim
        )
        self.target_user_embedding = nn.Embedding(
            self.total_num_users, self.embed_id_dim
        )
        self.source_item_embedding = nn.Embedding(
            self.total_num_items, self.embed_id_dim
        )
        self.target_item_embedding = nn.Embedding(
            self.total_num_items, self.embed_id_dim
        )

        source_config = config["source_domain"]
        target_config = config["target_domain"]
        self._register_multimodal_features(
            source_config, target_config, dataset
        )

        self.source_interaction_matrix = dataset.inter_matrix(
            form="coo", value_field=None, domain="source"
        ).astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(
            form="coo", value_field=None, domain="target"
        ).astype(np.float32)
        self.register_buffer(
            "source_norm_adj_matrix",
            self.get_norm_adj_mat(self.source_interaction_matrix),
            persistent=False,
        )
        self.register_buffer(
            "target_norm_adj_matrix",
            self.get_norm_adj_mat(self.target_interaction_matrix),
            persistent=False,
        )

        self.domain_disen = Domain_Disen(
            self.device,
            self.embed_id_dim,
            self.domain_disen_dim,
            self.drop_rate,
        )
        self.source_disen_concat_layer = nn.Linear(
            self.domain_disen_dim * 2, self.domain_disen_dim
        )
        self.target_disen_concat_layer = nn.Linear(
            self.domain_disen_dim * 2, self.domain_disen_dim
        )
        self.source_v_feat_layer = nn.Linear(
            self.embed_id_dim, self.domain_disen_dim
        )
        self.target_v_feat_layer = nn.Linear(
            self.embed_id_dim, self.domain_disen_dim
        )
        self.source_predictor = _ClickPredictor(self.domain_disen_dim)
        self.target_predictor = _ClickPredictor(self.domain_disen_dim)

        self.rec_loss = nn.BCEWithLogitsLoss()
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        self.apply(self._init_weights)
        self._mask_domain_only_embeddings()
        # Evaluation caches are intentionally omitted from checkpoints.
        self.other_parameter_name = []

    @staticmethod
    def _config_get(config, key, default=None):
        if isinstance(config, dict):
            return config[key] if key in config else default
        try:
            return config[key] if key in config else default
        except (KeyError, TypeError):
            return default

    def _validate_configuration(self):
        if self.embed_id_dim <= 0 or self.domain_disen_dim <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        if self.n_layers < 0:
            raise ValueError("n_layers must be non-negative.")
        if self.ssl_temp <= 0:
            raise ValueError("ssl_temp must be positive.")
        if self.ssl_user_batch_size <= 0:
            raise ValueError("ssl_user_batch_size must be positive.")
        if self.full_sort_chunk_size <= 0:
            raise ValueError("full_sort_chunk_size must be positive.")
        if self.disen_agg_way not in {"sum", "avg", "concat"}:
            raise ValueError(
                "agg_way must be one of: sum, avg, concat."
            )
        if self.ssl_reduction not in {"sum", "mean"}:
            raise ValueError("ssl_reduction must be 'sum' or 'mean'.")
        if self.alpha > 0 and self.overlapped_num_users <= 1:
            raise ValueError(
                "P2M2-CDR contrastive learning requires overlapped users."
            )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _mask_domain_only_embeddings(self):
        """Zero embeddings for entities that do not belong to a domain."""
        with torch.no_grad():
            self.source_user_embedding.weight[
                self.overlapped_num_users : self.target_num_users
            ].zero_()
            self.source_item_embedding.weight[
                self.overlapped_num_items : self.target_num_items
            ].zero_()
            self.target_user_embedding.weight[
                self.target_num_users :
            ].zero_()
            self.target_item_embedding.weight[
                self.target_num_items :
            ].zero_()

    @staticmethod
    def _resolve_global_id(remap_dict, original_id):
        for candidate in (original_id, str(original_id)):
            value = remap_dict.get(candidate)
            if value is not None:
                return int(value)
        return None

    @staticmethod
    def _expected_global_ids(remap_dict):
        return {
            int(value)
            for value in remap_dict.values()
            if int(value) != 0
        }

    def _feature_error(self, message, required):
        if required or self.strict_multimodal_loading:
            raise ValueError(message)
        return None

    def _load_mapped_feature(
        self,
        domain_config,
        remap_dict,
        *,
        feature_key,
        map_key,
        id_field_key,
        index_field_key,
        default_id_field,
        default_index_field,
        rows,
        feature_name,
        required,
    ):
        feature_dir = self._config_get(
            domain_config, "multimodal_feat_path"
        )
        feature_file = self._config_get(domain_config, feature_key)
        map_file = self._config_get(domain_config, map_key)
        if not feature_dir or not feature_file:
            return self._feature_error(
                f"{feature_name} feature path is not configured.", required
            )

        feature_path = (
            feature_file
            if os.path.isabs(feature_file)
            else os.path.join(feature_dir, feature_file)
        )
        if not os.path.isfile(feature_path):
            return self._feature_error(
                f"{feature_name} feature file not found: {feature_path}",
                required,
            )

        raw_feature = np.load(feature_path, mmap_mode="r")
        if raw_feature.ndim == 1:
            raw_feature = raw_feature[:, None]
        if raw_feature.ndim != 2:
            return self._feature_error(
                f"{feature_name} must be a 2-D array: {feature_path}",
                required,
            )
        if not np.isfinite(raw_feature).all():
            return self._feature_error(
                f"{feature_name} contains NaN or Inf: {feature_path}",
                required,
            )

        if not map_file:
            return self._feature_error(
                f"{feature_name} ID map is not configured.", required
            )
        map_path = (
            map_file
            if os.path.isabs(map_file)
            else os.path.join(feature_dir, map_file)
        )
        if not os.path.isfile(map_path):
            return self._feature_error(
                f"{feature_name} ID map not found: {map_path}", required
            )

        id_field = self._config_get(
            domain_config, id_field_key, default_id_field
        )
        index_field = self._config_get(
            domain_config, index_field_key, default_index_field
        )
        aligned_feature = np.zeros(
            (rows, raw_feature.shape[1]), dtype=np.float32
        )
        mapped_ids = set()
        with open(map_path, "r", encoding="utf-8") as mapping_handle:
            reader = csv.DictReader(mapping_handle)
            fieldnames = set(reader.fieldnames or [])
            if id_field not in fieldnames or index_field not in fieldnames:
                return self._feature_error(
                    f"{map_path} must contain columns "
                    f"[{id_field}] and [{index_field}].",
                    required,
                )
            for row in reader:
                global_id = self._resolve_global_id(
                    remap_dict, row[id_field]
                )
                if global_id is None:
                    continue
                try:
                    feature_index = int(row[index_field])
                except (TypeError, ValueError):
                    continue
                if (
                    0 <= feature_index < raw_feature.shape[0]
                    and 0 <= global_id < rows
                ):
                    aligned_feature[global_id] = raw_feature[feature_index]
                    if global_id:
                        mapped_ids.add(global_id)

        missing_ids = self._expected_global_ids(remap_dict) - mapped_ids
        if missing_ids and self.strict_multimodal_loading:
            sample = sorted(missing_ids)[:5]
            raise ValueError(
                f"{feature_name} is missing {len(missing_ids)} mapped "
                f"entities; sample global IDs: {sample}"
            )
        aligned_feature[0] = 0
        return torch.from_numpy(aligned_feature)

    def _load_item_feature(
        self, domain_config, remap_dict, modality, rows, required
    ):
        default_id_field = self._config_get(
            domain_config, "ITEM_ID_FIELD", "item_id"
        )
        return self._load_mapped_feature(
            domain_config,
            remap_dict,
            feature_key=f"{modality}_feat_file",
            map_key="item_map_file",
            id_field_key="item_feat_id_field",
            index_field_key="item_feat_index_field",
            default_id_field=default_id_field,
            default_index_field="itemID",
            rows=rows,
            feature_name=f"{modality} item",
            required=required,
        )

    def _load_review_feature(
        self, domain_config, remap_dict, rows, required
    ):
        default_id_field = self._config_get(
            domain_config, "USER_ID_FIELD", "user_id"
        )
        return self._load_mapped_feature(
            domain_config,
            remap_dict,
            feature_key="review_feat_file",
            map_key="user_map_file",
            id_field_key="user_feat_id_field",
            index_field_key="user_feat_index_field",
            default_id_field=default_id_field,
            default_index_field="userID",
            rows=rows,
            feature_name="review user",
            required=required,
        )

    def _register_feature(self, name, feature, expected_dim):
        if feature is None:
            self.register_buffer(name, None, persistent=False)
            return
        if feature.size(1) != expected_dim:
            raise ValueError(
                f"{name} dimension is {feature.size(1)}, but the "
                f"configured dimension is {expected_dim}."
            )
        self.register_buffer(
            name, feature.to(self.device), persistent=False
        )

    def _register_multimodal_features(
        self, source_config, target_config, dataset
    ):
        source_text = target_text = None
        source_review = target_review = None
        source_visual = target_visual = None

        if self.use_text_features:
            source_text = self._load_item_feature(
                source_config,
                dataset.source_item_ID_remap_dict,
                "text",
                self.total_num_items,
                required=True,
            )
            target_text = self._load_item_feature(
                target_config,
                dataset.target_item_ID_remap_dict,
                "text",
                self.total_num_items,
                required=True,
            )
        if self.use_review_features:
            source_review = self._load_review_feature(
                source_config,
                dataset.source_user_ID_remap_dict,
                self.total_num_users,
                required=True,
            )
            target_review = self._load_review_feature(
                target_config,
                dataset.target_user_ID_remap_dict,
                self.total_num_users,
                required=True,
            )
        if self.use_visual_features:
            source_visual = self._load_item_feature(
                source_config,
                dataset.source_item_ID_remap_dict,
                "visual",
                self.total_num_items,
                required=True,
            )
            target_visual = self._load_item_feature(
                target_config,
                dataset.target_item_ID_remap_dict,
                "visual",
                self.total_num_items,
                required=True,
            )

        self._register_feature(
            "source_text_feature", source_text, self.text_embed_dim
        )
        self._register_feature(
            "target_text_feature", target_text, self.text_embed_dim
        )
        self._register_feature(
            "source_review_feature",
            source_review,
            self.review_embed_dim,
        )
        self._register_feature(
            "target_review_feature",
            target_review,
            self.review_embed_dim,
        )
        self._register_feature(
            "source_visual_feature",
            source_visual,
            self.visual_embed_dim,
        )
        self._register_feature(
            "target_visual_feature",
            target_visual,
            self.visual_embed_dim,
        )

        if self.use_text_features:
            self.source_text_feat_layer = nn.Linear(
                self.text_embed_dim, self.embed_id_dim
            )
            self.target_text_feat_layer = nn.Linear(
                self.text_embed_dim, self.embed_id_dim
            )
        if self.use_review_features:
            self.source_review_feat_layer = nn.Linear(
                self.review_embed_dim, self.embed_id_dim
            )
            self.target_review_feat_layer = nn.Linear(
                self.review_embed_dim, self.embed_id_dim
            )
        if self.use_visual_features:
            self.source_visual_feat_layer = nn.Linear(
                self.visual_embed_dim, self.embed_id_dim
            )
            self.target_visual_feat_layer = nn.Linear(
                self.visual_embed_dim, self.embed_id_dim
            )

    def get_norm_adj_mat(self, interaction_matrix):
        """Build the symmetric normalized user-item adjacency matrix."""
        n_users = self.total_num_users
        n_items = self.total_num_items
        interaction = interaction_matrix.tocoo()
        row = np.concatenate(
            [interaction.row, interaction.col + n_users]
        )
        col = np.concatenate(
            [interaction.col + n_users, interaction.row]
        )
        adjacency = sp.coo_matrix(
            (np.ones(row.size, dtype=np.float32), (row, col)),
            shape=(n_users + n_items, n_users + n_items),
        )
        adjacency.sum_duplicates()
        adjacency.data[:] = 1.0
        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        degree_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        nonzero = degree > 0
        degree_inv_sqrt[nonzero] = np.power(
            degree[nonzero], -0.5
        )
        normalized = (
            sp.diags(degree_inv_sqrt)
            .dot(adjacency)
            .dot(sp.diags(degree_inv_sqrt))
            .tocoo()
        )
        indices = torch.from_numpy(
            np.vstack([normalized.row, normalized.col]).astype(np.int64)
        )
        values = torch.from_numpy(
            normalized.data.astype(np.float32, copy=False)
        )
        return torch.sparse_coo_tensor(
            indices, values, normalized.shape, dtype=torch.float32
        ).coalesce()

    def _propagate(self, domain):
        """Run pure LightGCN propagation on ID embeddings."""
        if domain == "source":
            user_weight = self.source_user_embedding.weight
            item_weight = self.source_item_embedding.weight
            adjacency = self.source_norm_adj_matrix
        elif domain == "target":
            user_weight = self.target_user_embedding.weight
            item_weight = self.target_item_embedding.weight
            adjacency = self.target_norm_adj_matrix
        else:
            raise ValueError(f"Unknown domain: {domain}")

        all_embedding = torch.cat([user_weight, item_weight], dim=0)
        layer_embeddings = [all_embedding]
        for _ in range(self.n_layers):
            all_embedding = torch.sparse.mm(adjacency, all_embedding)
            layer_embeddings.append(all_embedding)
        final_embedding = torch.stack(
            layer_embeddings, dim=1
        ).mean(dim=1)
        user_embedding, item_embedding = torch.split(
            final_embedding,
            [self.total_num_users, self.total_num_items],
            dim=0,
        )
        # This residual is present in the authors' released implementation.
        return user_embedding, item_embedding + item_weight

    def _compute_gcn(self):
        source_user, source_item = self._propagate("source")
        target_user, target_item = self._propagate("target")
        return source_user, source_item, target_user, target_item

    def _compose_user_features(
        self, id_embedding, user_ids, domain
    ):
        user_embedding = id_embedding
        if not self.use_review_features:
            return user_embedding
        if domain == "source":
            review = self.source_review_feature[user_ids]
            layer = self.source_review_feat_layer
        else:
            review = self.target_review_feature[user_ids]
            layer = self.target_review_feat_layer
        return user_embedding + F.relu(layer(review))

    def _compose_item_features(
        self, id_embedding, item_ids, domain
    ):
        item_embedding = id_embedding
        if self.use_text_features:
            if domain == "source":
                text = self.source_text_feature[item_ids]
                layer = self.source_text_feat_layer
            else:
                text = self.target_text_feature[item_ids]
                layer = self.target_text_feat_layer
            item_embedding = item_embedding + F.relu(layer(text))
        if self.use_visual_features:
            if domain == "source":
                visual = self.source_visual_feature[item_ids]
                layer = self.source_visual_feat_layer
            else:
                visual = self.target_visual_feature[item_ids]
                layer = self.target_visual_feat_layer
            item_embedding = item_embedding + F.relu(layer(visual))
        return item_embedding

    def _aggregate_user_factors(
        self, common, specific, domain
    ):
        if self.disen_agg_way == "sum":
            return common + specific
        if self.disen_agg_way == "avg":
            return (common + specific) / 2
        layer = (
            self.source_disen_concat_layer
            if domain == "source"
            else self.target_disen_concat_layer
        )
        return F.relu(layer(torch.cat([common, specific], dim=-1)))

    def _laplace_noise(self, reference):
        if not self.obf or self.noise_control <= 0:
            return torch.zeros_like(reference)
        first = torch.empty_like(reference).exponential_()
        second = torch.empty_like(reference).exponential_()
        return (first - second) * self.noise_control

    def _domain_user_features(self, user_embedding, domain):
        common, common_aug, specific, specific_aug = (
            self.domain_disen.forward_domain(user_embedding, domain)
        )
        use_noise = (
            self.obf
            and self.noise_control > 0
            and (self.training or self.apply_noise_in_eval)
        )
        if use_noise:
            # The released implementation shares one draw across both factors
            # and their augmented views for each domain/user.
            noise = self._laplace_noise(common)
            common = common + noise
            common_aug = common_aug + noise
            specific = specific + noise
            specific_aug = specific_aug + noise
        factors = (common, common_aug, specific, specific_aug)
        final_user = self._aggregate_user_factors(
            common, specific, domain
        )
        return factors, final_user

    def _project_items(self, item_embedding, domain):
        layer = (
            self.source_v_feat_layer
            if domain == "source"
            else self.target_v_feat_layer
        )
        return F.relu(layer(item_embedding))

    @staticmethod
    def _cosine_logits(left, right, temperature):
        left = F.normalize(left, p=2, dim=-1)
        right = F.normalize(right, p=2, dim=-1)
        return (left * right).sum(dim=-1) / temperature

    @staticmethod
    def _contrastive_ratio_loss(
        positive_logits, negative_logits, reduction
    ):
        positive_log_mass = torch.logsumexp(
            positive_logits, dim=-1
        )
        all_log_mass = torch.logsumexp(
            torch.cat([positive_logits, negative_logits], dim=-1),
            dim=-1,
        )
        loss = all_log_mass - positive_log_mass
        return loss.sum() if reduction == "sum" else loss.mean()

    def _intra_domain_loss(self, factors):
        common, common_aug, specific, specific_aug = factors
        positive_logits = torch.stack(
            [
                self._cosine_logits(
                    common, common_aug, self.ssl_temp
                ),
                self._cosine_logits(
                    specific, specific_aug, self.ssl_temp
                ),
            ],
            dim=-1,
        )
        negative_logits = torch.stack(
            [
                self._cosine_logits(common, specific, self.ssl_temp),
                self._cosine_logits(
                    common, specific_aug, self.ssl_temp
                ),
                self._cosine_logits(
                    specific, common_aug, self.ssl_temp
                ),
                self._cosine_logits(
                    specific_aug, common_aug, self.ssl_temp
                ),
            ],
            dim=-1,
        )
        return self._contrastive_ratio_loss(
            positive_logits, negative_logits, self.ssl_reduction
        )

    def _inter_domain_loss(self, source_factors, target_factors):
        source_common, _, source_specific, _ = source_factors
        target_common, _, target_specific, _ = target_factors
        positive_logits = self._cosine_logits(
            source_common, target_common, self.ssl_temp
        ).unsqueeze(-1)
        negative_logits = torch.stack(
            [
                self._cosine_logits(
                    source_common, target_specific, self.ssl_temp
                ),
                self._cosine_logits(
                    target_common, source_specific, self.ssl_temp
                ),
                self._cosine_logits(
                    source_specific, target_specific, self.ssl_temp
                ),
            ],
            dim=-1,
        )
        return self._contrastive_ratio_loss(
            positive_logits, negative_logits, self.ssl_reduction
        )

    def _sample_overlapped_users(self):
        """Sample aligned source/target rows using the same global user IDs."""
        return torch.randint(
            low=1,
            high=self.overlapped_num_users,
            size=(self.ssl_user_batch_size,),
            device=self.source_user_embedding.weight.device,
        )

    def _ssl_loss(self, source_user_all, target_user_all):
        if self.alpha == 0:
            return None
        overlap_users = self._sample_overlapped_users()
        source_user = self._compose_user_features(
            source_user_all[overlap_users], overlap_users, "source"
        )
        target_user = self._compose_user_features(
            target_user_all[overlap_users], overlap_users, "target"
        )
        source_factors, _ = self._domain_user_features(
            source_user, "source"
        )
        target_factors, _ = self._domain_user_features(
            target_user, "target"
        )
        contrastive_loss = (
            self._intra_domain_loss(source_factors)
            + self._intra_domain_loss(target_factors)
            + self._inter_domain_loss(source_factors, target_factors)
        )
        return self.alpha * contrastive_loss

    def _score_pairs(self, user_embedding, item_embedding, domain):
        predictor = (
            self.source_predictor
            if domain == "source"
            else self.target_predictor
        )
        return predictor(user_embedding, item_embedding)

    def _regularization_loss(self, domain, user_ids, item_ids):
        if self.reg_weight == 0:
            return self.source_user_embedding.weight.new_zeros(())
        if domain == "source":
            user = self.source_user_embedding(user_ids)
            item = self.source_item_embedding(item_ids)
        else:
            user = self.target_user_embedding(user_ids)
            item = self.target_item_embedding(item_ids)
        return self.reg_weight * 0.5 * (
            user.square().sum(dim=-1).mean()
            + item.square().sum(dim=-1).mean()
        )

    def set_phase(self, phase):
        self.phase = phase

    def calculate_loss(self, interaction):
        self.init_restore_e()
        source_user_all, source_item_all, target_user_all, target_item_all = (
            self._compute_gcn()
        )
        need_source = self.phase.upper() in {
            "SOURCE",
            "BOTH",
            "OVERLAP",
        }
        need_target = self.phase.upper() in {
            "TARGET",
            "BOTH",
            "OVERLAP",
        }
        losses = []

        if need_source:
            source_user_ids = interaction[self.SOURCE_USER_ID]
            source_item_ids = interaction[self.SOURCE_ITEM_ID]
            source_label = interaction[self.SOURCE_LABEL].float()
            source_user = self._compose_user_features(
                source_user_all[source_user_ids],
                source_user_ids,
                "source",
            )
            _, source_user = self._domain_user_features(
                source_user, "source"
            )
            source_item = self._compose_item_features(
                source_item_all[source_item_ids],
                source_item_ids,
                "source",
            )
            source_item = self._project_items(source_item, "source")
            source_logits = self._score_pairs(
                source_user, source_item, "source"
            )
            losses.append(
                self.rec_loss(source_logits, source_label)
                + self._regularization_loss(
                    "source", source_user_ids, source_item_ids
                )
            )

        if need_target:
            target_user_ids = interaction[self.TARGET_USER_ID]
            target_item_ids = interaction[self.TARGET_ITEM_ID]
            target_label = interaction[self.TARGET_LABEL].float()
            target_user = self._compose_user_features(
                target_user_all[target_user_ids],
                target_user_ids,
                "target",
            )
            _, target_user = self._domain_user_features(
                target_user, "target"
            )
            target_item = self._compose_item_features(
                target_item_all[target_item_ids],
                target_item_ids,
                "target",
            )
            target_item = self._project_items(target_item, "target")
            target_logits = self._score_pairs(
                target_user, target_item, "target"
            )
            losses.append(
                self.rec_loss(target_logits, target_label)
                + self._regularization_loss(
                    "target", target_user_ids, target_item_ids
                )
            )

        ssl_loss = self._ssl_loss(source_user_all, target_user_all)
        if ssl_loss is not None:
            losses.append(ssl_loss)
        return tuple(losses)

    def _all_domain_representations(self, domain):
        user_id_embedding, item_id_embedding = self._propagate(domain)
        user_ids = torch.arange(
            self.total_num_users,
            device=user_id_embedding.device,
        )
        item_ids = torch.arange(
            self.total_num_items,
            device=item_id_embedding.device,
        )
        user_embedding = self._compose_user_features(
            user_id_embedding, user_ids, domain
        )
        _, user_embedding = self._domain_user_features(
            user_embedding, domain
        )
        item_embedding = self._compose_item_features(
            item_id_embedding, item_ids, domain
        )
        item_embedding = self._project_items(item_embedding, domain)
        return user_embedding, item_embedding

    def forward(self):
        source_user, source_item = self._all_domain_representations(
            "source"
        )
        target_user, target_item = self._all_domain_representations(
            "target"
        )
        return source_user, source_item, target_user, target_item

    def predict(self, interaction):
        user_ids = interaction[self.TARGET_USER_ID]
        item_ids = interaction[self.TARGET_ITEM_ID]
        user_embedding, item_embedding = self.get_restore_e()
        return self._score_pairs(
            user_embedding[user_ids],
            item_embedding[item_ids],
            "target",
        )

    def _full_sort_scores(
        self, user_embedding, item_embedding, domain
    ):
        score_chunks = []
        for start in range(
            0, item_embedding.size(0), self.full_sort_chunk_size
        ):
            item_chunk = item_embedding[
                start : start + self.full_sort_chunk_size
            ]
            user_grid = user_embedding[:, None, :].expand(
                -1, item_chunk.size(0), -1
            )
            item_grid = item_chunk[None, :, :].expand(
                user_embedding.size(0), -1, -1
            )
            chunk_scores = self._score_pairs(
                user_grid.reshape(-1, user_embedding.size(-1)),
                item_grid.reshape(-1, item_embedding.size(-1)),
                domain,
            )
            score_chunks.append(
                chunk_scores.view(
                    user_embedding.size(0), item_chunk.size(0)
                )
            )
        return torch.cat(score_chunks, dim=1)

    def full_sort_predict(self, interaction):
        user_ids = interaction[self.TARGET_USER_ID]
        user_embedding, item_embedding = self.get_restore_e()
        scores = self._full_sort_scores(
            user_embedding[user_ids],
            item_embedding[: self.target_num_items],
            "target",
        )
        return scores.reshape(-1)

    def load_state_dict(self, state_dict, strict=True):
        """Load checkpoint weights without retaining a stale eval cache."""
        load_result = super().load_state_dict(state_dict, strict=strict)
        self.init_restore_e()
        return load_result

    def init_restore_e(self):
        self.target_restore_user_e = None
        self.target_restore_item_e = None

    def get_restore_e(self):
        if (
            self.target_restore_user_e is None
            or self.target_restore_item_e is None
        ):
            (
                self.target_restore_user_e,
                self.target_restore_item_e,
            ) = self._all_domain_representations("target")
        return self.target_restore_user_e, self.target_restore_item_e


__all__ = ["P2M2CDR"]
