import sys

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


def config_get(config, key, default=None):
    try:
        value = config[key]
    except (KeyError, TypeError):
        return default
    return default if value is None else value


class Adapter(nn.Module):
    """Preference/Bias 用の非線形 Adapter。"""

    def __init__(self, in_size, out_size, small_output=False):
        super().__init__()
        # 幅を縮めず、Phase 1 の情報を通せるようにする。
        bottleneck_size = max(in_size, out_size, 64)
        self.layers = nn.Sequential(
            nn.Linear(in_size, bottleneck_size, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck_size, out_size, bias=False),
        )
        nn.init.xavier_normal_(self.layers[0].weight)
        if small_output:
            nn.init.normal_(self.layers[2].weight, mean=0.0, std=0.01)
        else:
            nn.init.xavier_normal_(self.layers[2].weight)

    def forward(self, values):
        return self.layers(values)


class DGCDRBASE(CrossDomainRecommender):
    """Phase 1: ソース・ターゲット両ドメインの GCN を学習する。"""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embedding_size = int(config_get(config, "embedding_size", 64))
        self.n_layers = int(config_get(config, "n_layers", 3))
        self.drop_rate = float(config_get(config, "drop_rate", 0.1))
        self.connect_way = str(config_get(config, "connect_way", "mean")).lower()
        self.phase = "BOTH"

        if self.connect_way not in {"mean", "concat"}:
            raise ValueError("connect_way must be 'mean' or 'concat'")
        if not 0 <= self.drop_rate < 1:
            raise ValueError("drop_rate must be in [0, 1)")

        self.gcn_output_size = (
            self.embedding_size * (self.n_layers + 1)
            if self.connect_way == "concat"
            else self.embedding_size
        )

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

        for embedding in (
            self.source_user_embedding,
            self.source_item_embedding,
            self.target_user_embedding,
            self.target_item_embedding,
        ):
            nn.init.xavier_normal_(embedding.weight)

        self.dropout = nn.Dropout(self.drop_rate)
        self.bpr_loss = BPRLoss()
        self.other_parameter_name = []

        for domain in ("source", "target"):
            interaction = dataset.inter_matrix(
                form="coo", value_field=None, domain=domain
            ).astype(np.float32)
            self.register_buffer(
                f"{domain}_norm_adj",
                self._normalized_adjacency(interaction),
                persistent=False,
            )

    def _normalized_adjacency(self, interaction):
        interaction = interaction.tocoo()
        n_users = self.total_num_users
        n_items = self.total_num_items

        rows = np.concatenate(
            (interaction.row, interaction.col + n_users)
        )
        cols = np.concatenate(
            (interaction.col + n_users, interaction.row)
        )
        adjacency = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_users + n_items, n_users + n_items),
        ).tocsr()
        adjacency.sum_duplicates()
        adjacency.data[:] = 1.0

        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        inv_sqrt = np.power(degree + 1e-7, -0.5)
        normalized = (
            sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)
        ).tocoo()

        indices = torch.from_numpy(
            np.stack((normalized.row, normalized.col)).astype(np.int64)
        )
        values = torch.from_numpy(
            normalized.data.astype(np.float32)
        )
        return torch.sparse_coo_tensor(
            indices, values, normalized.shape
        ).coalesce()

    def _graph_layer(self, adjacency, embeddings):
        # DGCDR の伝播規則: 近傍項・要素積・自己項を結合する。
        side = torch.sparse.mm(adjacency, embeddings)
        return self.dropout(
            embeddings + side + embeddings * side
        )

    def _propagate(self, domain):
        users = getattr(self, f"{domain}_user_embedding").weight
        items = getattr(self, f"{domain}_item_embedding").weight
        adjacency = getattr(self, f"{domain}_norm_adj")

        embeddings = torch.cat((users, items), dim=0)
        layers = [embeddings]
        for _ in range(self.n_layers):
            embeddings = self._graph_layer(adjacency, embeddings)
            layers.append(embeddings)

        if self.connect_way == "concat":
            combined = torch.cat(layers, dim=1)
        else:
            combined = torch.stack(layers, dim=0).mean(dim=0)

        return torch.split(
            combined,
            (self.total_num_users, self.total_num_items),
            dim=0,
        )

    def forward(self):
        source_user, source_item = self._propagate("source")
        target_user, target_item = self._propagate("target")
        return source_user, source_item, target_user, target_item

    def set_phase(self, phase):
        self.phase = str(phase).upper()

    def _domain_bpr(self, interaction, domain, user_vectors, item_vectors):
        if domain == "source":
            users = interaction[self.SOURCE_USER_ID]
            positive = interaction[self.SOURCE_ITEM_ID]
            negative = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positive = interaction[self.TARGET_ITEM_ID]
            negative = interaction[self.TARGET_NEG_ITEM_ID]

        positive_score = (
            user_vectors[users] * item_vectors[positive]
        ).sum(dim=1)
        negative_score = (
            user_vectors[users] * item_vectors[negative]
        ).sum(dim=1)
        return self.bpr_loss(positive_score, negative_score)

    def calculate_loss(self, interaction):
        source_user, source_item, target_user, target_item = self.forward()
        if self.phase == "SOURCE":
            return self._domain_bpr(
                interaction, "source", source_user, source_item
            )
        if self.phase == "TARGET":
            return self._domain_bpr(
                interaction, "target", target_user, target_item
            )
        return self._domain_bpr(
            interaction, "source", source_user, source_item
        ) + self._domain_bpr(
            interaction, "target", target_user, target_item
        )

    def predict(self, interaction):
        _, _, target_user, target_item = self.forward()
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        return (target_user[users] * target_item[items]).sum(dim=1)

    def full_sort_predict(self, interaction):
        _, _, target_user, target_item = self.forward()
        users = interaction[self.TARGET_USER_ID]
        scores = (
            target_user[users]
            @ target_item[: self.target_num_items].transpose(0, 1)
        )
        return scores.reshape(-1)


class Proposed(DGCDRBASE):
    """Phase 2: 固定した GCN 表現を嗜好とアイテム人気度に分解する。"""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.lambda_preference_rank = float(
            config_get(config, "lambda_preference_rank", 1.0)
        )
        self.lambda_bias = float(config_get(config, "lambda_bias", 0.1))
        self.lambda_distill = float(config_get(config, "lambda_distill", 1.0))
        self.bias_warmup_epochs = int(config_get(config, "bias_warmup_epochs", 3))
        self.distill_epochs = int(config_get(config, "distill_epochs", 5))
        self.distill_temperature = float(config_get(config, "distill_temperature", 1.0))
        self.bias_sample_size = int(config_get(config, "bias_sample_size", 4096))
        self.preference_negative_mode = str(
            config_get(config, "preference_negative_mode", "mixed")
        ).lower()
        self.matched_negative_weight = float(
            config_get(config, "matched_negative_weight", 0.5)
        )
        self.observation_modulation_strength = float(
            config_get(config, "observation_modulation_strength", 0.25)
        )
        self.distill_schedule = str(
            config_get(config, "distill_schedule", "constant")
        ).lower()
        self.inference_score_mode = str(
            config_get(config, "inference_score_mode", "popularity_subtract")
        ).lower()
        self.inference_popularity_alpha = float(
            config_get(config, "inference_popularity_alpha", 0.0)
        )
        self.counterfactual_reference = float(
            config_get(config, "counterfactual_reference", 0.5)
        )
        if min(self.bias_warmup_epochs, self.distill_epochs) < 0:
            raise ValueError("Warmup epoch counts must be nonnegative")
        if self.distill_temperature <= 0 or self.bias_sample_size < 1:
            raise ValueError("Distillation temperature and bias sample size must be positive")
        if self.preference_negative_mode not in {"matched", "uniform", "mixed"}:
            raise ValueError("preference_negative_mode must be matched, uniform or mixed")
        if not 0 <= self.matched_negative_weight <= 1:
            raise ValueError("matched_negative_weight must be in [0, 1]")
        if not 0 <= self.observation_modulation_strength <= 1:
            raise ValueError("observation_modulation_strength must be in [0, 1]")
        if self.distill_schedule not in {"constant", "linear"}:
            raise ValueError("distill_schedule must be constant or linear")
        if self.inference_score_mode not in {
            "baseline", "preference", "popularity_subtract", "counterfactual"
        }:
            raise ValueError("Unsupported inference_score_mode")
        if self.inference_popularity_alpha < 0:
            raise ValueError("inference_popularity_alpha must be nonnegative")
        self.current_epoch = 0
        self._sampling_stats = {"matched": 0, "fallback": 0}
        self._known_positive = {}
        self._popularity_bins = {}
        self._popularity_pools = {}

        # 嗜好はユーザーとアイテム、人気度バイアスはアイテムから抽出する。
        self.preference_adapter = nn.ModuleDict()
        self.bias_adapter = nn.ModuleDict()
        self.popularity_head = nn.ModuleDict()
        for domain in ("source", "target"):
            for entity in ("user", "item"):
                key = f"{domain}_{entity}"
                self.preference_adapter[key] = Adapter(
                    self.gcn_output_size, self.embedding_size, small_output=True
                )
            self.bias_adapter[domain] = Adapter(
                self.gcn_output_size, self.embedding_size, small_output=True
            )
            self.popularity_head[domain] = nn.Linear(self.embedding_size, 1)
            nn.init.xavier_normal_(self.popularity_head[domain].weight)
            nn.init.zeros_(self.popularity_head[domain].bias)

            interaction = dataset.inter_matrix(
                form="coo", value_field=None, domain=domain
            )
            counts = np.bincount(
                interaction.col, minlength=self.total_num_items
            ).astype(np.float32)
            self.register_buffer(
                f"{domain}_item_log_popularity",
                torch.from_numpy(np.log1p(counts)),
                persistent=False,
            )
            observed_items = np.flatnonzero(counts > 0).astype(np.int64)
            self.register_buffer(
                f"{domain}_observed_item_ids",
                torch.from_numpy(observed_items),
                persistent=False,
            )
            # 人気度帯を事前構築し、嗜好用の負例は各行1回だけ抽出する。
            bins = np.floor(np.log2(counts + 1)).astype(np.int64)
            self._popularity_bins[domain] = bins
            self._popularity_pools[domain] = {
                int(bin_id): observed_items[bins[observed_items] == bin_id]
                for bin_id in np.unique(bins[observed_items])
            }
            self._known_positive[domain] = interaction.tocsr()

        # Phase 2 では Phase 1 の GCN 埋め込みを固定する。
        self._freeze_backbone()
        for name in (
            "source_z_user", "source_z_item", "target_z_user", "target_z_item"
        ):
            self.register_buffer(name, torch.empty(0), persistent=False)
        self._cache_ready = False

    def _freeze_backbone(self):
        for domain in ("source", "target"):
            for entity in ("user", "item"):
                getattr(self, f"{domain}_{entity}_embedding").requires_grad_(False)

    def backbone_is_frozen(self):
        return all(
            not parameter.requires_grad
            for domain in ("source", "target")
            for entity in ("user", "item")
            for parameter in getattr(
                self, f"{domain}_{entity}_embedding"
            ).parameters()
        )

    def validate_pretrained_checkpoint(self, checkpoint):
        if checkpoint is None:
            raise ValueError("A trained DGCDRBASE checkpoint is required")
        checkpoint_config = checkpoint.get("config")
        if checkpoint_config is None:
            raise ValueError("Phase 1 checkpoint has no config")
        if str(config_get(checkpoint_config, "model", "")).upper() != "DGCDRBASE":
            raise ValueError("Phase 1 checkpoint must be DGCDRBASE")
        for key, expected in (
            ("embedding_size", self.embedding_size),
            ("n_layers", self.n_layers),
            ("connect_way", self.connect_way),
        ):
            actual = config_get(checkpoint_config, key)
            if actual is not None and str(actual).lower() != str(expected).lower():
                raise ValueError(
                    f"Phase 1 {key} mismatch: {actual!r} != {expected!r}"
                )

    def _refresh_gcn_cache(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            source_user, source_item, target_user, target_item = (
                DGCDRBASE.forward(self)
            )
            self.source_z_user = source_user.detach()
            self.source_z_item = source_item.detach()
            self.target_z_user = target_user.detach()
            self.target_z_item = target_item.detach()
        self.train(was_training)
        self._cache_ready = True

    def load_pretrained_backbone_checkpoint(self, checkpoint, path=None):
        self.validate_pretrained_checkpoint(checkpoint)
        state = checkpoint["state_dict"]
        with torch.no_grad():
            for domain in ("source", "target"):
                for entity in ("user", "item"):
                    name = f"{domain}_{entity}_embedding"
                    weight = getattr(self, name).weight
                    saved = state[f"{name}.weight"]
                    if weight.shape != saved.shape:
                        raise ValueError(f"Phase 1 shape mismatch: {name}")
                    weight.copy_(saved.to(weight.device))
        self._freeze_backbone()
        self._refresh_gcn_cache()
        return path

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._freeze_backbone()
        self._refresh_gcn_cache()
        return result

    def phase2_parameters(self):
        for module in (
            self.preference_adapter, self.bias_adapter, self.popularity_head
        ):
            yield from module.parameters()

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)
        self._sampling_stats = {"matched": 0, "fallback": 0}

    def sampling_stats(self):
        return dict(self._sampling_stats)

    def _matched_negative(self, domain, users, positive, fallback):
        """同じ人気度帯から1回抽出し、既知の正例なら既存負例を使う。"""
        if self.preference_negative_mode == "uniform":
            return fallback
        user_ids = users.detach().cpu().numpy()
        positive_ids = positive.detach().cpu().numpy()
        chosen = fallback.detach().cpu().numpy().copy()
        bins = self._popularity_bins[domain][positive_ids]
        matched = 0
        for bin_id in np.unique(bins):
            rows = np.flatnonzero(bins == bin_id)
            pool = self._popularity_pools[domain].get(int(bin_id))
            if pool is None or len(pool) < 2:
                continue
            candidates = pool[np.random.randint(len(pool), size=len(rows))]
            known = np.asarray(
                self._known_positive[domain][user_ids[rows], candidates]
            ).reshape(-1)
            valid = (candidates != positive_ids[rows]) & (known == 0)
            chosen[rows[valid]] = candidates[valid]
            matched += int(valid.sum())
        self._sampling_stats["matched"] += matched
        self._sampling_stats["fallback"] += len(chosen) - matched
        return torch.as_tensor(chosen, dtype=positive.dtype, device=positive.device)

    def _domain_bias_loss(self, domain, item_bias):
        # アイテムを一様抽出し、人気正例の出現頻度でMSEが偏らないようにする。
        observed = getattr(self, f"{domain}_observed_item_ids")
        if observed.numel() == 0:
            return item_bias.sum() * 0.0
        sample = torch.randint(
            observed.numel(),
            (min(self.bias_sample_size, observed.numel()),),
            device=observed.device,
        )
        items = observed[sample]
        prediction = self.popularity_head[domain](item_bias[items]).squeeze(-1)
        truth = getattr(self, f"{domain}_item_log_popularity")[items]
        return F.mse_loss(prediction, truth)

    def _distill_weight(self):
        rank_epoch = self.current_epoch - self.bias_warmup_epochs
        if rank_epoch < 0 or self.lambda_distill <= 0:
            return 0.0
        if self.distill_schedule == "constant":
            return self.lambda_distill
        if self.distill_epochs == 0 or rank_epoch >= self.distill_epochs:
            return 0.0
        return self.lambda_distill * (1.0 - rank_epoch / self.distill_epochs)

    def _pref_score(self, domain, users, items, representations):
        # Phase 1 の内積に小さなAdapter補正を加える。
        base_users = getattr(self, f"{domain}_z_user")[users]
        base_items = getattr(self, f"{domain}_z_item")[items]
        adapter_users = representations[f"{domain}_user_preference"][users]
        adapter_items = representations[f"{domain}_item_preference"][items]
        return (base_users * base_items).sum(dim=1) + (
            adapter_users * adapter_items
        ).sum(dim=1)

    def forward(self, bias_only=False):
        if not self._cache_ready:
            raise RuntimeError("Load the DGCDRBASE checkpoint first")
        representations = {}
        for domain in ("source", "target"):
            if not bias_only:
                for entity in ("user", "item"):
                    key = f"{domain}_{entity}"
                    z = getattr(self, f"{domain}_z_{entity}")
                    representations[f"{key}_preference"] = (
                        self.preference_adapter[key](z)
                    )
            item_z = getattr(self, f"{domain}_z_item")
            representations[f"{domain}_item_bias"] = (
                self.bias_adapter[domain](item_z)
            )
        return representations

    def _domain_loss(self, interaction, domain, representations):
        if domain == "source":
            users = interaction[self.SOURCE_USER_ID]
            positive = interaction[self.SOURCE_ITEM_ID]
            negative = interaction[self.SOURCE_NEG_ITEM_ID]
        else:
            users = interaction[self.TARGET_USER_ID]
            positive = interaction[self.TARGET_ITEM_ID]
            negative = interaction[self.TARGET_NEG_ITEM_ID]

        item_bias = representations[f"{domain}_item_bias"]
        positive_preference = self._pref_score(
            domain, users, positive, representations
        )
        negative_observation_preference = self._pref_score(
            domain, users, negative, representations
        )
        positive_bias = self.popularity_head[domain](
            item_bias[positive]
        ).squeeze(-1)
        negative_bias = self.popularity_head[domain](
            item_bias[negative]
        ).squeeze(-1)

        # 変調を弱め、観測BPRから人気度ヘッドへの勾配は止める。
        positive_observation = positive_preference * (
            1.0 + self.observation_modulation_strength * F.elu(positive_bias.detach())
        )
        negative_observation = negative_observation_preference * (
            1.0 + self.observation_modulation_strength * F.elu(negative_bias.detach())
        )
        observation_loss = self.bpr_loss(
            positive_observation, negative_observation
        )

        preference_loss = observation_loss.new_zeros(())
        if self.lambda_preference_rank > 0:
            if self.preference_negative_mode == "uniform":
                preference_loss = self.bpr_loss(
                    positive_preference, negative_observation_preference
                )
            else:
                preference_negative = self._matched_negative(
                    domain, users, positive, negative
                )
                matched_score = self._pref_score(
                    domain, users, preference_negative, representations
                )
                matched_loss = self.bpr_loss(positive_preference, matched_score)
                if self.preference_negative_mode == "matched":
                    preference_loss = matched_loss
                else:
                    uniform_loss = self.bpr_loss(
                        positive_preference, negative_observation_preference
                    )
                    preference_loss = (
                        (1.0 - self.matched_negative_weight) * uniform_loss
                        + self.matched_negative_weight * matched_loss
                    )
        bias_loss = self._domain_bias_loss(domain, item_bias)

        distill_weight = self._distill_weight()
        distill_loss = observation_loss.new_zeros(())
        if distill_weight > 0:
            with torch.no_grad():
                base_users = getattr(self, f"{domain}_z_user")[users]
                base_items = getattr(self, f"{domain}_z_item")
                teacher_margin = (
                    base_users * (base_items[positive] - base_items[negative])
                ).sum(dim=1)
                soft_target = torch.sigmoid(
                    teacher_margin / self.distill_temperature
                )
            student_margin = (
                positive_preference - negative_observation_preference
            )
            # Phase 1 のペア順位を設定したスケジュールで蒸留する。
            distill_loss = F.binary_cross_entropy_with_logits(
                student_margin / self.distill_temperature, soft_target
            )
        return (
            observation_loss
            + self.lambda_preference_rank * preference_loss
            + self.lambda_bias * bias_loss
            + distill_weight * distill_loss
        )

    def calculate_loss(self, interaction):
        bias_only = self.current_epoch < self.bias_warmup_epochs
        representations = self.forward(bias_only=bias_only)
        if self.phase == "SOURCE":
            domains = ("source",)
        elif self.phase == "TARGET":
            domains = ("target",)
        else:
            domains = ("source", "target")
        if bias_only:
            # Phase 2 冒頭は人気度ヘッドだけを校正する。
            return sum(
                self._domain_bias_loss(
                    domain, representations[f"{domain}_item_bias"]
                )
                for domain in domains
            )
        return sum(
            self._domain_loss(interaction, domain, representations)
            for domain in domains
        )

    def _target_scores(self, representations, users, items=None):
        base_user = self.target_z_user[users]
        base_item = self.target_z_item
        adapter_user = representations["target_user_preference"][users]
        adapter_item = representations["target_item_preference"]
        if items is None:
            base_score = base_user @ base_item[: self.target_num_items].transpose(0, 1)
            adapter_score = adapter_user @ adapter_item[: self.target_num_items].transpose(0, 1)
        else:
            base_score = (base_user * base_item[items]).sum(dim=1)
            adapter_score = (adapter_user * adapter_item[items]).sum(dim=1)
        if self.inference_score_mode == "baseline":
            return base_score
        scores = base_score + adapter_score
        if self.inference_score_mode == "preference":
            return scores
        item_bias = representations["target_item_bias"]
        if items is None:
            item_bias = item_bias[: self.target_num_items]
        else:
            item_bias = item_bias[items]
        bias_score = self.popularity_head["target"](item_bias).squeeze(-1)
        if self.inference_score_mode == "popularity_subtract":
            return scores - self.inference_popularity_alpha * bias_score
        # MACR型の参照スコアを引いた反実仮想スコア。
        modulation = F.elu(bias_score) + 1.0
        return (scores - self.counterfactual_reference) * modulation

    def predict(self, interaction):
        representations = self.forward()
        users = interaction[self.TARGET_USER_ID]
        items = interaction[self.TARGET_ITEM_ID]
        return self._target_scores(representations, users, items)

    def full_sort_predict(self, interaction):
        representations = self.forward()
        users = interaction[self.TARGET_USER_ID]
        return self._target_scores(representations, users).reshape(-1)


__all__ = ["Adapter", "DGCDRBASE", "Proposed"]

# 旧 Phase 1 チェックポイントのモジュール名をこのファイルに対応づける。
sys.modules.setdefault(
    "recbole_cdr.model.cross_domain_recommender.dgcdrbase",
    sys.modules[__name__],
)
