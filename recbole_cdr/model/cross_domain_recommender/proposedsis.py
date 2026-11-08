"""SIS candidate: compact preference/popularity cross-domain recommendation.

Two-stage training:
  (1) Train DGCDRBASE on source+target interactions (unchanged).
  (2) Load its checkpoint, freeze/copy its GCN outputs, and train this model.

Ablation controls:
  lambda_adv=0     : compact preference + scalar popularity bias (M2)
  lambda_adv>0     : add gradient-reversal popularity classifier (M3)
  lambda_cross=0   : remove source -> target preference alignment (M4)

This class expects the same checkpoint-loading entrypoint as the earlier
PROPOSED class: load_pretrained_backbone_checkpoint(checkpoint, path).
Only training interactions are used for popularity and overlap statistics.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.cross_domain_recommender.dgcdrbase import DGCDRBASE


class _GradientReverse(torch.autograd.Function):
    """Identity forward; multiply preference gradient by a negative scale."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(x, scale)


class PROPOSEDSIS(DGCDRBASE):
    """Frozen DGCDR + low-rank preference + explicit popularity + optional GRL."""

    BACKBONE_PARAMETER_NAMES = (
        "source_user_embedding.weight",
        "source_item_embedding.weight",
        "target_user_embedding.weight",
        "target_item_embedding.weight",
    )

    @staticmethod
    def _get(config, key, default=None):
        try:
            return config[key] if key in config else default
        except (KeyError, TypeError):
            return default

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.phase = "BOTH"
        self.pretrained_model_file = self._get(config, "pretrained_model_file")
        if not self.pretrained_model_file:
            raise ValueError("A DGCDRBASE pretrained_model_file is required.")
        self.pretrained_model_file = os.path.abspath(
            os.path.expanduser(str(self.pretrained_model_file))
        )

        self.preference_dim = int(self._get(config, "preference_dim", 32))
        self.lambda_preference_rank = float(
            self._get(config, "lambda_preference_rank", 1.0)
        )
        self.lambda_cross = float(self._get(config, "lambda_cross", 0.1))
        self.lambda_adv = float(self._get(config, "lambda_adv", 0.0))
        self.lambda_beta_l2 = float(self._get(config, "lambda_beta_l2", 1e-4))
        self.cross_sample_size = int(self._get(config, "cross_sample_size", 512))
        self.adv_sample_size = int(self._get(config, "adv_sample_size", 256))
        self.adv_hidden_dim = int(self._get(config, "adv_hidden_dim", 64))
        self.gcn_output_size = self.embedding_size * (
            self.n_layers + 1 if self.connect_way == "concat" else 1
        )
        if self.preference_dim <= 0 or self.preference_dim >= self.gcn_output_size:
            raise ValueError("preference_dim must be >0 and below GCN output size.")
        if min(self.cross_sample_size, self.adv_sample_size, self.adv_hidden_dim) <= 0:
            raise ValueError("sample sizes and adv_hidden_dim must be positive.")
        if min(
            self.lambda_preference_rank,
            self.lambda_cross,
            self.lambda_adv,
            self.lambda_beta_l2,
        ) < 0:
            raise ValueError("Loss strengths must be non-negative.")

        input_dim = self.gcn_output_size
        dim = self.preference_dim
        self.source_user_preference_projection = nn.Linear(input_dim, dim, bias=False)
        self.source_item_preference_projection = nn.Linear(input_dim, dim, bias=False)
        self.target_user_preference_projection = nn.Linear(input_dim, dim, bias=False)
        self.target_item_preference_projection = nn.Linear(input_dim, dim, bias=False)

        # A one-dimensional popularity sensitivity is projected from each
        # user's *frozen GCN representation*: no separate user bias embedding.
        self.source_user_popularity_coefficient = nn.Linear(input_dim, 1, bias=True)
        self.target_user_popularity_coefficient = nn.Linear(input_dim, 1, bias=True)

        for domain in ("source", "target"):
            matrix = dataset.inter_matrix(
                form="coo", value_field=None, domain=domain
            ).tocoo()
            self._register_popularity(domain, matrix)

        self.source_popularity_classifier = self._make_classifier(
            dim, self.adv_hidden_dim, self.source_popularity_num_classes
        )
        self.target_popularity_classifier = self._make_classifier(
            dim, self.adv_hidden_dim, self.target_popularity_num_classes
        )

        for module in self._phase2_modules():
            module.apply(self._init_weights)
        # Starting with zero coefficients reproduces the preference-only score.
        for name in ("source_user_popularity_coefficient",
                     "target_user_popularity_coefficient"):
            head = getattr(self, name)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self._expected_source = str(
            self._get(self._get(config, "source_domain", {}), "dataset", "")
        )
        self._expected_target = str(
            self._get(self._get(config, "target_domain", {}), "dataset", "")
        )
        self.register_buffer(
            "overlap_user_ids", self._verified_overlap_ids(dataset), persistent=False
        )
        if self.lambda_cross > 0 and self.overlap_user_ids.numel() == 0:
            raise ValueError("Cross-domain loss requires overlapping users.")

        for name in (
            "source_z_user", "source_z_item", "target_z_user", "target_z_item"
        ):
            self.register_buffer(name, torch.empty(0), persistent=False)
        self._frozen_cache_ready = False
        self._loaded_backbone_checkpoint = None
        self._freeze_backbone()
        self.other_parameter_name = []

    @staticmethod
    def _make_classifier(input_dim: int, hidden_dim: int, classes: int):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(classes, 2)),
        )

    def _phase2_modules(self):
        return (
            self.source_user_preference_projection,
            self.source_item_preference_projection,
            self.target_user_preference_projection,
            self.target_item_preference_projection,
            self.source_user_popularity_coefficient,
            self.target_user_popularity_coefficient,
            self.source_popularity_classifier,
            self.target_popularity_classifier,
        )

    def _register_popularity(self, domain, matrix):
        users = matrix.row.astype(np.int64, copy=False)
        items = matrix.col.astype(np.int64, copy=False)
        valid = (users > 0) & (items > 0)
        counts = np.bincount(
            items[valid], minlength=self.total_num_items
        ).astype(np.float32)
        active = np.flatnonzero(counts > 0).astype(np.int64)
        log_counts = np.log1p(counts)
        if active.size:
            mean = log_counts[active].mean()
            std = max(float(log_counts[active].std()), 1e-6)
        else:
            mean, std = 0.0, 1.0
        standardized = ((log_counts - mean) / std).astype(np.float32)
        standardized[counts == 0] = 0.0
        standardized[0] = 0.0
        self.register_buffer(
            f"{domain}_standardized_popularity",
            torch.from_numpy(standardized), persistent=False,
        )
        self.register_buffer(
            f"{domain}_training_item_ids", torch.from_numpy(active),
            persistent=False,
        )

        # Quantile labels from *training* item degrees only. Equal degrees are
        # assigned equal labels; avoid arbitrary splitting of tied items.
        labels = np.zeros(self.total_num_items, dtype=np.int64)
        if active.size:
            cuts = np.unique(np.quantile(log_counts[active], [1/3, 2/3]))
            raw_labels = np.searchsorted(cuts, log_counts[active], side="right")
            present = np.unique(raw_labels)
            labels[active] = np.searchsorted(present, raw_labels)
            num_classes = len(present)
        else:
            num_classes = 0
        self.register_buffer(
            f"{domain}_popularity_classes", torch.from_numpy(labels),
            persistent=False,
        )
        setattr(self, f"{domain}_popularity_num_classes", num_classes)
        if self.lambda_adv > 0 and num_classes < 2:
            raise ValueError(
                f"{domain}: training-item popularity has fewer than two classes. "
                "Disable lambda_adv or use a dataset with varying item degrees."
            )

    @staticmethod
    def _verified_overlap_ids(dataset):
        source = dataset.source_user_ID_remap_dict
        target = dataset.target_user_ID_remap_dict
        overlap = []
        for raw in set(source).intersection(target):
            if raw == "[PAD]" or (isinstance(raw, float) and np.isnan(raw)):
                continue
            su, ta = int(source[raw]), int(target[raw])
            if su != ta:
                raise ValueError(f"Overlap user {raw!r}: mismatched global IDs.")
            if su > 0:
                overlap.append(su)
        ids = torch.tensor(sorted(set(overlap)), dtype=torch.long)
        expected = int(dataset.num_overlap_user) - 1
        if ids.numel() != expected:
            raise ValueError(
                f"Overlap count {ids.numel()} differs from expected {expected}."
            )
        return ids

    def _freeze_backbone(self):
        parameters = dict(self.named_parameters())
        for name in self.BACKBONE_PARAMETER_NAMES:
            parameters[name].requires_grad_(False)

    def backbone_is_frozen(self):
        parameters = dict(self.named_parameters())
        return all(
            not parameters[name].requires_grad
            for name in self.BACKBONE_PARAMETER_NAMES
        )

    def phase2_named_parameters(self):
        for name, parameter in self.named_parameters():
            if name not in self.BACKBONE_PARAMETER_NAMES:
                yield name, parameter

    def phase2_parameters(self):
        for _, parameter in self.phase2_named_parameters():
            yield parameter

    def set_phase(self, phase):
        self.phase = str(phase).upper()

    def refresh_frozen_representations(self):
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                source_user, source_item, target_user, target_item = (
                    DGCDRBASE.forward(self)
                )
                self.source_z_user = source_user.detach()
                self.source_z_item = source_item.detach()
                self.target_z_user = target_user.detach()
                self.target_z_item = target_item.detach()
                self._frozen_cache_ready = True
        finally:
            self.train(was_training)

    def load_pretrained_backbone_checkpoint(self, checkpoint, path=None):
        """Call this after .to(device), before Phase 2 optimizer/training."""
        config = checkpoint.get("config")
        if config is None or str(self._get(config, "model", "")).upper() != "DGCDRBASE":
            raise ValueError("Checkpoint must be a DGCDRBASE checkpoint with config.")
        for key, current in (
            ("embedding_size", self.embedding_size),
            ("n_layers", self.n_layers),
            ("connect_way", self.connect_way),
        ):
            previous = self._get(config, key)
            if previous is not None and str(previous).lower() != str(current).lower():
                raise ValueError(f"Backbone {key} does not match: {previous} != {current}")
        for domain, expected in (
            ("source_domain", self._expected_source),
            ("target_domain", self._expected_target),
        ):
            old_dataset = str(self._get(self._get(config, domain, {}), "dataset", ""))
            if expected and old_dataset and old_dataset != expected:
                raise ValueError(f"Backbone {domain} dataset does not match.")
        state = checkpoint.get("state_dict", checkpoint)
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name in self.BACKBONE_PARAMETER_NAMES:
                if name not in state:
                    raise ValueError(f"Missing backbone weight: {name}")
                if state[name].shape != parameters[name].shape:
                    raise ValueError(f"Backbone weight shape mismatch: {name}")
                parameters[name].copy_(state[name].to(parameters[name].device))
        self._freeze_backbone()
        self.refresh_frozen_representations()
        self._loaded_backbone_checkpoint = str(path or self.pretrained_model_file)
        return {"cache_ready": self._frozen_cache_ready,
                "pretrained_checkpoint": self._loaded_backbone_checkpoint}

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._freeze_backbone()
        self.refresh_frozen_representations()
        return result

    def _preference_representations(self):
        if not self._frozen_cache_ready:
            raise RuntimeError(
                "Frozen GCN cache not ready. Call "
                "load_pretrained_backbone_checkpoint(...) before training."
            )
        return {
            "source_user": self.source_user_preference_projection(self.source_z_user),
            "source_item": self.source_item_preference_projection(self.source_z_item),
            "target_user": self.target_user_preference_projection(self.target_z_user),
            "target_item": self.target_item_preference_projection(self.target_z_item),
        }

    def forward(self):
        p = self._preference_representations()
        return p["source_user"], p["source_item"], p["target_user"], p["target_item"]

    def _domain_losses(self, domain, p, user, positive, negative):
        user_p = p[f"{domain}_user"][user]
        pos_p = p[f"{domain}_item"][positive]
        neg_p = p[f"{domain}_item"][negative]
        pos_pref = (user_p * pos_p).sum(dim=-1)
        neg_pref = (user_p * neg_p).sum(dim=-1)

        frozen_user_z = getattr(self, f"{domain}_z_user")[user]
        coefficient = getattr(self, f"{domain}_user_popularity_coefficient")(
            frozen_user_z
        ).squeeze(-1)
        popularity = getattr(self, f"{domain}_standardized_popularity")
        pos_bias = coefficient * popularity[positive]
        neg_bias = coefficient * popularity[negative]
        observation = self.bpr_loss(pos_pref + pos_bias, neg_pref + neg_bias)
        preference = self.bpr_loss(pos_pref, neg_pref)
        beta_reg = coefficient.square().mean()
        return observation, preference, beta_reg

    def _cross_loss(self, p):
        if self.lambda_cross == 0 or self.overlap_user_ids.numel() == 0:
            return p["target_user"].new_zeros(())
        overlap = self.overlap_user_ids
        indices = torch.randint(
            overlap.numel(),
            (min(self.cross_sample_size, overlap.numel()),),
            device=overlap.device,
        )
        users = overlap[indices]
        source_teacher = p["source_user"][users].detach()
        target_student = p["target_user"][users]
        return (1.0 - F.cosine_similarity(
            source_teacher, target_student, dim=-1, eps=1e-8
        )).mean()

    def _adversarial_loss(self, domain, p):
        if self.lambda_adv == 0:
            return p[f"{domain}_item"].new_zeros(())
        candidates = getattr(self, f"{domain}_training_item_ids")
        positions = torch.randint(
            candidates.numel(), (self.adv_sample_size,), device=candidates.device
        )
        ids = candidates[positions]
        item_p = p[f"{domain}_item"][ids]
        head = getattr(self, f"{domain}_popularity_classifier")
        labels = getattr(self, f"{domain}_popularity_classes")[ids]
        # The classifier receives normal gradients; preference projections
        # receive reversed gradients scaled by lambda_adv.
        return F.cross_entropy(head(gradient_reverse(item_p, self.lambda_adv)), labels)

    def calculate_loss_components(self, interaction):
        p = self._preference_representations()
        source = self._domain_losses(
            "source", p,
            interaction[self.SOURCE_USER_ID],
            interaction[self.SOURCE_ITEM_ID],
            interaction[self.SOURCE_NEG_ITEM_ID],
        )
        target = self._domain_losses(
            "target", p,
            interaction[self.TARGET_USER_ID],
            interaction[self.TARGET_ITEM_ID],
            interaction[self.TARGET_NEG_ITEM_ID],
        )
        obs = source[0] + target[0]
        pref = source[1] + target[1]
        beta_reg = source[2] + target[2]
        cross = self._cross_loss(p)
        adv_source = self._adversarial_loss("source", p)
        adv_target = self._adversarial_loss("target", p)
        adv = adv_source + adv_target
        return {
            "observation": obs,
            "preference_rank": pref,
            "cross": cross,
            "adversarial_classifier": adv,
            "beta_regularization": beta_reg,
            "total": obs + self.lambda_preference_rank * pref
                     + self.lambda_cross * cross + adv
                     + self.lambda_beta_l2 * beta_reg,
        }

    def calculate_loss(self, interaction):
        parts = self.calculate_loss_components(interaction)
        # GRL is intentionally applied inside the classifier inputs.
        return (
            parts["observation"],
            self.lambda_preference_rank * parts["preference_rank"],
            self.lambda_cross * parts["cross"],
            parts["adversarial_classifier"],
            self.lambda_beta_l2 * parts["beta_regularization"],
        )

    def predict(self, interaction):
        p = self._preference_representations()
        u = p["target_user"][interaction[self.TARGET_USER_ID]]
        i = p["target_item"][interaction[self.TARGET_ITEM_ID]]
        return (u * i).sum(dim=-1)

    def full_sort_predict(self, interaction):
        p = self._preference_representations()
        users = p["target_user"][interaction[self.TARGET_USER_ID]]
        items = p["target_item"][:self.target_num_items]
        scores = []
        for start in range(0, self.target_num_items, self.full_sort_chunk_size):
            scores.append(users @ items[start:start + self.full_sort_chunk_size].T)
        return torch.cat(scores, dim=1).reshape(-1)


__all__ = ["PROPOSEDSIS", "gradient_reverse"]
