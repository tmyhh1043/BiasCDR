"""Inspect score and representation collapse in a saved Proposed model."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from recbole.utils import init_seed

from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.model.cross_domain_recommender.Proposed import Adapter, Proposed
from recbole_cdr.quick_start.quick_start import _torch_load_checkpoint


def describe(values):
    values = values.detach().float().reshape(-1)
    return {
        "mean": round(values.mean().item(), 6),
        "std": round(values.std(unbiased=False).item(), 6),
        "min": round(values.min().item(), 6),
        "max": round(values.max().item(), 6),
    }


def effective_rank(values):
    values = values[: min(5000, len(values))].float()
    values = values - values.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.eigvalsh(values.T @ values).clamp_min(0)
    probabilities = spectrum / spectrum.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return round(entropy.exp().item(), 3)


def matrix_energy_rank(values):
    singular = torch.linalg.svdvals(values.float())
    energy = singular.square()
    probabilities = energy / energy.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return {
        "effective_rank": round(entropy.exp().item(), 3),
        "first_axis_energy_share": round(probabilities[0].item(), 4),
    }


def ranking_stats(users, items, popularity, history, positives, user_ids, topk=20):
    recommended = []
    for start in range(0, len(user_ids), 32):
        ids = user_ids[start : start + 32]
        scores = users[ids] @ items.T
        scores[:, 0] = -torch.inf
        for row, uid in enumerate(ids.tolist()):
            seen = history[uid]
            if seen is not None and len(seen):
                scores[row, seen] = -torch.inf
        recommended.append(scores.topk(topk, dim=1).indices)
    recommended = torch.cat(recommended)
    recall = []
    for uid, row in zip(user_ids.tolist(), recommended):
        positive = positives[uid]
        recall.append(
            torch.isin(row, positive).sum().item() / max(len(positive), 1)
        )
    return {
        "unique_top20_items": int(recommended.unique().numel()),
        "sample_coverage": round(
            recommended.unique().numel() / items.shape[0], 7
        ),
        "mean_top20_popularity": round(
            popularity[recommended].float().mean().item(), 4
        ),
        "sample_recall_at_20": round(sum(recall) / len(recall), 6),
        "recommendations": recommended,
    }


def inspect(config_file, checkpoint_file, sample_users):
    sys.argv = sys.argv[:1]  # RecBole に解析用引数を渡さない。
    config = CDRConfig(
        model="Proposed",
        config_file_list=[config_file, "recbole_cdr/properties/model/Proposed.yaml"],
        config_dict={"use_gpu": False, "train_batch_size": 256},
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train, _, test = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = Proposed(config, train.dataset)
    checkpoint = _torch_load_checkpoint(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        rep = model.forward()
        domain = "target"
        user_pref = rep[f"{domain}_user_preference"]
        item_pref = rep[f"{domain}_item_preference"][: model.target_num_items]
        item_bias = rep[f"{domain}_item_bias"][: model.target_num_items]
        base_user = model.target_z_user
        base_item = model.target_z_item[: model.target_num_items]
        bias_score = model.popularity_head[domain](item_bias).squeeze(-1)
        log_pop = model.target_item_log_popularity[: model.target_num_items]
        popularity = log_pop.expm1()
        valid_items = torch.arange(1, model.target_num_items)
        exposure_gap = {
            "uniform_item_mean_log_popularity": round(
                log_pop[valid_items].mean().item(), 4
            ),
            "observed_positive_mean_log_popularity": round(
                (popularity[valid_items] * log_pop[valid_items]).sum().item()
                / popularity[valid_items].sum().item(), 4
            ),
        }

        user_ids = test.uid_list[:sample_users].long()
        history = test.uid2history_item
        positives = test.uid2positive_item
        baseline = ranking_stats(
            base_user, base_item, popularity, history, positives, user_ids
        )
        preference = ranking_stats(
            user_pref, item_pref, popularity, history, positives, user_ids
        )
        cosine_preference = ranking_stats(
            F.normalize(user_pref, dim=1),
            F.normalize(item_pref, dim=1),
            popularity, history, positives, user_ids,
        )
        modulation = F.elu(bias_score) + 1.0
        observed = ranking_stats(
            user_pref, item_pref * modulation[:, None],
            popularity, history, positives, user_ids,
        )
        batch = next(iter(train))
        users = batch[model.TARGET_USER_ID]
        positive = batch[model.TARGET_ITEM_ID]
        negative = batch[model.TARGET_NEG_ITEM_ID]
        pref_pos = (user_pref[users] * item_pref[positive]).sum(-1)
        pref_neg = (user_pref[users] * item_pref[negative]).sum(-1)
        obs_pos = pref_pos * modulation[positive]
        obs_neg = pref_neg * modulation[negative]
        unique_items = torch.unique(torch.cat((positive, negative)))
        unique_items = unique_items[unique_items > 0]
        losses = {
            "observation_bpr": round(model.bpr_loss(obs_pos, obs_neg).item(), 6),
            "preference_bpr": round(model.bpr_loss(pref_pos, pref_neg).item(), 6),
            "bias_mse": round(F.mse_loss(
                bias_score[unique_items], log_pop[unique_items]
            ).item(), 6),
        }
        cosine = F.cosine_similarity(
            user_pref[user_ids[:-1]], user_pref[user_ids[1:]], dim=1
        )
        torch.manual_seed(2022)
        fresh_user = Adapter(model.gcn_output_size, model.embedding_size)
        fresh_item = Adapter(model.gcn_output_size, model.embedding_size)
        fresh_user_pref = fresh_user(base_user)
        fresh_item_pref = fresh_item(base_item)
        fresh_ranking = ranking_stats(
            fresh_user_pref, fresh_item_pref,
            popularity, history, positives, user_ids,
        )
        initial_rank = {
            "user": effective_rank(fresh_user_pref),
            "item": effective_rank(fresh_item_pref),
        }
        fresh_positive_score = (
            fresh_user_pref[users] * fresh_item_pref[positive]
        ).sum(-1)
        trained_layers = model.preference_adapter["target_item"].layers
        before_gelu = trained_layers[0](base_item)
        after_gelu = trained_layers[1](before_gelu)
        item_mean_score = user_pref[user_ids].mean(dim=0) @ item_pref.T

    overlap = (
        (preference["recommendations"][:, :, None]
         == baseline["recommendations"][:, None, :])
        .any(dim=2).float().sum(dim=1).mean().item()
    )
    result = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "sample_users": len(user_ids),
        "losses_target_one_batch": losses,
        "sampling_popularity_gap": exposure_gap,
        "score_distribution": {
            "preference_positive": describe(pref_pos),
            "preference_negative": describe(pref_neg),
            "observed_positive": describe(obs_pos),
            "bias_score": describe(bias_score[valid_items]),
            "log_popularity": describe(log_pop[valid_items]),
            "modulation": describe(modulation[valid_items]),
            "fresh_adapter_positive": describe(fresh_positive_score),
        },
        "effective_rank": {
            "base_user": effective_rank(base_user),
            "fresh_adapter_user": initial_rank["user"],
            "preference_user": effective_rank(user_pref),
            "base_item": effective_rank(base_item),
            "fresh_adapter_item": initial_rank["item"],
            "trained_item_before_gelu": effective_rank(before_gelu),
            "trained_item_after_gelu": effective_rank(after_gelu),
            "preference_item": effective_rank(item_pref),
        },
        "user_preference_pairwise_cosine": describe(cosine),
        "target_item_adapter_final_matrix": matrix_energy_rank(
            trained_layers[2].weight
        ),
        "preference_mean_score_popularity_spearman": round(float(spearmanr(
            item_mean_score[valid_items].numpy(), log_pop[valid_items].numpy()
        ).correlation), 6),
        "preference_item_norm_popularity_spearman": round(float(spearmanr(
            item_pref[valid_items].norm(dim=1).numpy(),
            log_pop[valid_items].numpy()
        ).correlation), 6),
        "base_item_norm_popularity_spearman": round(float(spearmanr(
            base_item[valid_items].norm(dim=1).numpy(),
            log_pop[valid_items].numpy()
        ).correlation), 6),
        "bias_popularity_spearman": round(float(spearmanr(
            bias_score[valid_items].numpy(), log_pop[valid_items].numpy()
        ).correlation), 6),
        "baseline_ranking": {k: v for k, v in baseline.items() if k != "recommendations"},
        "fresh_adapter_ranking": {k: v for k, v in fresh_ranking.items() if k != "recommendations"},
        "preference_ranking": {k: v for k, v in preference.items() if k != "recommendations"},
        "cosine_preference_ranking": {k: v for k, v in cosine_preference.items() if k != "recommendations"},
        "observed_ranking": {k: v for k, v in observed.items() if k != "recommendations"},
        "pref_vs_base_mean_top20_overlap": round(overlap, 3),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-users", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = inspect(args.config, args.checkpoint, args.sample_users)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
