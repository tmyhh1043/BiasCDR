# Proposed: fixed-weight evaluation

Run all five seeds for both directions:

```bash
./run_proposed_fixed_weight.sh
```

Run one checkpoint first:

```bash
FIXED_WEIGHT_DATASETS=AmazonSport_AmazonPhone \
FIXED_WEIGHT_SEEDS=2022 \
FIXED_WEIGHT_MAX_USERS=200 \
./run_proposed_fixed_weight.sh
```

Set `FIXED_WEIGHT_PYTHON` to the Python executable used for training if the
default `python3` does not have PyTorch, SciPy, and RecBole. Results are written
to `results/proposed_fixed_weight/` as one JSON per checkpoint and a summary CSV
with one row per graph keep rate.

| Analysis | Output | Interpretation |
| --- | --- | --- |
| Overlap-user groups | Target-history and source-popularity group NDCG/Recall | Which already-overlapping users benefit |
| Popularity and diversity | Head/middle/tail/cold recall, coverage, exposure Gini, warm-tail exposure | Accuracy and popularity tradeoff |
| Cross-domain correspondence | Correct-pair versus shuffled-pair cosine, user retrieval | Representation alignment only |
| Target graph thinning | Same metrics for 100%, 75%, 50%, 25% retained target train edges | Sensitivity to missing graph edges at inference |
| Test timestamp groups | Recall for older/middle/recent test interactions | Variation within the original random split |

All model parameters remain frozen. The target recommender does not read source
embeddings at inference, so permuting source users after training cannot test
the causal contribution of transfer. Graph thinning does not simulate training
with fewer target interactions. Timestamp groups do not form a chronological
train/test split because the checkpoint was trained with a random split.

For graph thinning, the original test history mask and popularity groups stay
fixed. This isolates the embedding change caused by missing graph edges.
Timestamp mapping uses the raw target `.inter` file. If several raw interactions
share a user-item pair, that pair is excluded from timestamp analysis because
its held-out event cannot be identified uniquely.
