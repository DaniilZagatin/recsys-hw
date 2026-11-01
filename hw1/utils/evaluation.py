import numpy as np
import pandas as pd
from scipy import sparse


def stable_top_k(scores, candidate_ids, k=10):

    scores = np.asarray(scores)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if scores.ndim != 1 or scores.shape != candidate_ids.shape:
        raise ValueError("Scores and candidate IDs must be equally sized vectors")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("Non-finite model scores")
    available = np.flatnonzero(np.isfinite(scores))
    if k <= 0 or len(available) == 0:
        return np.empty(0, dtype=np.int64)
    count = min(k, len(available))
    if len(available) > count:
        boundary = np.partition(scores[available], -count)[-count]
        better = available[scores[available] > boundary]
        ties = available[scores[available] == boundary]
        ties = ties[np.argsort(candidate_ids[ties], kind="stable")]
        selected = np.concatenate([better, ties[:count - len(better)]])
    else:
        selected = available
    order = np.lexsort((candidate_ids[selected], -scores[selected]))
    return candidate_ids[selected[order]]


def ranking_values(recommended, relevant, k=10):

    relevant = set(relevant)
    if not relevant:
        raise ValueError("Metrics require at least one relevant item")
    recommended = list(recommended)[:k]
    hits = np.asarray([item in relevant for item in recommended], dtype=np.float64)
    discount = 1 / np.log2(np.arange(2, len(recommended) + 2))
    ideal = np.sum(1 / np.log2(np.arange(2, min(k, len(relevant)) + 2)))
    return float(hits @ discount / ideal), float(hits.sum() / len(relevant))


def evaluate(model, truth: pd.DataFrame, past_observed: sparse.csr_matrix,
             train_users, train_catalog, k=10, batch_size=64,
             return_recommendations=False, return_user_metrics=False):

    if batch_size < 1 or k < 1:
        raise ValueError("batch_size and k must be positive")
    catalog = np.sort(np.unique(np.asarray(train_catalog, dtype=np.int64)))
    known_users = set(np.asarray(train_users, dtype=np.int64).tolist())
    observed = past_observed.tocsr()
    positive = truth.loc[truth.rating >= 4, ["user_id", "item_id"]].drop_duplicates()
    warm_user = positive.user_id.isin(known_users)
    warm_item = positive.item_id.isin(catalog)
    warm = positive.loc[warm_user & warm_item]
    relevant = {}
    for user, group in warm.groupby("user_id", sort=True):
        user = int(user)
        past = set(observed.indices[observed.indptr[user]:observed.indptr[user + 1]].tolist())
        targets = set(group.item_id.astype(int)) - past
        if targets:
            relevant[user] = targets
    users = np.asarray(sorted(relevant), dtype=np.int64)
    diagnostics = {
        "future_positive_pairs": int(len(positive)),
        "future_users_with_positive": int(positive.user_id.nunique()),
        "positive_pairs_known_train_user": int(warm_user.sum()),
        "positive_pairs_known_train_item": int(warm_item.sum()),
        "positive_pairs_known_train_user_and_item": int(len(warm)),
        "eligible_unseen_positive_pairs": int(sum(map(len, relevant.values()))),
        "eligible_users": int(len(users)),
        "catalog_target_coverage_for_known_users": float(len(warm) / warm_user.sum()) if warm_user.sum() else 0.0,
    }
    if len(users) == 0 or len(catalog) == 0:
        result = {"NDCG@%d" % k: float("nan"), "Recall@%d" % k: float("nan"),
                  "Coverage@%d" % k: 0.0, "n_eval_users": 0,
                  "n_candidates": int(len(catalog)), "diagnostics": diagnostics}
        if return_user_metrics:
            result.update(user_ids=users, user_metrics=np.empty((0, 2), dtype=np.float64))
        return result

    local_item = np.full(observed.shape[1], -1, dtype=np.int64)
    local_item[catalog] = np.arange(len(catalog))
    values, recommended_items, recommendations = [], set(), {}
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.asarray(model.score(batch, catalog)).copy()
        if scores.shape != (len(batch), len(catalog)):
            raise ValueError("model.score returned the wrong shape")
        if not np.issubdtype(scores.dtype, np.floating):
            scores = scores.astype(np.float64)
        for row, user in enumerate(batch):
            history = observed.indices[observed.indptr[user]:observed.indptr[user + 1]]
            positions = local_item[history]
            scores[row, positions[positions >= 0]] = -np.inf
            ranked = stable_top_k(scores[row], catalog, k)
            recommended_items.update(ranked.tolist())
            values.append(ranking_values(ranked, relevant[int(user)], k))
            if return_recommendations:
                recommendations[int(user)] = ranked.tolist()
    means = np.asarray(values).mean(axis=0)
    result = {"NDCG@%d" % k: float(means[0]), "Recall@%d" % k: float(means[1]),
              "Coverage@%d" % k: float(len(recommended_items) / len(catalog)),
              "n_eval_users": int(len(users)), "n_candidates": int(len(catalog)),
              "diagnostics": diagnostics}
    if return_recommendations:
        result["recommendations"] = recommendations
    if return_user_metrics:
        result["user_ids"] = users.copy()
        result["user_metrics"] = np.asarray(values, dtype=np.float64)
    return result
