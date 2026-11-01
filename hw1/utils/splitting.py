from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class InteractionMatrices:
    observed: sparse.csr_matrix
    positive: sparse.csr_matrix
    ratings: sparse.csr_matrix
    timestamps: sparse.csr_matrix


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_users: np.ndarray
    train_catalog: np.ndarray
    boundaries: Dict[str, int]

    def summary(self) -> dict:
        result = {"boundaries_unix_seconds": self.boundaries,
                  "boundaries_utc": {k: pd.Timestamp(v, unit="s", tz="UTC").isoformat()
                                     for k, v in self.boundaries.items()},
                  "train_users": int(len(self.train_users)),
                  "train_catalog": int(len(self.train_catalog)), "parts": {}}
        known_users, known_items = set(self.train_users), set(self.train_catalog)
        for name, frame in (("train", self.train), ("validation", self.validation), ("test", self.test)):
            positive = frame.loc[frame.rating >= 4]
            warm_user = positive.user_id.isin(known_users)
            warm_item = positive.item_id.isin(known_items)
            result["parts"][name] = {
                "rows": int(len(frame)), "users": int(frame.user_id.nunique()),
                "items": int(frame.item_id.nunique()), "positive_pairs": int(len(positive)),
                "positive_pairs_known_train_user": int(warm_user.sum()),
                "positive_pairs_known_train_item": int(warm_item.sum()),
                "positive_pairs_known_train_user_and_item": int((warm_user & warm_item).sum()),
                "users_with_warm_positive": int(positive.loc[warm_user & warm_item, "user_id"].nunique()),
            }
        return result


def make_temporal_split(ratings: pd.DataFrame, n_users=None, n_items=None,
                        train_fraction=0.70, validation_fraction=0.15) -> TemporalSplit:

    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Need positive train/validation/test fractions")
    if len(ratings) < 3:
        raise ValueError("At least three observations are needed")
    ordered = ratings.sort_values(["timestamp", "user_id", "item_id"], kind="stable")
    times = ordered.timestamp.to_numpy(dtype=np.int64)
    pos1 = int(np.floor(len(times) * train_fraction))
    pos2 = int(np.floor(len(times) * round(train_fraction + validation_fraction, 12)))
    if not (0 < pos1 < pos2 < len(times)):
        raise ValueError("Fractions create an empty period")
    first, second = int(times[pos1]), int(times[pos2])
    train = ordered.loc[ordered.timestamp < first].copy().reset_index(drop=True)
    valid = ordered.loc[(ordered.timestamp >= first) & (ordered.timestamp < second)].copy().reset_index(drop=True)
    test = ordered.loc[ordered.timestamp >= second].copy().reset_index(drop=True)
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Timestamp ties create an empty period; choose explicit wider periods")
    return TemporalSplit(train, valid, test,
                         np.sort(train.user_id.unique()).astype(np.int64),
                         np.sort(train.item_id.unique()).astype(np.int64),
                         {"validation_start": first, "test_start": second})


def build_interactions(frame: pd.DataFrame, n_users: int, n_items: int,
                       positive_threshold=4) -> InteractionMatrices:

    if frame.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Clean duplicate user/item pairs before constructing matrices")
    ordered = frame.sort_values(["user_id", "item_id"], kind="stable")
    users = ordered.user_id.to_numpy(dtype=np.int64)
    items = ordered.item_id.to_numpy(dtype=np.int64)
    values = ordered.rating.to_numpy(dtype=np.float32)
    times = ordered.timestamp.to_numpy(dtype=np.int64)
    if ((users < 0).any() or (users >= n_users).any()
            or (items < 0).any() or (items >= n_items).any()):
        raise ValueError("Encoded IDs exceed the supplied matrix shape")
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.add.at(indptr, users + 1, 1)
    np.cumsum(indptr, out=indptr)
    shape = (n_users, n_items)
    observed = sparse.csr_matrix((np.ones(len(items), np.float32), items, indptr), shape=shape)
    ratings = sparse.csr_matrix((values.copy(), observed.indices.copy(), observed.indptr.copy()), shape=shape)
    timestamps = sparse.csr_matrix((times.copy(), observed.indices.copy(), observed.indptr.copy()), shape=shape)
    positive = observed.copy()
    positive.data = (values >= positive_threshold).astype(np.float32)
    positive.eliminate_zeros()
    return InteractionMatrices(observed, positive, ratings, timestamps)
