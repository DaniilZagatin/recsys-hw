from __future__ import annotations

import numpy as np
from scipy import sparse

from .ials import IALS, _observed_csr, _values_on_observations


class CustomIALS(IALS):


    def __init__(
        self,
        factors: int = 32,
        reg: float = 0.1,
        alpha: float = 20.0,
        epochs: int = 10,
        seed: int = 42,
        half_life: float = 1095.0,
    ) -> None:
        super().__init__(factors=factors, reg=reg, alpha=alpha, epochs=epochs, seed=seed)
        if np.isnan(half_life) or half_life <= 0:
            raise ValueError("half_life must be positive or np.inf")
        self.half_life = float(half_life)

    def fit(
        self,
        X_observed: sparse.spmatrix,
        ratings: sparse.spmatrix,
        timestamps: sparse.spmatrix,
        cutoff: float,
    ) -> "CustomIALS":
        if not np.isfinite(cutoff):
            raise ValueError("cutoff must be a finite Unix timestamp in seconds")
        observed = _observed_csr(X_observed)
        rating = _values_on_observations(ratings, observed)
        timestamp = _values_on_observations(timestamps, observed)
        valid_rating = np.isfinite(rating) & (rating >= 0) & (rating <= 5)
        valid_timestamp = np.isfinite(timestamp) & (timestamp > 0)
        if np.any(valid_timestamp & (timestamp > cutoff)):
            raise ValueError("A training interaction timestamp is later than cutoff")
        preference = np.clip((np.where(valid_rating, rating, 0) - 2.0) / 3.0, 0.0, 1.0)
        if np.isinf(self.half_life):
            decay = np.ones(observed.nnz)
        else:
            age_days = np.maximum(0.0, (cutoff - np.where(valid_timestamp, timestamp, cutoff)) / 86400.0)
            decay = np.where(valid_timestamp, np.exp2(-age_days / self.half_life), 0.0)
        target = np.nan_to_num(preference * decay, nan=0.0, posinf=0.0, neginf=0.0)
        preferences = sparse.csr_matrix(
            (target, observed.indices.copy(), observed.indptr.copy()), shape=observed.shape
        )
        super().fit(observed, preferences=preferences)
        self.custom_preferences_ = self.preferences_
        self.training_summary.update({
            "target": "max(0, (rating-2)/3) * 2**(-age_days/half_life)",
            "half_life_days": self.half_life, "cutoff_unix_seconds": float(cutoff),
            "invalid_rating_pairs": int(np.count_nonzero(~valid_rating)),
            "missing_timestamp_pairs": int(np.count_nonzero(~valid_timestamp)),
            "target_min": float(target.min()) if target.size else 0.0,
            "target_max": float(target.max()) if target.size else 0.0,
            "confidence_changed": False,
        })
        return self
