from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import sparse


def _observed_csr(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError("A nonempty user-by-item matrix is required")
    if np.any(~np.isfinite(result.data)) or np.any(result.data < 0):
        raise ValueError("Observed strengths must be finite and nonnegative")
    return result


def _values_on_observations(
    values: sparse.spmatrix, observed: sparse.csr_matrix
) -> np.ndarray:

    matrix = sparse.csr_matrix(values, dtype=np.float64)
    if matrix.shape != observed.shape:
        raise ValueError("Sparse input shapes must match")
    rows = np.repeat(np.arange(observed.shape[0]), np.diff(observed.indptr))
    return np.asarray(matrix[rows, observed.indices]).reshape(-1)


def _preference_csr(
    preferences: sparse.spmatrix | None, observed: sparse.csr_matrix
) -> sparse.csr_matrix:
    if preferences is None:
        values = np.ones(observed.nnz, dtype=np.float64)
    else:
        source = sparse.csr_matrix(preferences, dtype=np.float64, copy=True)
        source.sum_duplicates()
        source.eliminate_zeros()
        if source.shape != observed.shape:
            raise ValueError("Preferences and observations must have equal shape")
        if np.any(~np.isfinite(source.data)) or np.any(source.data < 0):
            raise ValueError("Preferences must be finite and nonnegative")
        supported = source.multiply(observed.astype(bool))
        if supported.nnz != source.nnz:
            raise ValueError("A nonzero preference requires an observed interaction")
        values = _values_on_observations(source, observed)

    return sparse.csr_matrix(
        (values, observed.indices.copy(), observed.indptr.copy()), shape=observed.shape
    )


def _solve_rows(
    observed: sparse.csr_matrix,
    preferences: sparse.csr_matrix,
    fixed_factors: np.ndarray,
    reg: float,
    alpha: float,
    batch_size: int = 512,
    max_padded_entries: int = 131_072,
) -> np.ndarray:

    n_rows = observed.shape[0]
    factors = fixed_factors.shape[1]
    result = np.zeros((n_rows, factors), dtype=np.float64)
    degrees = np.diff(observed.indptr)
    rows = np.flatnonzero(degrees)
    rows = rows[np.argsort(degrees[rows], kind="stable")]
    gram = fixed_factors.T @ fixed_factors + reg * np.eye(factors)
    cursor = 0
    while cursor < len(rows):
        end = min(cursor + batch_size, len(rows))
        while end > cursor + 1 and (end - cursor) * degrees[rows[end - 1]] > max_padded_entries:
            end = cursor + max(1, (end - cursor) // 2)
        batch = rows[cursor:end]
        width = int(degrees[batch[-1]])
        positions = observed.indptr[batch, None] + np.arange(width)[None, :]
        mask = np.arange(width)[None, :] < degrees[batch, None]

        positions = np.where(mask, positions, 0)
        item_ids = observed.indices[positions]
        local = fixed_factors[item_ids]
        strength = np.where(mask, observed.data[positions], 0.0)
        preference = np.where(mask, preferences.data[positions], 0.0)
        extra_confidence = alpha * strength
        systems = np.broadcast_to(gram, (len(batch), factors, factors)).copy()
        systems += np.swapaxes(local, 1, 2) @ (extra_confidence[:, :, None] * local)
        rhs = np.sum(local * ((1.0 + extra_confidence) * preference)[:, :, None], axis=1)

        result[batch] = np.linalg.solve(systems, rhs[:, :, None])[:, :, 0]
        cursor = end
    return result


def _index_array(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return result.astype(np.int64, copy=False)


class IALS:


    def __init__(
        self,
        factors: int = 32,
        reg: float = 0.1,
        alpha: float = 20.0,
        epochs: int = 10,
        seed: int = 42,
    ) -> None:
        if factors < 1 or epochs < 1 or not np.isfinite(reg) or not np.isfinite(alpha) or reg <= 0 or alpha <= 0:
            raise ValueError("factors/epochs, reg and alpha must be positive")
        self.factors, self.reg, self.alpha = int(factors), float(reg), float(alpha)
        self.epochs, self.seed = int(epochs), int(seed)

    def fit(
        self, X_observed: sparse.spmatrix, preferences: sparse.spmatrix | None = None
    ) -> "IALS":
        started = perf_counter()
        self.observed_ = _observed_csr(X_observed)
        self.preferences_ = _preference_csr(preferences, self.observed_)
        self.n_users_, self.n_items_ = self.observed_.shape
        self.confidence_ = sparse.csr_matrix(
            (1.0 + self.alpha * self.observed_.data,
             self.observed_.indices.copy(), self.observed_.indptr.copy()),
            shape=self.observed_.shape,
        )
        transposed_observed = self.observed_.T.tocsr()
        transposed_preferences = self.preferences_.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors_ = rng.normal(0, 0.01, (self.n_items_, self.factors))
        self.item_factors_[np.diff(transposed_observed.indptr) == 0] = 0
        self.user_factors_ = np.zeros((self.n_users_, self.factors), dtype=np.float64)
        self.historylosses: list[float] = []
        for _ in range(self.epochs):
            self.user_factors_ = _solve_rows(
                self.observed_, self.preferences_, self.item_factors_, self.reg, self.alpha
            )
            self.item_factors_ = _solve_rows(
                transposed_observed, transposed_preferences, self.user_factors_, self.reg, self.alpha
            )
            self.historylosses.append(self.objective())
        item_mass = np.asarray(self.preferences_.sum(axis=0)).reshape(-1)
        self.cold_item_scores_ = np.log1p(item_mass)
        scale = float(self.cold_item_scores_.max(initial=0))
        if scale:
            self.cold_item_scores_ /= scale


        self.warm_users_ = np.any(self.user_factors_ != 0, axis=1)
        self.training_summary = {
            "model": type(self).__name__, "n_users": self.n_users_, "n_items": self.n_items_,
            "observed_pairs": self.observed_.nnz,
            "nonzero_target_pairs": int(np.count_nonzero(self.preferences_.data)),
            "factors": self.factors, "reg": self.reg, "alpha": self.alpha,
            "epochs": self.epochs, "seed": self.seed, "device": "cpu",
            "fit_seconds": perf_counter() - started,
            "losses": self.historylosses.copy(),
            "confidence": "1 + alpha * observed; unobserved confidence is 1",
            "zero_user_fallback": "log1p(item preference mass), normalized",
        }
        return self

    def objective(self) -> float:

        all_scores_squared = np.sum(
            (self.user_factors_.T @ self.user_factors_)
            * (self.item_factors_.T @ self.item_factors_)
        )
        rows = np.repeat(np.arange(self.n_users_), np.diff(self.observed_.indptr))
        correction = 0.0
        for start in range(0, self.observed_.nnz, 65_536):
            stop = start + 65_536
            score = np.einsum(
                "ij,ij->i", self.user_factors_[rows[start:stop]],
                self.item_factors_[self.observed_.indices[start:stop]],
            )
            confidence = 1.0 + self.alpha * self.observed_.data[start:stop]
            correction += np.sum(
                confidence * (self.preferences_.data[start:stop] - score) ** 2 - score ** 2
            )
        penalty = self.reg * (
            np.sum(self.user_factors_ ** 2) + np.sum(self.item_factors_ ** 2)
        )
        return float(all_scores_squared + correction + penalty)

    def score(self, users: np.ndarray, items: np.ndarray | None = None) -> np.ndarray:
        users = _index_array(users, "users")
        items = np.arange(self.n_items_) if items is None else _index_array(items, "items")
        if np.any((items < 0) | (items >= self.n_items_)):
            raise IndexError("Items must belong to the fitted catalog")
        known = (users >= 0) & (users < self.n_users_)
        warm = np.zeros(len(users), dtype=bool)
        warm[known] = self.warm_users_[users[known]]
        result = np.broadcast_to(self.cold_item_scores_[items], (len(users), len(items))).copy()
        result[warm] = self.user_factors_[users[warm]] @ self.item_factors_[items].T
        return result
