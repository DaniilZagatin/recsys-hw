from time import perf_counter

import numpy as np
from scipy import sparse


def _positive_mask(X):
    if not sparse.issparse(X):
        raise TypeError("X must be a SciPy sparse user-item matrix")
    X = sparse.csr_matrix(X, dtype=np.float32, copy=True)
    if not np.isfinite(X.data).all() or np.any(X.data < 0):
        raise ValueError("X must contain finite, nonnegative interactions")
    X.sum_duplicates()
    X.eliminate_zeros()
    X.data.fill(1.0)
    X.sort_indices()
    return X


def _top_indices(values, item_ids, limit):

    if len(values) > limit:
        threshold = np.partition(values, len(values) - limit)[len(values) - limit]
        larger = np.flatnonzero(values > threshold)
        tied = np.flatnonzero(values == threshold)
        tied = tied[np.argsort(item_ids[tied], kind="stable")[:limit - len(larger)]]
        selected = np.concatenate((larger, tied))
    else:
        selected = np.arange(len(values))
    return selected[np.lexsort((item_ids[selected], -values[selected]))]


def _checked_indices(values, size, name):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values >= size):
        raise IndexError(f"{name} contains an out-of-range index")
    return values.astype(np.intp, copy=False)


def _sparse_bytes(matrix):
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


class ItemKNN:
    def __init__(self, k=100, shrink=10.0, block_size=256):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if not np.isfinite(shrink) or shrink < 0:
            raise ValueError("shrink must be finite and nonnegative")
        if not isinstance(block_size, (int, np.integer)) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self.k = int(k)
        self.shrink = float(shrink)
        self.block_size = int(block_size)

    def fit(self, X):
        started = perf_counter()
        self.X_ = _positive_mask(X)
        n_users, n_items = self.X_.shape
        XT = self.X_.T.tocsr()
        counts = np.asarray(self.X_.sum(axis=0)).ravel()
        inv_norm = np.zeros(n_items, dtype=np.float64)
        np.divide(1.0, np.sqrt(counts), out=inv_norm, where=counts > 0)

        indptr = np.zeros(n_items + 1, dtype=np.int64)
        indices, values = [], []
        max_block_nnz = max_block_bytes = 0
        for start in range(0, n_items, self.block_size):
            stop = min(start + self.block_size, n_items)
            block = (XT[start:stop] @ self.X_).tocsr()
            max_block_nnz = max(max_block_nnz, block.nnz)
            max_block_bytes = max(max_block_bytes, _sparse_bytes(block))
            for local, item in enumerate(range(start, stop)):
                left, right = block.indptr[local:local + 2]
                ids = block.indices[left:right]
                overlap = block.data[left:right].astype(np.float64)
                keep = (ids != item) & (overlap > 0)
                ids, overlap = ids[keep], overlap[keep]
                similarity = overlap * inv_norm[item] * inv_norm[ids]
                if self.shrink:
                    similarity *= overlap / (overlap + self.shrink)
                chosen = _top_indices(similarity, ids, self.k)
                indices.append(ids[chosen].copy())
                values.append(similarity[chosen].astype(np.float32))
                indptr[item + 1] = indptr[item] + len(chosen)
        all_indices = np.concatenate(indices) if indices else np.empty(0, dtype=np.int32)
        all_values = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
        self.weights_ = sparse.csc_matrix(
            (all_values, all_indices, indptr), shape=(n_items, n_items)
        )
        self.weights_.sort_indices()
        self.coef_ = self.weights_
        self.training_summary = {
            "model": "ItemKNN",
            "fit_seconds": float(perf_counter() - started),
            "n_users": int(n_users), "n_items": int(n_items),
            "input_nnz": int(self.X_.nnz), "weight_nnz": int(self.weights_.nnz),
            "k": self.k, "shrink": self.shrink, "block_size": self.block_size,
            "cold_train_items": int(np.count_nonzero(counts == 0)),
            "items_without_neighbors": int(np.count_nonzero(np.diff(indptr) == 0)),
            "input_csr_bytes": _sparse_bytes(self.X_),
            "weights_csc_bytes": _sparse_bytes(self.weights_),
            "max_block_nnz": int(max_block_nnz),
            "max_block_csr_bytes": int(max_block_bytes),
            "resource_note": "Sparse buffers only; not a measurement of process peak RSS.",
            "convergence": "Exact blockwise similarities and top-k selection; no optimizer.",
        }
        return self

    def score(self, users, items=None):
        if not hasattr(self, "weights_"):
            raise RuntimeError("Call fit before score")
        users = _checked_indices(users, self.X_.shape[0], "users")
        if items is None:
            weights = self.weights_
        else:
            items = _checked_indices(items, self.X_.shape[1], "items")
            weights = self.weights_[:, items]
        return (self.X_[users] @ weights).toarray()
