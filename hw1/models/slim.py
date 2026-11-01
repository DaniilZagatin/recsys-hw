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


def _top_indices(values, ids, limit):
    if len(values) > limit:
        threshold = np.partition(values, len(values) - limit)[len(values) - limit]
        larger = np.flatnonzero(values > threshold)
        tied = np.flatnonzero(values == threshold)
        tied = tied[np.argsort(ids[tied], kind="stable")[:limit - len(larger)]]
        selected = np.concatenate((larger, tied))
    else:
        selected = np.arange(len(values))
    return selected[np.lexsort((ids[selected], -values[selected]))]


def _sparse_bytes(matrix):
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _kkt(weights, gradient):

    residual = np.where(weights > 0, np.abs(gradient), np.maximum(-gradient, 0))
    return float(residual.max(initial=0.0))


def _solve_nonnegative(H, b, l1, max_iter, tol):

    weights = np.zeros(len(b), dtype=np.float64)
    gradient = -b + l1
    scale = max(1.0, float(np.max(b, initial=0.0)))
    violation = _kkt(weights, gradient)
    if violation <= tol * scale:
        return weights, 0, violation, True, 0.0


    lipschitz = float(np.max(np.sum(H, axis=1)))
    extrapolated = weights.copy()
    momentum = 1.0
    objective = 0.0
    for iteration in range(1, max_iter + 1):
        proposal = np.maximum(0.0, extrapolated - (H @ extrapolated - b + l1) / lipschitz)
        Hproposal = H @ proposal
        next_objective = float(0.5 * proposal @ Hproposal - b @ proposal + l1 * proposal.sum())
        if next_objective > objective + 1e-12 * max(1.0, abs(objective)):

            extrapolated = weights
            momentum = 1.0
            proposal = np.maximum(0.0, weights - (H @ weights - b + l1) / lipschitz)
            Hproposal = H @ proposal
            next_objective = float(0.5 * proposal @ Hproposal - b @ proposal + l1 * proposal.sum())
        gradient = Hproposal - b + l1
        violation = _kkt(proposal, gradient)
        converged = violation <= tol * scale
        if converged:
            return proposal, iteration, violation, True, next_objective
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        if np.dot(extrapolated - proposal, proposal - weights) > 0:
            extrapolated = proposal.copy()
            next_momentum = 1.0
        else:
            extrapolated = proposal + (momentum - 1.0) / next_momentum * (proposal - weights)
        weights, momentum, objective = proposal, next_momentum, next_objective
    return weights, max_iter, violation, False, objective


class SLIM:
    def __init__(self, l1=0.1, l2=10.0, max_iter=100, tol=1e-4, seed=42,
                 max_candidates=64, block_size=128):
        if not np.isfinite(l1) or l1 < 0 or not np.isfinite(l2) or l2 < 0:
            raise ValueError("l1 and l2 must be finite and nonnegative")
        for name, value in (("max_iter", max_iter), ("max_candidates", max_candidates),
                            ("block_size", block_size)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        self.l1, self.l2 = float(l1), float(l2)
        self.max_iter, self.tol, self.seed = int(max_iter), float(tol), int(seed)
        self.max_candidates, self.block_size = int(max_candidates), int(block_size)

    def fit(self, X):
        started = perf_counter()
        self.X_ = _positive_mask(X)
        Xcsc = self.X_.tocsc()
        XT = self.X_.T.tocsr()
        n_users, n_items = self.X_.shape
        counts = np.asarray(self.X_.sum(axis=0)).ravel().astype(np.float64)
        inv_norm = np.zeros(n_items, dtype=np.float64)
        np.divide(1.0, np.sqrt(counts), out=inv_norm, where=counts > 0)
        indptr = np.zeros(n_items + 1, dtype=np.int64)
        indices, values = [], []
        self.n_iter_ = np.zeros(n_items, dtype=np.int32)
        self.kkt_violation_ = np.zeros(n_items, dtype=np.float64)
        self.converged_ = np.ones(n_items, dtype=bool)
        self.candidate_counts_ = np.zeros(n_items, dtype=np.int32)
        objective_initial = 0.5 * self.X_.nnz
        objective_final = objective_initial
        max_block_nnz = max_block_bytes = max_gram_bytes = max_design_nnz = 0
        solved = 0
        optimization_seconds = candidate_seconds = 0.0
        for start in range(0, n_items, self.block_size):
            block_started = perf_counter()
            stop = min(start + self.block_size, n_items)
            block = (XT[start:stop] @ self.X_).tocsr()
            max_block_nnz = max(max_block_nnz, block.nnz)
            max_block_bytes = max(max_block_bytes, _sparse_bytes(block))
            candidate_seconds += perf_counter() - block_started
            for local, item in enumerate(range(start, stop)):
                left, right = block.indptr[local:local + 2]
                ids = block.indices[left:right]
                overlaps = block.data[left:right].astype(np.float64)
                keep = (ids != item) & (overlaps > 0)
                ids, overlaps = ids[keep], overlaps[keep]
                similarities = overlaps * inv_norm[item] * inv_norm[ids]
                chosen = _top_indices(similarities, ids, self.max_candidates)
                candidates, b = ids[chosen], overlaps[chosen]


                eligible = b > self.l1
                candidates, b = candidates[eligible], b[eligible]
                self.candidate_counts_[item] = len(candidates)
                if len(candidates):
                    solve_started = perf_counter()
                    design = Xcsc[:, candidates]
                    H = (design.T @ design).toarray().astype(np.float64)
                    H.flat[::len(candidates) + 1] += self.l2
                    max_gram_bytes = max(max_gram_bytes, H.nbytes)
                    max_design_nnz = max(max_design_nnz, design.nnz)
                    coefficients, n_iter, violation, converged, change = _solve_nonnegative(
                        H, b, self.l1, self.max_iter, self.tol
                    )
                    self.n_iter_[item] = n_iter
                    self.kkt_violation_[item] = violation
                    self.converged_[item] = converged
                    objective_final += change
                    nonzero = coefficients > 0
                    indices.append(candidates[nonzero].copy())
                    values.append(coefficients[nonzero].astype(np.float32))
                    indptr[item + 1] = indptr[item] + np.count_nonzero(nonzero)
                    optimization_seconds += perf_counter() - solve_started
                    solved += 1
                else:
                    indptr[item + 1] = indptr[item]
        all_indices = np.concatenate(indices) if indices else np.empty(0, dtype=np.int32)
        all_values = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
        self.weights_ = sparse.csc_matrix(
            (all_values, all_indices, indptr), shape=(n_items, n_items)
        )
        self.weights_.sort_indices()
        self.coef_ = self.weights_
        self.objective_history_ = [float(objective_initial), float(objective_final)]
        self.training_summary = {
            "model": "SLIM", "variant": "cosine-candidate-constrained nonnegative ElasticNet",
            "optimizer": "self-written projected FISTA with monotone restart",
            "fit_seconds": float(perf_counter() - started),
            "candidate_product_seconds": float(candidate_seconds),
            "optimization_and_gram_seconds": float(optimization_seconds),
            "n_users": int(n_users), "n_items": int(n_items),
            "input_nnz": int(self.X_.nnz), "weight_nnz": int(self.weights_.nnz),
            "l1": self.l1, "l2": self.l2, "max_iter": self.max_iter, "tol": self.tol,
            "max_candidates": self.max_candidates, "block_size": self.block_size,
            "seed": self.seed, "seed_note": "The optimizer and tie breaking are deterministic.",
            "objective_initial": float(objective_initial), "objective_final": float(objective_final),
            "objective_definition": "0.5||X-XW||_F^2 + l1*sum(W) + 0.5*l2*||W||_F^2",
            "objective_scope": "W>=0; diagonal=0; support limited to cosine candidates per target",
            "subproblems_optimized": int(solved),
            "subproblems_converged": int(np.count_nonzero(self.converged_)),
            "subproblems_not_converged": int(np.count_nonzero(~self.converged_)),
            "max_iterations_used": int(self.n_iter_.max(initial=0)),
            "mean_iterations_optimized": float(self.n_iter_.sum() / solved) if solved else 0.0,
            "restricted_kkt_max": float(self.kkt_violation_.max(initial=0.0)),
            "kkt_stop_rule": "KKT_inf <= tol * max(1, max(X_C.T @ x_i)) for each target",
            "cold_train_items": int(np.count_nonzero(counts == 0)),
            "input_csr_bytes": _sparse_bytes(self.X_),
            "weights_csc_bytes": _sparse_bytes(self.weights_),
            "max_block_nnz": int(max_block_nnz), "max_block_csr_bytes": int(max_block_bytes),
            "max_gram_bytes": int(max_gram_bytes), "max_design_nnz": int(max_design_nnz),
            "resource_note": "Sparse buffers and largest local Gram; not process peak RSS.",
        }
        return self

    def score(self, users, items=None):
        if not hasattr(self, "weights_"):
            raise RuntimeError("Call fit before score")
        users = np.asarray(users)
        if users.ndim != 1 or not np.issubdtype(users.dtype, np.integer):
            raise ValueError("users must be a one-dimensional integer array")
        if np.any(users < 0) or np.any(users >= self.X_.shape[0]):
            raise IndexError("users contains an out-of-range index")
        if items is None:
            weights = self.weights_
        else:
            items = np.asarray(items)
            if items.ndim != 1 or not np.issubdtype(items.dtype, np.integer):
                raise ValueError("items must be a one-dimensional integer array")
            if np.any(items < 0) or np.any(items >= self.X_.shape[1]):
                raise IndexError("items contains an out-of-range index")
            weights = self.weights_[:, items]
        return (self.X_[users] @ weights).toarray()
