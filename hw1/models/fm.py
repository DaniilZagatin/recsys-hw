from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from .ials import _index_array


def fm_second_order(values: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:

    weighted = values.unsqueeze(-1) * factors
    return 0.5 * (weighted.sum(dim=-2).square() - weighted.square().sum(dim=-2)).sum(dim=-1)


class _FMParameters(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_metadata: int, factors: int, seed: int):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, factors)
        self.item_factors = nn.Embedding(n_items, factors)
        self.item_linear = nn.Embedding(n_items, 1)
        self.metadata_factors = nn.Embedding(n_metadata, factors) if n_metadata else None
        self.metadata_linear = nn.Embedding(n_metadata, 1) if n_metadata else None
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for embedding in (self.user_factors, self.item_factors, self.metadata_factors):
                if embedding is not None:
                    embedding.weight.copy_(0.01 * torch.randn(embedding.weight.shape, generator=generator))
            self.item_linear.weight.zero_()
            if self.metadata_linear is not None:
                self.metadata_linear.weight.zero_()


class FactorizationMachine:


    def __init__(
        self,
        factors: int = 16,
        lr: float = 0.01,
        reg: float = 1e-4,
        epochs: int = 8,
        batch_size: int = 4096,
        seed: int = 42,
        device: str = "auto",
        use_metadata: bool = True,
    ) -> None:
        if (factors < 1 or epochs < 1 or batch_size < 1 or not np.isfinite(lr)
                or not np.isfinite(reg) or lr <= 0 or reg < 0):
            raise ValueError("Positive dimensions/epochs/batch_size/lr and nonnegative reg required")
        self.factors, self.lr, self.reg = int(factors), float(lr), float(reg)
        self.epochs, self.batch_size, self.seed = int(epochs), int(batch_size), int(seed)
        self.device, self.use_metadata = device, bool(use_metadata)

    def _metadata_bag(self, item_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.item_features_[item_ids]
        return (
            torch.as_tensor(features.indices.astype(np.int64, copy=False), device=self.device_),
            torch.as_tensor(features.indptr.astype(np.int64, copy=False), device=self.device_),
            torch.as_tensor(features.data, dtype=torch.float32, device=self.device_),
        )

    def _item_terms(self, item_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        ids = torch.as_tensor(item_ids, dtype=torch.long, device=self.device_)
        item_vectors = self._network.item_factors(ids)
        item_sum = item_vectors
        squared_norm = item_vectors.square().sum(dim=1)
        linear = self._network.item_linear(ids).squeeze(1)
        penalty = item_vectors.square().mean() + linear.square().mean()
        if self._network.metadata_factors is not None:
            indices, offsets, values = self._metadata_bag(item_ids)
            metadata_weights = self._network.metadata_factors.weight
            metadata_sum = F.embedding_bag(
                indices, metadata_weights, offsets, mode="sum",
                per_sample_weights=values, include_last_offset=True,
            )
            metadata_norms = F.embedding_bag(
                indices, metadata_weights.square().sum(dim=1, keepdim=True), offsets, mode="sum",
                per_sample_weights=values.square(), include_last_offset=True,
            ).squeeze(1)
            metadata_linear = F.embedding_bag(
                indices, self._network.metadata_linear.weight, offsets, mode="sum",
                per_sample_weights=values, include_last_offset=True,
            ).squeeze(1)
            item_sum = item_sum + metadata_sum
            squared_norm = squared_norm + metadata_norms
            linear = linear + metadata_linear
            if indices.numel():
                penalty = penalty + metadata_weights[indices].square().mean()
                penalty = penalty + self._network.metadata_linear.weight[indices].square().mean()
        cross = 0.5 * (item_sum.square().sum(dim=1) - squared_norm)
        return item_sum, linear + cross, penalty

    @staticmethod
    def _negative_items(
        users: np.ndarray, seen: sparse.csr_matrix, rng: np.random.Generator,
        catalog: np.ndarray | None = None,
    ) -> np.ndarray:


        catalog = np.unique(seen.indices) if catalog is None else catalog
        if not len(catalog):
            raise ValueError("Negative sampling requires a nonempty fit catalog")
        negative = rng.choice(catalog, size=len(users))
        invalid = np.asarray(seen[users, negative]).reshape(-1) != 0
        for _ in range(20):
            if not invalid.any():
                return negative
            negative[invalid] = rng.choice(catalog, size=int(invalid.sum()))
            invalid = np.asarray(seen[users, negative]).reshape(-1) != 0

        for user in np.unique(users[invalid]):
            positions = np.flatnonzero(invalid & (users == user))
            start, stop = seen.indptr[user:user + 2]
            available = np.setdiff1d(catalog, seen.indices[start:stop], assume_unique=True)
            if not len(available):
                raise ValueError("Negative sampling was asked to use a saturated user")
            negative[positions] = rng.choice(available, size=len(positions))
        return negative

    def fit(
        self,
        X_positive: sparse.spmatrix,
        item_features: sparse.spmatrix,
        seen: sparse.spmatrix | None = None,
    ) -> "FactorizationMachine":
        started = perf_counter()
        positive = sparse.csr_matrix(X_positive, dtype=np.float32, copy=True)
        positive.sum_duplicates()
        positive.eliminate_zeros()
        positive.sort_indices()
        if min(positive.shape) == 0 or positive.shape[1] < 2:
            raise ValueError("FM requires users and a catalog of at least two items")
        if np.any(~np.isfinite(positive.data)) or np.any(positive.data < 0):
            raise ValueError("Positive strengths must be finite and nonnegative")
        self.n_users_, self.n_items_ = positive.shape
        features = sparse.csr_matrix(item_features, dtype=np.float32, copy=True)
        features.sum_duplicates()
        features.eliminate_zeros()
        features.sort_indices()
        if features.shape[0] != self.n_items_ or np.any(~np.isfinite(features.data)):
            raise ValueError("Finite additional item features must have one row per catalog item")
        self.item_features_ = features if self.use_metadata else sparse.csr_matrix((self.n_items_, 0), dtype=np.float32)
        if seen is None:
            exclusion = positive.astype(bool)
        else:
            exclusion = sparse.csr_matrix(seen, copy=True)
            if exclusion.shape != positive.shape:
                raise ValueError("seen and positives must have the same shape")
            exclusion = exclusion.astype(bool) + positive.astype(bool)
        exclusion = exclusion.astype(bool).tocsr()
        exclusion.sum_duplicates()
        exclusion.eliminate_zeros()
        exclusion.sort_indices()
        self.negative_catalog_ = np.unique(exclusion.indices)
        users = np.repeat(np.arange(self.n_users_), np.diff(positive.indptr))
        items = positive.indices.copy()
        usable = np.diff(exclusion.indptr)[users] < len(self.negative_catalog_)
        train_users, train_items = users[usable], items[usable]
        self.warm_users_ = np.diff(positive.indptr) > 0
        self.device_ = torch.device(
            "cuda" if self.device == "auto" and torch.cuda.is_available()
            else "cpu" if self.device == "auto" else self.device
        )
        self._network = _FMParameters(
            self.n_users_, self.n_items_, self.item_features_.shape[1], self.factors, self.seed
        ).to(self.device_)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        self.historylosses: list[float] = []
        self.bpr_losses_: list[float] = []
        self._network.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(train_users))
            total_loss = total_bpr = 0.0
            for start in range(0, len(order), self.batch_size):
                selection = order[start:start + self.batch_size]
                batch_users = train_users[selection]
                batch_positive = train_items[selection]
                batch_negative = self._negative_items(batch_users, exclusion, rng, self.negative_catalog_)
                paired_items = np.concatenate([batch_positive, batch_negative])
                item_vectors, item_bias, item_penalty = self._item_terms(paired_items)
                n_batch = len(selection)
                user_vectors = self._network.user_factors(
                    torch.as_tensor(batch_users, dtype=torch.long, device=self.device_)
                )
                difference = (
                    user_vectors * (item_vectors[:n_batch] - item_vectors[n_batch:])
                ).sum(dim=1) + item_bias[:n_batch] - item_bias[n_batch:]
                bpr = F.softplus(-difference).mean()
                loss = bpr + 0.5 * self.reg * (user_vectors.square().mean() + item_penalty)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * n_batch
                total_bpr += float(bpr.detach()) * n_batch
            self.historylosses.append(total_loss / max(1, len(order)))
            self.bpr_losses_.append(total_bpr / max(1, len(order)))
        self._compile_scores()
        item_mass = np.asarray(positive.astype(bool).sum(axis=0)).reshape(-1)
        self.cold_item_scores_ = np.log1p(item_mass).astype(np.float32)
        scale = float(self.cold_item_scores_.max(initial=0))
        if scale:
            self.cold_item_scores_ /= scale
        self.training_summary = {
            "model": type(self).__name__, "objective": "sampled BPR + sampled L2 regularization",
            "n_users": self.n_users_, "n_items": self.n_items_,
            "positive_pairs": positive.nnz, "training_pairs": len(train_users),
            "skipped_saturated_positive_pairs": int(np.count_nonzero(~usable)),
            "negative_sampling_catalog_items": int(len(self.negative_catalog_)),
            "negative_sampling_catalog": "Only items observed in the current fit history",
            "metadata_features": self.item_features_.shape[1],
            "metadata_nnz": self.item_features_.nnz, "use_metadata": self.use_metadata,
            "factors": self.factors, "lr": self.lr, "reg": self.reg,
            "epochs": self.epochs, "batch_size": self.batch_size, "seed": self.seed,
            "device": str(self.device_), "fit_seconds": perf_counter() - started,
            "parameters": sum(parameter.numel() for parameter in self._network.parameters()),
            "losses": self.historylosses.copy(), "bpr_losses": self.bpr_losses_.copy(),
        }
        return self

    def _compile_scores(self) -> None:

        self._network.eval()
        self.item_factors_ = np.empty((self.n_items_, self.factors), dtype=np.float32)
        self.item_bias_ = np.empty(self.n_items_, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, self.n_items_, self.batch_size):
                stop = min(start + self.batch_size, self.n_items_)
                factors, bias, _ = self._item_terms(np.arange(start, stop))
                self.item_factors_[start:stop] = factors.cpu().numpy()
                self.item_bias_[start:stop] = bias.cpu().numpy()
            self.user_factors_ = self._network.user_factors.weight.detach().cpu().numpy().copy()

    def score(self, users: np.ndarray, items: np.ndarray | None = None) -> np.ndarray:
        users = _index_array(users, "users")
        items = np.arange(self.n_items_) if items is None else _index_array(items, "items")
        if np.any((items < 0) | (items >= self.n_items_)):
            raise IndexError("Items must belong to the fitted catalog")
        known = (users >= 0) & (users < self.n_users_)
        warm = np.zeros(len(users), dtype=bool)
        warm[known] = self.warm_users_[users[known]]
        result = np.broadcast_to(self.cold_item_scores_[items], (len(users), len(items))).copy()
        result[warm] = (
            self.user_factors_[users[warm]] @ self.item_factors_[items].T + self.item_bias_[items]
        )
        return result
