from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

if __package__:
    from .models.item_knn import ItemKNN
    from .models.slim import SLIM
    from .models.ials import IALS
    from .models.custom_ials import CustomIALS
    from .models.fm import FactorizationMachine
    from .utils.data import prepare_dataset
    from .utils.splitting import make_temporal_split, build_interactions
    from .utils.features import build_features
    from .utils.evaluation import evaluate
else:
    from models.item_knn import ItemKNN
    from models.slim import SLIM
    from models.ials import IALS
    from models.custom_ials import CustomIALS
    from models.fm import FactorizationMachine
    from utils.data import prepare_dataset
    from utils.splitting import make_temporal_split, build_interactions
    from utils.features import build_features
    from utils.evaluation import evaluate


MAIN_MODELS = ("ItemKNN", "SLIM", "IALS", "FM", "CustomIALS")
METRICS = ("NDCG@10", "Recall@10", "Coverage@10")


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value == np.inf else "-inf" if value == -np.inf else None
    if isinstance(value, Path):
        return str(value)
    return value


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _digest(value):
    encoded = json.dumps(_json_value(value), sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class _Popularity:
    def __init__(self, positive):
        self.item_scores = np.asarray(positive.sum(axis=0)).reshape(-1).astype(np.float32)
        self.training_summary = {"model": "Popularity", "definition": "Unique positive users per item", "fit_seconds": 0.0}

    def score(self, users, items=None):
        values = self.item_scores if items is None else self.item_scores[items]
        return np.broadcast_to(values, (len(users), len(values)))


class _CommonColdPolicy:

    def __init__(self, model, positive):
        self.model = model
        self.zero_positive = np.diff(positive.indptr) == 0
        self.popularity = np.asarray(positive.sum(axis=0)).reshape(-1).astype(np.float32)

    def score(self, users, items=None):
        scores = np.asarray(self.model.score(users, items)).copy()
        cold = self.zero_positive[users]
        if cold.any():
            scores[cold] = self.popularity if items is None else self.popularity[items]
        return scores


class Experiment:
    def __init__(self, data_dir, results_dir, seed=42):
        self.data_dir, self.results_dir = Path(data_dir), Path(results_dir)
        self.seed = int(seed)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        self.dataset_ = None
        self.best_parameters_ = {}
        self.validation_results_ = pd.DataFrame()
        self.final_results_ = pd.DataFrame()
        self.grids_ = {
            "ItemKNN": [{"k": k, "shrink": shrink}
                        for k in (50, 150, 300) for shrink in (0.0, 10.0, 50.0)],
            "SLIM": [{"l1": l1, "l2": l2, "max_candidates": 64, "max_iter": 300, "tol": 1e-4}
                     for l1 in (0.1, 1.0) for l2 in (1.0, 10.0, 100.0)],
            "IALS": [{"factors": factors, "reg": reg, "alpha": alpha, "epochs": 10}
                     for factors in (16, 32, 64) for reg in (0.1, 1.0, 10.0) for alpha in (1.0, 10.0, 40.0)],
            "FM": [{"factors": factors, "lr": lr, "epochs": 15, "reg": reg,
                    "batch_size": 2048, "device": "auto"}
                   for factors in (16, 32, 64) for lr in (0.001, 0.005) for reg in (1e-4, 1e-3)],
            "CustomIALS": {"base_parameters": "validation-selected IALS configuration",
                           "half_life": [180.0, 365.0, 730.0, 1095.0, "inf"]},
        }
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self._thread_limit = threadpool_limits(limits=2)

    @staticmethod
    def _log(message):
        print(message, flush=True)

    def search_space(self):

        counts = {name: len(configs) if isinstance(configs, list) else len(configs["half_life"])
                  for name, configs in self.grids_.items()}
        return {"primary_metric": "NDCG@10", "tie_rule": "First configuration in the declared order",
                "experiment_version": "expanded_hpo_v2", "grids": deepcopy(self.grids_),
                "trials_by_model": counts, "base_trials": sum(count for name, count in counts.items() if name != "CustomIALS"),
                "custom_trials": counts["CustomIALS"], "total_trials": sum(counts.values()),
                "controls": ["Popularity", "FM_ID with selected FM parameters", "CustomRatingOnly: half_life=inf with selected IALS parameters"],
                "test_seen_in_preliminary_run": True,
                "test_use": "Previously examined holdout, reused for the expanded study; choose parameters on validation and freeze before this run's final evaluation"}

    def environment(self):
        packages = {}
        for name in ("numpy", "scipy", "pandas", "torch", "matplotlib", "threadpoolctl", "nbformat", "ipykernel"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        result = {"python": sys.version, "platform": platform.platform(), "packages": packages,
                  "seed": self.seed, "torch_threads": torch.get_num_threads(), "blas_threads": 2,
                  "cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "deterministic_algorithms": True, "deterministic_warn_only": True,
                  "numerical_note": "GPU sparse reductions may warn about nondeterminism; seeds do not imply bitwise agreement across hardware."}
        _dump(self.results_dir / "environment.json", result)
        return result

    def prepare(self):
        if self.dataset_ is not None:
            return self.dataset_.profile
        self._log("Preparing both complete official archives; no user/item/activity sampling.")
        started = perf_counter()
        self.dataset_ = prepare_dataset(self.data_dir, profile_path=self.results_dir / "data_profile.json")
        self.split_ = make_temporal_split(self.dataset_.ratings)
        self.train_ = build_interactions(self.split_.train, self.dataset_.n_users, self.dataset_.n_items)
        self.item_features_, self.feature_info_ = build_features(
            self.dataset_.books, self.split_.train_catalog, self.dataset_.n_items
        )
        _dump(self.results_dir / "features_train.json", self.feature_info_)
        self.environment()
        self._log(f"Prepared {len(self.dataset_.ratings):,} rows, {self.dataset_.n_users:,} users, "
                  f"{self.dataset_.n_items:,} items in {perf_counter()-started:.1f}s; "
                  f"train catalog {len(self.split_.train_catalog):,}.")
        return self.dataset_.profile

    def _ready(self):
        if self.dataset_ is None:
            self.prepare()

    def _figure(self, name, figure=None):

        figure = plt.gcf() if figure is None else figure
        for axis in figure.axes:
            axis.set_axisbelow(True)
            axis.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.7, zorder=0)
        figure.tight_layout()
        figure.savefig(self.results_dir / "figures" / (name + ".png"), dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(figure)


    def eda(self, plot=False):

        self._ready()
        self._log("EDA: complete cleaned data, missing metadata, activity, sparsity and time.")
        ratings, books = self.dataset_.ratings, self.dataset_.books
        user_counts = np.bincount(ratings.user_id, minlength=self.dataset_.n_users)
        item_counts = np.bincount(ratings.item_id, minlength=self.dataset_.n_items)
        distribution = ratings.rating.value_counts().sort_index().rename_axis("rating").reset_index(name="rows")
        activity = pd.DataFrame({"user_interactions": pd.Series(user_counts).describe(percentiles=[.5, .9, .99]),
                                 "item_interactions": pd.Series(item_counts).describe(percentiles=[.5, .9, .99])})
        monthly = ratings.assign(month=pd.to_datetime(ratings.timestamp, unit="s", utc=True).dt.strftime("%Y-%m"))
        monthly = monthly.groupby("month", sort=True).agg(rows=("rating", "size"), mean_rating=("rating", "mean"))
        missing_rows = []
        for field in ("title", "authors", "language_code", "publisher", "format", "num_pages", "publication_year"):
            missing = books[field].map(lambda value: not bool(value) if isinstance(value, (list, tuple, str)) else bool(pd.isna(value)))
            missing_rows.append({"field": field, "missing_items": int(missing.sum()), "missing_fraction": float(missing.mean())})
        metadata_missing = pd.DataFrame(missing_rows)
        metadata_categories = pd.concat([
            books[field].fillna("").replace("", "<MISSING>").value_counts().head(12).rename_axis("value").reset_index(name="items").assign(field=field)
            for field in ("language_code", "publisher", "format")
        ], ignore_index=True)
        profile = self.dataset_.profile
        summary = pd.DataFrame([
            {"quantity": "raw review rows", "value": profile["reviews"]["raw_rows"]},
            {"quantity": "clean interaction rows", "value": len(ratings)},
            {"quantity": "users", "value": self.dataset_.n_users},
            {"quantity": "items", "value": self.dataset_.n_items},
            {"quantity": "sparsity", "value": 1 - len(ratings)/(self.dataset_.n_users*self.dataset_.n_items)},
            {"quantity": "rating-0 missing rows removed", "value": profile["reviews"]["rating_zero_missing_rows"]},
            {"quantity": "duplicate user/item rows removed", "value": profile["reviews"]["duplicate_pair_rows_removed"]},
        ])
        date_audit = pd.DataFrame(profile["reviews"]["date_fields"]).T
        tables = {"summary": summary, "rating_distribution": distribution, "activity": activity,
                  "metadata_missing": metadata_missing, "metadata_categories": metadata_categories,
                  "monthly_activity": monthly, "date_audit": date_audit}
        for name, table in tables.items():
            table.to_csv(self.results_dir / ("eda_" + name + ".csv"), index=True)
        self.eda_tables_ = tables
        self._eda_counts = {"activity": user_counts, "popularity": item_counts}
        if plot:
            for name in ("ratings", "activity", "popularity", "time", "metadata"):
                self.plot_eda(name)
        return tables

    def plot_eda(self, name):

        if name not in ("ratings", "activity", "popularity", "time", "metadata"):
            raise ValueError("Unknown EDA plot: " + str(name))
        self._ready()
        if not hasattr(self, "eda_tables_"):
            self.eda(plot=False)
        tables = self.eda_tables_
        if name == "metadata":
            figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            titles = {"language_code": "Язык", "publisher": "Издатель", "format": "Формат"}
            for axis, field in zip(axes, titles):
                top = tables["metadata_categories"].loc[
                    tables["metadata_categories"].field == field
                ].head(8).iloc[::-1]
                labels = top.value.astype(str).replace("<MISSING>", "Нет данных").str.slice(0, 27)
                axis.barh(labels, top["items"], color="#386cb0", zorder=2)
                axis.set(title=titles[field], xlabel="Число книг")
            self._figure("eda_metadata", figure)
            return
        figure, axis = plt.subplots(figsize=(8, 4.5))
        if name == "ratings":
            distribution = tables["rating_distribution"]
            axis.bar(distribution.rating, distribution.rows, color="#386cb0", zorder=2)
            axis.set(title="Распределение оценок", xlabel="Оценка", ylabel="Число отзывов")
            axis.set_xticks(np.arange(1, 6))
        elif name in ("activity", "popularity"):
            counts = self._eda_counts[name]
            nonzero = counts[counts > 0]
            if len(nonzero):
                bins = np.geomspace(1, int(nonzero.max()) + 1, 40)
                axis.hist(nonzero, bins=bins, color="#386cb0", zorder=2)
                axis.set_xscale("log")
                axis.set_yscale("log")
            title, ylabel = (("Активность пользователей", "Число пользователей")
                             if name == "activity" else ("Популярность книг", "Число книг"))
            axis.set(title=title, xlabel="Число оценок", ylabel=ylabel)
        else:
            monthly = tables["monthly_activity"]
            axis.plot(pd.to_datetime(monthly.index), monthly.rows, color="#386cb0", linewidth=1.6, zorder=2)
            axis.set(title="Отзывы по времени обновления", xlabel="Месяц", ylabel="Число отзывов")
        self._figure("eda_" + name, figure)

    def plot_validation(self):

        if self.validation_results_.empty:
            raise RuntimeError("Run validation before plotting its results")
        figure, axis = plt.subplots(figsize=(9, 4.8))
        models = [name for name in MAIN_MODELS if name in set(self.validation_results_.model)]
        rng = np.random.default_rng(self.seed)
        for index, name in enumerate(models):
            trials = self.validation_results_.loc[self.validation_results_.model == name]
            jitter = rng.uniform(-0.20, 0.20, size=len(trials))
            axis.scatter(index + jitter, trials["NDCG@10"], s=34, color="#386cb0", alpha=0.7,
                         edgecolors="white", linewidths=0.4, zorder=2,
                         label="Все конфигурации" if index == 0 else None)
            if "selected" in trials:
                selected = trials.loc[trials.selected]
            else:
                selected = trials.loc[[trials["NDCG@10"].idxmax()]]
            axis.scatter(np.full(len(selected), index), selected["NDCG@10"], marker="D", s=78,
                         color="#e78b35", edgecolors="black", linewidths=0.6, zorder=3,
                         label="Выбрано по validation" if index == 0 else None)
        axis.set_xticks(np.arange(len(models)), ["iALS" if name == "IALS" else "Custom iALS" if name == "CustomIALS" else name
                                               for name in models])
        axis.set(xlabel="Модель", ylabel="NDCG@10", title="Подбор параметров на validation")
        axis.legend(loc="best")
        self._figure("validation_search", figure)

    def describe_protocol(self):
        self._ready()
        result = {"split": self.split_.summary(), "split_definition": "Global chronological 70/15/15; tied timestamps never cross boundaries",
                  "training": "All rows; ItemKNN/SLIM/FM use rating>=4 positives; ALS confidence uses all observed ratings",
                  "relevance": "Future rating>=4, within initial-train users/catalog, excluding every past rated item",
                  "candidate_pool": "Every item observed in the initial train; no sampled negatives in evaluation",
                  "candidate_count": len(self.split_.train_catalog), "fixed_train_user_count": len(self.split_.train_users),
                  "positive_threshold": 4, "k": 10, "primary_selection_metric": "NDCG@10",
                  "metrics": {"NDCG@10": "Rewards relevant books near the top, normalized by each user's ideal ranking",
                              "Recall@10": "Fraction of each user's eligible future liked books recovered",
                              "Coverage@10": "Share of the fixed candidate catalog recommended at least once"},
                  "cold_policy": "For every model, users with zero positive fit-history receive the same positive-item popularity scores",
                  "final_refit": "Train+validation; past mask includes both. Initial train users and candidate catalog remain fixed",
                  "metadata": "Vocabulary/scales fit on stage history only; descriptive book fields, no rating aggregates or shelves",
                  "source_limitations": ["Review subset, not every Goodreads rating", "date_updated marks the available review version",
                                         "Static descriptive metadata is a release snapshot, not a versioned historical catalog"],
                  "bootstrap": {"unit": "eligible user", "replicates": 2000, "seed": self.seed, "interval": "paired percentile 95%"}}
        _dump(self.results_dir / "protocol.json", result)
        self._log("Protocol: fixed full train catalog, common cohorts and past masks; selection uses validation NDCG@10 only.")
        return result

    def _plan(self):
        root = Path(__file__).resolve().parent
        code = {}
        for path in [Path(__file__).resolve(), *sorted((root / "models").glob("*.py")), *sorted((root / "utils").glob("*.py"))]:
            code[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"version": 1, "seed": self.seed, "search_space": self.search_space(),
                "files": self.dataset_.profile["files"], "clean": self.dataset_.profile["clean"],
                "boundaries": self.split_.boundaries, "code_sha256": code,
                "fixed_catalog_items": len(self.split_.train_catalog), "evaluation_batch_size": 64,
                "primary_metric": "NDCG@10", "common_zero_positive_fallback": True}

    def _persist_plan(self):
        self._ready()
        plan = _json_value(self._plan())
        path = self.results_dir / "validation_plan.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Saved experiment plan differs from code/data/configuration. Use a new results directory for the new run.")
        _dump(path, plan)
        self.run_fingerprint_ = _digest(plan)

    def _new_model(self, name, params):
        params = dict(params)
        if name == "ItemKNN":
            return ItemKNN(**params)
        params.setdefault("seed", self.seed)
        if name == "SLIM": return SLIM(**params)
        if name == "IALS": return IALS(**params)
        if name in ("CustomIALS", "CustomRatingOnly"):
            if params.get("half_life") == "inf": params["half_life"] = np.inf
            return CustomIALS(**params)
        if name in ("FM", "FM_ID"):
            return FactorizationMachine(**params, use_metadata=name == "FM")
        raise KeyError(name)

    def _fit(self, name, params, matrices, features, cutoff):


        in_colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or Path("/content").is_dir()
        if len(self.dataset_.ratings) >= 100_000 and not in_colab:
            raise RuntimeError("Run full model training in Colab. Local execution is limited to small mathematical fixtures.")
        self._log(f"Fit {name}: {json.dumps(_json_value(params), ensure_ascii=False)}; "
                  f"{matrices.observed.nnz:,} observed / {matrices.positive.nnz:,} positive pairs.")
        if name == "Popularity":
            return _Popularity(matrices.positive)
        model = self._new_model(name, params)
        if name in ("ItemKNN", "SLIM"):
            model.fit(matrices.positive)
        elif name == "IALS":
            model.fit(matrices.observed, preferences=matrices.positive)
        elif name in ("CustomIALS", "CustomRatingOnly"):
            model.fit(matrices.observed, matrices.ratings, matrices.timestamps, cutoff)
        else:
            model.fit(matrices.positive, features, seen=matrices.observed)
        return model

    def _run(self, name, params, phase, matrices, features, cutoff, truth):
        trial_id = name + "_" + _digest(params)[:12]
        path = self.results_dir / phase / (trial_id + ".json")
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["run_fingerprint"] != self.run_fingerprint_:
                raise ValueError("Cached result belongs to a different experiment plan")
            self._log(f"Reuse completed {phase} result: {trial_id}.")
            return cached
        started = perf_counter()
        model = self._fit(name, params, matrices, features, cutoff)
        self._log(f"Evaluate {phase} {trial_id} on the common full candidate catalog.")
        final = phase == "test"
        (self.results_dir / phase).mkdir(parents=True, exist_ok=True)
        metrics = evaluate(_CommonColdPolicy(model, matrices.positive), truth, matrices.observed,
                           self.split_.train_users, self.split_.train_catalog, k=10, batch_size=64,
                           return_recommendations=final, return_user_metrics=final)
        if not np.isfinite(metrics["NDCG@10"]) or not metrics["n_eval_users"]:
            raise ValueError(f"No finite {phase} evaluation for {name}; inspect protocol coverage")
        examples = {}
        if final:
            user_ids, values = metrics.pop("user_ids"), metrics.pop("user_metrics")
            pd.DataFrame({"user_id": user_ids, "NDCG@10": values[:, 0], "Recall@10": values[:, 1]}).to_csv(
                self.results_dir / phase / (trial_id + "_users.csv"), index=False
            )
            recommendations = metrics.pop("recommendations")
            examples = {str(user): recommendations[user] for user in sorted(recommendations)[:3]}
        history = getattr(model, "historylosses", getattr(model, "objective_history_", []))
        result = {"run_fingerprint": self.run_fingerprint_, "trial_id": trial_id, "model": name,
                  "phase": phase, "params": _json_value(params), "metrics": metrics,
                  "training_summary": getattr(model, "training_summary", {}),
                  "historylosses": history, "examples": examples, "total_seconds": perf_counter() - started}
        _dump(path, result)
        self._log(f"Done {trial_id}: NDCG@10={metrics['NDCG@10']:.6f}, "
                  f"Recall@10={metrics['Recall@10']:.6f}; {result['total_seconds']:.1f}s.")
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return result

    @staticmethod
    def _row(result):
        metrics = result["metrics"]
        return {"model": result["model"], "trial_id": result["trial_id"],
                **{key: metrics[key] for key in (*METRICS, "n_eval_users", "n_candidates")},
                "fit_seconds": 0.0 if result.get("duplicate_of") else result["training_summary"].get("fit_seconds", 0),
                "total_seconds": 0.0 if result.get("duplicate_of") else result["total_seconds"],
                "duplicate_of": result.get("duplicate_of"),
                "params": json.dumps(result["params"], ensure_ascii=False, sort_keys=True)}

    def validate(self, only=None, plots=False, progress=None):

        self._ready()
        self._persist_plan()
        self.describe_protocol()
        requested = set(MAIN_MODELS if only is None else [only] if isinstance(only, str) else only)
        if not requested or requested - set(MAIN_MODELS):
            raise ValueError(f"Choose models from {MAIN_MODELS}")
        if "CustomIALS" in requested and "IALS" not in requested:
            als_paths = [self.results_dir / "validation" / ("IALS_" + _digest(params)[:12] + ".json")
                         for params in self.grids_["IALS"]]
            if not all(path.exists() for path in als_paths):
                raise ValueError("Complete validate_model('IALS') before the CustomIALS stage")
        results = []
        total = self.search_space()["total_trials"]
        self.best_parameters_ = {}
        for name in MAIN_MODELS:
            if name == "CustomIALS":
                if "IALS" not in self.best_parameters_:
                    continue
                configs = [{**self.best_parameters_["IALS"], "half_life": half_life}
                           for half_life in self.grids_["CustomIALS"]["half_life"]]
                _dump(self.results_dir / "custom_validation_grid.json", configs)
            else:
                configs = self.grids_[name]
            model_results = []
            for params in configs:
                if name not in requested:
                    path = self.results_dir / "validation" / (name + "_" + _digest(params)[:12] + ".json")
                    if not path.exists():
                        continue
                    result = json.loads(path.read_text(encoding="utf-8"))
                    if result["run_fingerprint"] != self.run_fingerprint_:
                        raise ValueError("Cached validation result belongs to a different experiment plan")
                else:
                    self._log(f"Validation {name}: {len(model_results)+1}/{len(configs)} configurations.")
                    result = self._run(name, params, "validation", self.train_, self.item_features_,
                                       self.split_.boundaries["validation_start"], self.split_.validation)
                results.append(result); model_results.append(result)
                pd.DataFrame([self._row(record) for record in results]).to_csv(self.results_dir / "validation_trials.csv", index=False)
                if progress is not None:
                    progress(len(results), total, name)
            if len(model_results) == len(configs):
                best = max(model_results, key=lambda record: record["metrics"]["NDCG@10"])
                self.best_parameters_[name] = deepcopy(best["params"])
        rows = [self._row(result) for result in results]
        self.validation_results_ = pd.DataFrame(rows)
        self.validation_results_["selected"] = [
            result["params"] == self.best_parameters_.get(result["model"]) for result in results
        ]
        if self.validation_results_.n_eval_users.nunique() != 1:
            raise AssertionError("Validation cohorts differ across models")
        self.validation_best_ = self.validation_results_.loc[self.validation_results_.selected].copy()
        self.validation_results_.to_csv(self.results_dir / "validation_trials.csv", index=False)
        _dump(self.results_dir / "best_parameters.json", self.best_parameters_)
        if plots:
            self.plot_validation()
        self._log(f"Validation: {len(rows)}/{total} completed, {len(self.best_parameters_)}/{len(MAIN_MODELS)} models selected. "
                  "The previously examined test is not used for this selection.")
        return self.validation_results_.copy()

    def validate_model(self, name, plots=False, progress=None):

        return self.validate(only=name, plots=plots, progress=progress)

    def final_evaluate(self, plots=False, progress=None):
        self._ready()
        if set(MAIN_MODELS) - self.best_parameters_.keys():
            self.validate(progress=progress)
        self._persist_plan()
        configs = {name: deepcopy(self.best_parameters_[name]) for name in MAIN_MODELS}
        configs.update({"Popularity": {}, "FM_ID": deepcopy(self.best_parameters_["FM"]),
                        "CustomRatingOnly": {**self.best_parameters_["IALS"], "half_life": "inf"}})
        final_plan = {"run_fingerprint": self.run_fingerprint_, "configurations": configs,
                      "fit_history": "train+validation", "fixed_evaluation_catalog": "initial train",
                      "selection_metric": "validation NDCG@10", "controls_declared_before_test": True,
                      "test_seen_in_preliminary_run": True,
                      "test_interpretation": "Reused holdout; final parameters are frozen before this expanded run's evaluation",
                      "user_bootstrap": {"replicates": 2000, "seed": self.seed}}
        frozen_path = self.results_dir / "frozen_test_plan.json"
        if frozen_path.exists() and json.loads(frozen_path.read_text(encoding="utf-8")) != _json_value(final_plan):
            raise ValueError("The final test plan is already frozen with different configurations")
        _dump(frozen_path, final_plan)
        self._log("Expanded-run final plan frozen. Refit train+validation, then evaluate each configuration once in this run; "
                  "the same holdout was already examined in the preliminary study.")
        history = pd.concat([self.split_.train, self.split_.validation], ignore_index=True)
        self.final_matrices_ = build_interactions(history, self.dataset_.n_users, self.dataset_.n_items)
        features, feature_info = build_features(self.dataset_.books, np.unique(self.final_matrices_.observed.indices), self.dataset_.n_items)
        _dump(self.results_dir / "features_final.json", feature_info)
        self.final_records_ = {}
        for name, params in configs.items():
            if name == "CustomRatingOnly" and params == configs["CustomIALS"]:
                result = deepcopy(self.final_records_["CustomIALS"])
                result.update({"model": name, "duplicate_of": "CustomIALS"})
                self._log("CustomRatingOnly equals selected CustomIALS; reuse that exact evaluation.")
            else:
                result = self._run(name, params, "test", self.final_matrices_, features,
                                   self.split_.boundaries["test_start"], self.split_.test)
            self.final_records_[name] = result
            table = pd.DataFrame([self._row(record) for record in self.final_records_.values()])
            table.to_csv(self.results_dir / "final_comparison.csv", index=False)
            if progress is not None:
                progress(len(self.final_records_), len(configs), name)
        self.final_results_ = table
        if self.final_results_.n_eval_users.nunique() != 1:
            raise AssertionError("Final cohorts differ across models")
        _dump(self.results_dir / "final_metrics.json", self.final_records_)
        self._paired_comparisons()
        if plots:
            self._result_figures()
        self._log("Final comparison complete; every model used the same users, catalog, relevance and past mask.")
        return self.final_results_.copy()

    def _paired_comparisons(self):
        rows = []
        for first, second in (("FM", "FM_ID"), ("CustomIALS", "IALS"), ("CustomIALS", "CustomRatingOnly")):
            left, right = self.final_records_[first], self.final_records_[second]
            a = pd.read_csv(self.results_dir / "test" / (left["trial_id"] + "_users.csv")).set_index("user_id")
            b = pd.read_csv(self.results_dir / "test" / (right["trial_id"] + "_users.csv")).set_index("user_id")
            if not a.index.equals(b.index):
                raise AssertionError("Paired comparison requires identical ordered users")
            difference = (a - b).to_numpy()
            rng = np.random.default_rng(self.seed)
            bootstrap = []
            for start in range(0, 2000, 100):
                indices = rng.integers(len(a), size=(min(100, 2000-start), len(a)))
                bootstrap.append(difference[indices].mean(axis=1))
            bootstrap = np.concatenate(bootstrap)
            for column, metric in enumerate(("NDCG@10", "Recall@10")):
                low, high = np.quantile(bootstrap[:, column], [.025, .975])
                rows.append({"comparison": first + " - " + second, "metric": metric,
                             "difference": float(difference[:, column].mean()),
                             "ci95_low": float(low), "ci95_high": float(high), "users": len(a),
                             "replicates": 2000, "seed": self.seed})
        self.paired_comparisons_ = pd.DataFrame(rows)
        self.paired_comparisons_.to_csv(self.results_dir / "paired_comparisons.csv", index=False)

    def _result_figures(self):

        self.plot_test_metrics()
        self.plot_training_losses()

    def plot_test_metrics(self):

        if self.final_results_.empty:
            raise RuntimeError("Run final_evaluate before plotting test results")
        table = self.final_results_.set_index("model")
        colors = ["#386cb0" if name in MAIN_MODELS else "#999999" for name in table.index]
        labels = {"IALS": "iALS", "CustomIALS": "Custom iALS", "Popularity": "Популярность",
                  "FM_ID": "FM только ID", "CustomRatingOnly": "Custom без затухания"}
        table = table.rename(index=labels)
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for axis, metric in zip(axes, ("NDCG@10", "Recall@10")):
            table[metric].plot.barh(ax=axis, color=colors, zorder=2)
            axis.set(title=metric + " на test", xlabel=metric, ylabel="")
        self._figure("test_comparison", figure)

    def plot_training_losses(self):

        if self.final_results_.empty:
            raise RuntimeError("Run final_evaluate before plotting training losses")
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        labels = {"IALS": "iALS", "CustomIALS": "Custom iALS", "FM_ID": "FM только ID",
                  "CustomRatingOnly": "Custom без затухания"}
        for name, result in self.final_records_.items():
            losses = np.asarray(result.get("historylosses", []), dtype=float)
            if len(losses) == 0 or result.get("duplicate_of") or name == "SLIM":
                continue
            panel = 1 if name in ("FM", "FM_ID") else 0
            axes[panel].plot(np.arange(1, len(losses) + 1), losses, marker=".", linewidth=1.6,
                             label=labels.get(name, name), zorder=2)
        axes[0].set(title="Функция потерь ALS", xlabel="Эпоха", ylabel="Взвешенная ошибка и штраф")
        axes[1].set(title="Функция потерь FM", xlabel="Эпоха", ylabel="BPR и штраф")
        for axis in axes:
            if axis.lines:
                axis.legend()
        self._figure("training_losses", figure)

    def report(self):
        if self.final_results_.empty:
            raise RuntimeError("Run final_evaluate() before generating conclusions")
        table = self.final_results_.set_index("model")
        main = table.loc[list(MAIN_MODELS)]
        ordered = main.sort_values("NDCG@10", ascending=False, kind="stable")
        winner, runner = ordered.index[:2]
        gap = float(ordered.iloc[0]["NDCG@10"] - ordered.iloc[1]["NDCG@10"])
        best_recall = main["Recall@10"].idxmax()
        fastest, slowest = main.fit_seconds.idxmin(), main.fit_seconds.idxmax()
        lines = [
            f"На test лучший NDCG@10 у {winner}: {table.loc[winner, 'NDCG@10']:.5f}. "
            f"У {runner} он равен {table.loc[runner, 'NDCG@10']:.5f}; разница {gap:.5f} является точечной, "
            "отдельный доверительный интервал между лидерами не рассчитывался. "
            f"Лучший Recall@10 у {best_recall}: {table.loc[best_recall, 'Recall@10']:.5f}. "
            f"Охват каталога у {winner} и {runner}: {table.loc[winner, 'Coverage@10']:.1%} "
            f"и {table.loc[runner, 'Coverage@10']:.1%}. "
            f"Обучение быстрее всего у {fastest} ({main.loc[fastest, 'fit_seconds']:.1f} с), "
            f"дольше всего у {slowest} ({main.loc[slowest, 'fit_seconds']:.1f} с)."
        ]
        descriptions, summaries, boundaries = {}, [], []
        for name in MAIN_MODELS:
            params = self.best_parameters_[name]
            if name == "ItemKNN":
                choice = f"k={params['k']}, shrink={params['shrink']:g}"
            elif name == "SLIM":
                choice = f"l1={params['l1']:g}, l2={params['l2']:g}"
            elif name == "IALS":
                choice = f"factors={params['factors']}, reg={params['reg']:g}, alpha={params['alpha']:g}"
            elif name == "FM":
                choice = f"factors={params['factors']}, lr={params['lr']:g}, reg={params['reg']:g}, epochs={params['epochs']}"
            else:
                half_life = params['half_life']
                choice = "h=∞" if half_life == "inf" else f"h={half_life:g} дней"
            trials = self.validation_results_.loc[self.validation_results_.model == name, "NDCG@10"]
            low, high = float(trials.min()), float(trials.max())
            descriptions[name] = f"{name}: {choice}; validation NDCG@10 от {low:.5f} до {high:.5f}"
            summaries.append({"model": name, "trials": len(trials), "validation_min": low,
                              "validation_max": high, "validation_range": high-low, "selected": choice})
            if name == "CustomIALS":
                if params['half_life'] in (self.grids_[name]['half_life'][0], self.grids_[name]['half_life'][-1]):
                    boundaries.append(name + " " + choice)
            else:
                for key in ("k", "shrink", "l1", "l2", "factors", "reg", "alpha", "lr"):
                    values = sorted({config[key] for config in self.grids_[name] if key in config})
                    if len(values) > 1 and params[key] in (values[0], values[-1]):
                        boundaries.append(f"{name} {key}={params[key]:g}")
        self.validation_summary_ = pd.DataFrame(summaries)
        self.validation_summary_.to_csv(self.results_dir / "hpo_summary.csv", index=False)
        lines.append(f"Проверено {len(self.validation_results_)} конфигураций. "
                     + ". ".join(descriptions[name] for name in ("ItemKNN", "SLIM", "IALS")) + ".")
        edge_note = (" На границах сетки выбраны " + "; ".join(boundaries[:4]) + "."
                     if boundaries else " Выбранные значения находятся внутри проверенных диапазонов.")
        lines.append(". ".join(descriptions[name] for name in ("FM", "CustomIALS")) + "."
                     + edge_note + " Это лучшие проверенные варианты, а не установленный глобальный оптимум.")
        effects = []
        for comparison, explanation in (("FM - FM_ID", "Метаданные FM"),
                                        ("CustomIALS - IALS", "Новый target против обычного ALS"),
                                        ("CustomIALS - CustomRatingOnly", "Затухание против target только по рейтингу")):
            row = self.paired_comparisons_.query("comparison == @comparison and metric == 'NDCG@10'").iloc[0]
            uncertainty = "интервал включает ноль" if row.ci95_low <= 0 <= row.ci95_high else "интервал не включает ноль"
            effects.append(f"{explanation}: ΔNDCG@10={row.difference:+.5f}, 95% paired CI "
                           f"[{row.ci95_low:+.5f}; {row.ci95_high:+.5f}], {uncertainty}")
        if self.best_parameters_["CustomIALS"]["half_life"] == "inf":
            effects.append("Выбрано h=∞, поэтому временное затухание в итоговой модели не используется")
        lines.append(". ".join(effects) + ".")
        slim = self.final_records_["SLIM"]["training_summary"]
        lines.append(f"SLIM сохраняет максимум {slim.get('max_candidates', 64)} соседей на объект, весь каталог оставлен. "
                     f"Критерий KKT не достигнут в {slim.get('subproblems_not_converged', 'n/a')} из "
                     f"{slim.get('subproblems_optimized', 'n/a')} оптимизируемых подзадач. "
                     "Это приближённое решение ограниченной модели.")
        lines.append("Ограничения: одно временное разбиение, один seed и конечная сетка. "
                     "Этот test уже был просмотрен в предварительном прогоне и повторно используется после расширения поиска; "
                     "его нельзя считать нетронутым подтверждением. Bootstrap отражает различия пользователей, "
                     "не периодов или обучения, без поправки на множественные сравнения. "
                     "Метрики относятся к исходному тёплому каталогу. date_updated описывает доступную версию отзыва; "
                     "исторических снимков рейтингов и метаданных нет, описательные признаки считаются статическими.")
        markdown = "\n\n".join(lines)
        (self.results_dir / "conclusions.md").write_text(markdown + "\n", encoding="utf-8")
        self._log("Conclusions generated from saved observed metrics and paired controls.")
        return markdown

    def example_recommendations(self, model=None):
        if self.final_results_.empty:
            raise RuntimeError("Run final_evaluate() before requesting recommendations")
        if model is None:
            model = self.validation_best_.sort_values("NDCG@10", ascending=False, kind="stable").iloc[0]["model"]
        record = self.final_records_[model]
        books = self.dataset_.books.set_index("item_id")
        test_positive = set(map(tuple, self.split_.test.loc[self.split_.test.rating >= 4, ["user_id", "item_id"]].to_numpy()))
        rows = []
        for user, items in record["examples"].items():
            for rank, item in enumerate(items, 1):
                rows.append({"model": model, "user_id": int(user), "rank": rank, "item_id": int(item),
                             "source_book_id": books.loc[item, "source_item_id"],
                             "title": books.loc[item, "title"] or "<metadata missing>",
                             "future_relevant": (int(user), int(item)) in test_positive})
        result = pd.DataFrame(rows)
        result.to_csv(self.results_dir / "example_recommendations.csv", index=False)
        self._log(f"Examples: {model}, first three eligible encoded user IDs; model chosen by validation.")
        return result
