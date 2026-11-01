from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

if __package__:
    from .experiment import Experiment, MAIN_MODELS, _digest, _dump, _json_value
else:
    from experiment import Experiment, MAIN_MODELS, _digest, _dump, _json_value


FINAL_SUBDIRECTORY = "final_train_only"


def _source_validation_plan(experiment):

    experiment._ready()
    path = experiment.results_dir / "validation_plan.json"
    if not path.exists():
        return None, None
    plan = json.loads(path.read_text(encoding="utf-8"))
    if _json_value(experiment._plan()) != plan:
        raise ValueError("Current core code/data/grids differ from the saved validation plan")
    fingerprint = _digest(plan)
    existing = getattr(experiment, "run_fingerprint_", fingerprint)
    if existing != fingerprint:
        raise ValueError("Experiment has a different validation run fingerprint")
    return plan, fingerprint


def _validation_records(experiment, fingerprint):

    records, best, missing = [], {}, []
    for name in MAIN_MODELS:
        if name == "CustomIALS":
            if "IALS" not in best:
                missing.append("CustomIALS requires the complete IALS search")
                continue
            configs = [{**best["IALS"], "half_life": half_life}
                       for half_life in experiment.grids_["CustomIALS"]["half_life"]]
        else:
            configs = experiment.grids_[name]
        current = []
        for params in configs:
            trial_id = name + "_" + _digest(params)[:12]
            path = experiment.results_dir / "validation" / (trial_id + ".json")
            if not path.exists():
                missing.append(trial_id)
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record.get("run_fingerprint") != fingerprint
                    or record.get("phase") != "validation"
                    or record.get("model") != name
                    or record.get("trial_id") != trial_id
                    or record.get("params") != _json_value(params)):
                raise ValueError(f"Validation trial does not match its frozen plan: {trial_id}")
            if not np.isfinite(record["metrics"]["NDCG@10"]):
                raise ValueError(f"Non-finite validation metric in {trial_id}")
            if record["metrics"].get("n_eval_users", 0) < 1:
                raise ValueError(f"Empty validation cohort in {trial_id}")
            current.append(record)
            records.append(record)
        if len(current) == len(configs):
            selected = max(current, key=lambda item: item["metrics"]["NDCG@10"])
            best[name] = deepcopy(selected["params"])
    return records, best, missing


def _ensure_validation(experiment, progress=None):
    plan, fingerprint = _source_validation_plan(experiment)
    if plan is None:
        experiment.validate(progress=progress)
        plan, fingerprint = _source_validation_plan(experiment)
    records, best, missing = _validation_records(experiment, fingerprint)
    if missing or len(best) != len(MAIN_MODELS):
        experiment._log("Complete the missing validation trials; completed frozen trials are reused.")
        experiment.validate(progress=progress)
        plan, fingerprint = _source_validation_plan(experiment)
        records, best, missing = _validation_records(experiment, fingerprint)
    expected = experiment.search_space()["total_trials"]
    if missing or len(records) != expected or set(best) != set(MAIN_MODELS):
        raise ValueError(f"Need all {expected} validation trials and five complete model selections")
    users = {record["metrics"]["n_eval_users"] for record in records}
    candidates = {record["metrics"]["n_candidates"] for record in records}
    if len(users) != 1 or candidates != {len(experiment.split_.train_catalog)}:
        raise ValueError("Validation trials do not share the declared cohort/catalog")


    experiment.run_fingerprint_ = fingerprint
    experiment.best_parameters_ = deepcopy(best)
    rows = [experiment._row(record) for record in records]
    experiment.validation_results_ = pd.DataFrame(rows)
    experiment.validation_results_["selected"] = [
        record["params"] == best[record["model"]] for record in records
    ]
    experiment.validation_best_ = experiment.validation_results_.loc[
        experiment.validation_results_.selected
    ].copy()
    return plan, fingerprint


def describe_protocol_train_only(experiment):

    experiment._ready()
    path = experiment.results_dir / "protocol.json"
    protocol = (json.loads(path.read_text(encoding="utf-8"))
                if path.exists() else deepcopy(experiment.describe_protocol()))
    protocol.update({
        "final_policy": "train_only",
        "final_refit": "No train+validation refit. Fit only the initial train; validation only selects parameters.",
        "final_fit_history": "train",
        "final_past_observed_mask": "train",
        "final_target_cutoff_unix_seconds": int(experiment.split_.boundaries["validation_start"]),
        "final_target_cutoff_definition": "Initial train end, identical to validation fitting; not test_start",
        "final_observed_pairs": int(experiment.train_.observed.nnz),
        "final_positive_pairs": int(experiment.train_.positive.nnz),
        "candidate_pool": "Every item observed in the initial train; no sampled negatives in evaluation",
        "relevance": "Test rating>=4 within initial train users/items, excluding train-observed items",
        "metadata": "Reuse initial train item_features, vocabulary and scales; no validation feature refitting",
        "cold_policy": "Every model uses train-positive popularity for users with zero positive train history",
        "validation_interactions": "Not used for final training, target recency cutoff, metadata fitting, or seen-item masking",
        "test_seen_in_preliminary_run": True,
        "test_interpretation": "Previously examined holdout; final configurations are frozen before this train-only evaluation",
    })
    _dump(path, protocol)
    _dump(experiment.results_dir / FINAL_SUBDIRECTORY / "protocol.json", protocol)
    return protocol


def final_evaluate_train_only(experiment, *, progress=None, validation_progress=None, plots=False):

    validation_plan, validation_fingerprint = _ensure_validation(experiment, progress=validation_progress)
    helper_path = Path(__file__).resolve()
    helper_sha = hashlib.sha256(helper_path.read_bytes()).hexdigest()
    cutoff = int(experiment.split_.boundaries["validation_start"])
    configs = {name: deepcopy(experiment.best_parameters_[name]) for name in MAIN_MODELS}
    configs.update({"Popularity": {}, "FM_ID": deepcopy(experiment.best_parameters_["FM"]),
                    "CustomRatingOnly": {**experiment.best_parameters_["IALS"], "half_life": "inf"}})
    final_identity = {
        "validation_run_fingerprint": validation_fingerprint,
        "fit_history": "train", "past_observed_mask": "train", "truth": "test",
        "target_cutoff_unix_seconds": cutoff,
        "metadata_fit_history": "train", "configurations": configs,
        "final_code_sha256": helper_sha,
    }
    final_fingerprint = _digest(final_identity)
    final_dir = experiment.results_dir / FINAL_SUBDIRECTORY
    final_dir.mkdir(parents=True, exist_ok=True)
    frozen_plan = {
        **final_identity, "run_fingerprint": final_fingerprint,
        "fit_rows": int(len(experiment.split_.train)),
        "fit_observed_pairs": int(experiment.train_.observed.nnz),
        "fit_positive_pairs": int(experiment.train_.positive.nnz),
        "n_users": int(experiment.dataset_.n_users), "n_items": int(experiment.dataset_.n_items),
        "candidate_items": int(len(experiment.split_.train_catalog)),
        "selection_metric": "validation NDCG@10",
        "completed_validation_trials": int(len(experiment.validation_results_)),
        "validation_source_code_sha256": validation_plan["code_sha256"],
        "additional_source": {helper_path.name: helper_sha},
        "test_seen_in_preliminary_run": True,
        "bootstrap": {"unit": "eligible user", "replicates": 2000, "seed": experiment.seed},
    }
    frozen_path = final_dir / "frozen_test_plan.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding="utf-8")) != _json_value(frozen_plan):
        raise ValueError("This final directory already contains a different policy/configuration/source snapshot")
    _dump(frozen_path, frozen_plan)
    protocol = describe_protocol_train_only(experiment)
    final = Experiment(experiment.data_dir, final_dir, seed=experiment.seed)
    for attribute in ("dataset_", "split_", "train_", "item_features_", "feature_info_"):
        setattr(final, attribute, getattr(experiment, attribute))
    final.best_parameters_ = deepcopy(experiment.best_parameters_)
    final.validation_results_ = experiment.validation_results_.copy()
    final.validation_best_ = experiment.validation_best_.copy()
    final.run_fingerprint_ = final_fingerprint
    final.validation_run_fingerprint_ = validation_fingerprint
    final.final_policy_ = "train_only"
    final.final_matrices_ = experiment.train_
    final.environment()
    _dump(final_dir / "protocol.json", protocol)
    _dump(final_dir / "features_final.json", experiment.feature_info_)
    _dump(final_dir / "best_parameters.json", final.best_parameters_)
    final.validation_results_.to_csv(final_dir / "validation_trials.csv", index=False)
    final._log(f"Train-only final plan frozen: {experiment.train_.observed.nnz:,} observed pairs; "
               "reuse train features, train mask and validation_start cutoff. Validation is not fitted.")
    final.final_records_ = {}
    for name, params in configs.items():
        if name == "CustomRatingOnly" and params == configs["CustomIALS"]:
            result = deepcopy(final.final_records_["CustomIALS"])
            result.update({"model": name, "duplicate_of": "CustomIALS"})
        else:
            result = final._run(name, params, "test", experiment.train_, experiment.item_features_,
                                cutoff, experiment.split_.test)
        final.final_records_[name] = result
        final.final_results_ = pd.DataFrame([final._row(record) for record in final.final_records_.values()])
        final.final_results_.to_csv(final_dir / "final_comparison.csv", index=False)
        if progress is not None:
            progress(len(final.final_records_), len(configs), name)
    if final.final_results_.n_eval_users.nunique() != 1:
        raise AssertionError("Train-only final models use different evaluation cohorts")
    _dump(final_dir / "final_metrics.json", final.final_records_)
    final._paired_comparisons()
    if plots:
        final._result_figures()
    return final


def _bold_metrics(markdown):

    pattern = r"(?<![\w.*])([+-]?\d+\.\d{5}|\d+(?:\.\d+)?%)(?![\w.*])"
    return re.sub(pattern, lambda match: "**" + match.group(1) + "**", markdown)


def report_train_only(final):

    if getattr(final, "final_policy_", None) != "train_only":
        raise ValueError("Use the Experiment returned by final_evaluate_train_only")
    markdown = final.report()
    markdown = ("Финальные модели обучены только на train. Validation использована для выбора настроек, "
                "маска известных книг также построена только по train.\n\n" + markdown)
    markdown = _bold_metrics(markdown)
    (final.results_dir / "conclusions.md").write_text(markdown + "\n", encoding="utf-8")
    return markdown
