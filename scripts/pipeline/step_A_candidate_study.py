from contextlib import redirect_stderr, redirect_stdout
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from ts_benchmark.baselines.self_impl.CATCH.CATCH import CATCH
from ts_benchmark.models import get_models

TOP_K = 25


def load_timeseries(csv_path: Path):
    raw = pd.read_csv(csv_path)
    raw["_time"] = pd.to_datetime(raw["date"], errors="coerce")
    values = raw.pivot(index="_time", columns="cols", values="data")
    values = values.sort_index()
    return values, None


def clean_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.interpolate(limit_direction="both")
    features = features.ffill().bfill()

    nunique = features.nunique(dropna=True)
    keep_cols = list(nunique[nunique > 1].index)
    if not keep_cols:
        raise ValueError("All feature columns are constant; anomaly ranking is not meaningful.")
    return features[keep_cols]


def align_scores(scores: np.ndarray, length: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) == length:
        return scores
    if len(scores) > length:
        return scores[:length]
    return np.pad(scores, (0, length - len(scores)), mode="constant", constant_values=np.nan)


def catch_scores(
        features: pd.DataFrame,
        catch_params: Optional[dict],
) -> np.ndarray:
    params = {
        "seq_len": 96,
        "num_epochs": 3,
        "batch_size": 64,
        "lr": 0.0001,
        "Mlr": 0.00001,
        "d_model": 16,
        "d_ff": 64,
        "n_heads": 16,
        "e_layers": 3,
        "cf_dim": 32,
        "head_dim": 32,
        "patch_size": 16,
        "patch_stride": 8,
        "auxi_lambda": 0.1,
        "dc_lambda": 0.1,
    }
    if catch_params:
        params.update(catch_params)

    max_seq_len = max(3, int(len(features) * 0.6))
    params["seq_len"] = min(int(params["seq_len"]), max_seq_len)
    params["batch_size"] = min(int(params["batch_size"]), max(1, len(features)))

    print("CATCH startet", flush=True)
    model = CATCH(**params)
    model.device = torch.device("cpu")
    model.detect_fit(features, None)
    raw_scores, _ = model.detect_score(features)
    print("CATCH fertig", flush=True)
    return align_scores(raw_scores, len(features))


ANOMALY_TYPE_METHODS: Dict[str, List[str]] = {
    "global_point": [
        "itrans",
        "calf",
        "units",
        "datamole",
        "gpt4ts",
    ],
    "contextual_point": [
        "dp",
        "zms",
        "units",
        "isolation_forest",
        "calf",
    ],
    "subsequence_pattern": [
        "kmeans",
        "dwt",
        "calf",
        "itrans",
        "ocsvm",
    ],
    "multivariate_dependency": [
        "catch",
        "tsnet",
        "dagmm",
        "knn",
        "kmeans",
    ],
}

UNIVARIATE_GROUPS = {"global_point", "contextual_point", "subsequence_pattern"}
MULTIVARIATE_GROUPS = {"multivariate_dependency"}

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "itrans": {
        "model_name": "time_series_library.iTransformer",
        "adapter": "transformer_adapter",
        "model_hyper_params": {
            "batch_size": 16,
            "d_ff": 128,
            "d_model": 64,
            "e_layers": 3,
            "horizon": 0,
            "norm": True,
            "num_epochs": 3,
            "seq_len": 100,
        },
    },
    "calf": {
        "model_name": "LLM.CALFModel",
        "adapter": "llm_adapter",
        "model_hyper_params": {
            "d_ff": 768,
            "d_model": 768,
            "dataset": "uv",
            "dropout": 0.3,
            "gpt_layer": 6,
            "horizon": 1,
            "lr": 0.0005,
            "n_heads": 4,
            "norm": True,
            "sampling_rate": 0.05,
            "seq_len": 96,
        },
    },
    "units": {
        "model_name": "pre_train.UniTS",
        "adapter": "PreTrain_adapter",
        "model_hyper_params": {
            "horizon": 1,
            "is_train": 1,
            "norm": True,
            "num_epochs": 3,
            "sampling_rate": 0.05,
            "seq_len": 96,
        },
    },
    "datamole": {
        "model_name": "pre_train.DadaModel",
        "adapter": "PreTrain_adapter",
        "model_hyper_params": {
            "horizon": 1,
            "is_train": 1,
            "lr": 0.005,
            "norm": True,
            "sampling_rate": 0.05,
            "seq_len": 100,
        },
    },
    "gpt4ts": {
        "model_name": "LLM.GPT4TSModel",
        "adapter": "llm_adapter",
        "model_hyper_params": {
            "horizon": 1,
            "norm": True,
            "num_epochs": 3,
            "sampling_rate": 0.05,
            "seq_len": 100,
        },
    },
    "dp": {
        "model_name": "merlion.DeepPointAnomalyDetector",
        "model_hyper_params": {"enable_threshold": 0},
    },
    "zms": {
        "model_name": "merlion.ZMS",
        "model_hyper_params": {},
    },
    "isolation_forest": {
        "model_name": "merlion.IsolationForest",
        "model_hyper_params": {},
    },
    "kmeans": {
        "model_name": "self_impl.KMeans",
        "model_hyper_params": {"window_size": 100},
    },
    "dwt": {
        "model_name": "self_impl.DWT_MLEAD",
        "model_hyper_params": {},
    },
    "ocsvm": {
        "model_name": "tods.ocsvmski",
        "model_hyper_params": {},
    },
    "tsnet": {
        "model_name": "time_series_library.TimesNet",
        "adapter": "transformer_adapter",
        "model_hyper_params": {
            "batch_size": 32,
            "d_ff": 64,
            "d_model": 64,
            "e_layers": 2,
            "horizon": 0,
            "norm": True,
            "num_epochs": 10,
            "seq_len": 100,
        },
    },
    "dagmm": {
        "model_name": "merlion.DAGMM",
        "model_hyper_params": {},
    },
    "knn": {
        "model_name": "tods.knnski",
        "model_hyper_params": {},
    },
}

CATCH_PARAMS = {
    "Mlr": 1e-05,
    "auxi_lambda": 0.1,
    "batch_size": 64,
    "cf_dim": 32,
    "d_ff": 64,
    "d_model": 16,
    "dc_lambda": 0.1,
    "e_layers": 3,
    "head_dim": 32,
    "lr": 0.0001,
    "n_heads": 16,
    "num_epochs": 3,
    "patch_size": 16,
    "patch_stride": 8,
    "seq_len": 96,
}


def normalize_scores(raw_scores):
    normalized = pd.DataFrame(index=raw_scores.index)

    for column in raw_scores.columns:
        series = pd.to_numeric(raw_scores[column], errors="coerce")
        finite_mask = np.isfinite(series.to_numpy(dtype=float))
        percentile = pd.Series(np.nan, index=raw_scores.index, dtype=float)

        if finite_mask.any():
            finite_index = series.index[finite_mask]
            finite_values = series.loc[finite_index]
            percentile.loc[finite_index] = finite_values.rank(
                method="average",
                pct=True,
                ascending=True,
            )

        normalized[column] = percentile

    return normalized


def build_candidate_tables(
        timestamps: "pd.Series",
        raw_scores: "pd.DataFrame",
        normalized_scores: "pd.DataFrame",
        top_k: int,
) -> Dict[str, "pd.DataFrame"]:
    method_names = list(raw_scores.columns)
    raw = raw_scores.copy()
    norm = normalized_scores.copy()

    for method in method_names:
        raw[method] = pd.to_numeric(raw[method], errors="coerce")
        norm[method] = pd.to_numeric(norm[method], errors="coerce")

    finite_columns = {}
    for column in norm.columns:
        finite_columns[column] = np.isfinite(norm[column].to_numpy(dtype=float))
    finite_mask = pd.DataFrame(finite_columns, index=norm.index)

    available_method_count = finite_mask.sum(axis=1).astype(int)
    missing_method_count = len(method_names) - available_method_count

    consensus_score = norm.where(finite_mask).mean(axis=1, skipna=True)
    consensus_score = consensus_score.where(available_method_count > 0, np.nan)

    consensus_rank = consensus_score.rank(
        method="min",
        ascending=False,
        na_option="keep",
    )

    timestamp_values = pd.Series(timestamps).reset_index(drop=True)
    row_values = pd.Series(range(len(norm)), index=norm.index)

    def join_available(row_idx) -> str:
        return ";".join(
            method
            for method in method_names
            if bool(finite_mask.loc[row_idx, method])
        )

    def join_missing(row_idx) -> str:
        return ";".join(
            method
            for method in method_names
            if not bool(finite_mask.loc[row_idx, method])
        )

    method_ranks: Dict[str, "pd.Series"] = {}
    topk_by_method_rows = []
    method_top_rows: Dict[str, set] = {}

    for method in method_names:
        method_score = raw[method]
        method_pct = norm[method]
        method_rank = method_pct.rank(
            method="min",
            ascending=False,
            na_option="keep",
        )
        method_ranks[method] = method_rank

        method_valid = pd.DataFrame(
            {
                "method": method,
                "method_rank": method_rank,
                "row": row_values.to_numpy(),
                "timestamp": timestamp_values.to_numpy(),
                "score": method_score.to_numpy(),
                "score_pct": method_pct.to_numpy(),
                "consensus_score": consensus_score.to_numpy(),
                "consensus_rank": consensus_rank.astype("Int64").to_numpy(),
                "available_method_count": available_method_count.to_numpy(),
                "missing_method_count": missing_method_count.to_numpy(),
                "methods_available": [join_available(idx) for idx in norm.index],
                "methods_missing": [join_missing(idx) for idx in norm.index],
            }
        )
        method_valid = method_valid.dropna(subset=["score_pct"]).copy()
        method_valid = method_valid.sort_values(
            ["score_pct", "score", "row"],
            ascending=[False, False, True],
        ).head(top_k)
        method_valid["method_rank"] = method_valid["method_rank"].astype("Int64")
        method_valid["consensus_rank"] = method_valid["consensus_rank"].astype("Int64")

        method_top_rows[method] = set(method_valid["row"].astype(int).tolist())
        topk_by_method_rows.append(method_valid)

    topk_by_method = (
        pd.concat(topk_by_method_rows, ignore_index=True)
        if topk_by_method_rows
        else pd.DataFrame()
    )

    def triggered_methods_for_row(row: int) -> List[str]:
        return [
            method
            for method in method_names
            if row in method_top_rows.get(method, set())
        ]

    consensus_valid = pd.DataFrame(
        {
            "consensus_rank": consensus_rank,
            "row": row_values.to_numpy(),
            "timestamp": timestamp_values.to_numpy(),
            "consensus_score": consensus_score.to_numpy(),
            "available_method_count": available_method_count.to_numpy(),
            "missing_method_count": missing_method_count.to_numpy(),
            "methods_available": [join_available(idx) for idx in norm.index],
            "methods_missing": [join_missing(idx) for idx in norm.index],
        }
    )
    consensus_valid = consensus_valid.dropna(subset=["consensus_score"]).copy()
    consensus_valid = consensus_valid.sort_values(
        ["consensus_score", "available_method_count", "row"],
        ascending=[False, False, True],
    ).head(top_k)

    consensus_rows = []
    for item in consensus_valid.itertuples(index=False):
        row = int(item.row)
        triggered = triggered_methods_for_row(row)
        triggered_ranks = [
            method_ranks[method].iloc[row]
            for method in triggered
            if not pd.isna(method_ranks[method].iloc[row])
        ]
        best_method_rank = int(min(triggered_ranks)) if triggered_ranks else None

        consensus_rows.append(
            {
                "consensus_rank": int(item.consensus_rank),
                "row": row,
                "timestamp": item.timestamp,
                "consensus_score": item.consensus_score,
                "available_method_count": int(item.available_method_count),
                "missing_method_count": int(item.missing_method_count),
                "methods_available": item.methods_available,
                "methods_missing": item.methods_missing,
                "methods_in_top_k": len(triggered),
                "methods_triggered": ";".join(triggered),
                "best_method_rank": best_method_rank,
            }
        )

    consensus_topk = pd.DataFrame(consensus_rows)

    return {
        "consensus_topk": consensus_topk,
        "topk_by_method": topk_by_method,
    }


def to_score_array(score_output) -> "np.ndarray":
    if isinstance(score_output, dict):
        score_output = next(iter(score_output.values()))
    if isinstance(score_output, (tuple, list)):
        score_output = score_output[0]
    return np.asarray(score_output, dtype=float).reshape(-1)


def run_tab_model_scores(
        features: "pd.DataFrame",
        model_config: Dict[str, object],
        method_name: str,
) -> "np.ndarray":
    config_model = dict(model_config)
    config_model["model_hyper_params"] = dict(config_model.get("model_hyper_params", {}))
    config = {"models": [config_model]}
    model_factory = get_models(config)[0]
    model = model_factory()

    if hasattr(model, "detect_fit"):
        model.detect_fit(features, features)
    elif hasattr(model, "fit"):
        model.fit(features, None)
    else:
        raise ValueError(f"Model {method_name} does not implement detect_fit or fit")

    if not hasattr(model, "detect_score"):
        raise ValueError(f"Model {method_name} does not implement detect_score")

    score_output = model.detect_score(features)
    scores = to_score_array(score_output)
    return align_scores(scores, len(features))


def compute_multivariate_scores(
        features: "pd.DataFrame",
        methods: Iterable[str],
        catch_params: dict,
        log_dir: Path,
) -> "pd.DataFrame":
    scores: Dict[str, "np.ndarray"] = {}
    methods = list(methods)

    for method in methods:
        started_at = time.perf_counter()
        print(f"Running multivariate method: {method}", flush=True)
        try:
            log_path = log_dir / f"multivariate_{method}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\n--- {method} ---\n")
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    if method == "catch":
                        method_scores = catch_scores(features, catch_params)
                    else:
                        model_config = dict(MODEL_CONFIGS[method])
                        model_config["model_hyper_params"] = dict(
                            model_config.get("model_hyper_params", {})
                        )
                        method_scores = run_tab_model_scores(features, model_config, method)
        except Exception as exc:
            print(f"Multivariate method failed: {method}", flush=True)
            print(exc, flush=True)
            raise

        elapsed = time.perf_counter() - started_at
        scores[method] = method_scores
        print(f"Finished multivariate method {method} in {elapsed:.2f}s", flush=True)
    return pd.DataFrame(scores, index=features.index)


def compute_univariate_scores(
        features: "pd.DataFrame",
        methods: Iterable[str],
        log_dir: Path,
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    scores: Dict[str, "np.ndarray"] = {}
    channel_scores: Dict[str, "np.ndarray"] = {}
    methods = list(methods)

    for method in methods:
        started_at = time.perf_counter()
        print(f"Running univariate method: {method}", flush=True)

        column_score_arrays = []
        model_config = dict(MODEL_CONFIGS[method])
        model_config["model_hyper_params"] = dict(model_config.get("model_hyper_params", {}))
        log_path = log_dir / f"univariate_{method}.log"

        for column in features.columns:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"\n--- {method} column {column} ---\n")
                    with redirect_stdout(log_file), redirect_stderr(log_file):
                        series_scores = run_tab_model_scores(
                            features[[column]],
                            model_config,
                            method,
                        )
            except Exception as exc:
                print(f"Univariate method failed: {method} column {column}", flush=True)
                print(exc, flush=True)
                raise

            column_score_arrays.append(series_scores)
            channel_scores[f"{method}__{column}"] = series_scores

        if not column_score_arrays:
            raise RuntimeError(f"Univariate method {method} failed on all columns")

        stacked = np.vstack(column_score_arrays)

        method_scores = np.full(stacked.shape[1], np.nan, dtype=float)
        finite_positions = np.any(np.isfinite(stacked), axis=0)
        if np.any(finite_positions):
            method_scores[finite_positions] = np.nanmax(
                stacked[:, finite_positions],
                axis=0,
            )
        scores[method] = method_scores

        elapsed = time.perf_counter() - started_at
        print(f"Finished univariate method {method} in {elapsed:.2f}s", flush=True)

    return (
        pd.DataFrame(scores, index=features.index),
        pd.DataFrame(channel_scores, index=features.index),
    )


def build_customer_univariate_outputs(
        features: "pd.DataFrame",
        timestamps: "pd.Series",
        univariate_channel_scores: "pd.DataFrame",
        method_groups: Dict[str, List[str]],
        top_k: int,
        variant_dir: Path,
        variant_name: str,
) -> None:
    root_dir = variant_dir / "customer_univariate"
    root_dir.mkdir(parents=True, exist_ok=True)

    if univariate_channel_scores.empty:
        print("No customer-specific univariate scores available.", flush=True)
        return

    summary_consensus_frames = []
    summary_topk_by_method_frames = []

    for customer_column in features.columns:
        customer_label = str(customer_column)
        customer_dir = root_dir / f"customer_{customer_label}"
        customer_dir.mkdir(parents=True, exist_ok=True)

        for group_name, group_methods in method_groups.items():
            if group_name not in UNIVARIATE_GROUPS:
                continue

            method_score_columns = {}
            for method in group_methods:
                score_col = f"{method}__{customer_label}"
                if score_col in univariate_channel_scores.columns:
                    method_score_columns[method] = univariate_channel_scores[score_col]

            available_methods = list(method_score_columns.keys())
            if not available_methods:
                continue

            customer_raw_scores = pd.DataFrame(
                method_score_columns,
                index=features.index,
            )
            customer_normalized_scores = normalize_scores(customer_raw_scores)

            outputs = build_candidate_tables(
                timestamps=timestamps,
                raw_scores=customer_raw_scores,
                normalized_scores=customer_normalized_scores,
                top_k=top_k,
            )

            group_dir = customer_dir / group_name
            group_dir.mkdir(parents=True, exist_ok=True)
            outputs["consensus_topk"].to_csv(group_dir / "consensus_topk.csv", index=False)
            outputs["topk_by_method"].to_csv(group_dir / "topk_by_method.csv", index=False)

            consensus = outputs.get("consensus_topk")
            if consensus is not None and not consensus.empty:
                consensus_export = consensus.copy()
                consensus_export.insert(0, "variant", variant_name)
                consensus_export.insert(1, "customer", customer_label)
                consensus_export.insert(2, "anomaly_type", group_name)
                summary_consensus_frames.append(consensus_export)

            topk_by_method = outputs.get("topk_by_method")
            if topk_by_method is not None and not topk_by_method.empty:
                topk_export = topk_by_method.copy()
                topk_export.insert(0, "variant", variant_name)
                topk_export.insert(1, "customer", customer_label)
                topk_export.insert(2, "anomaly_type", group_name)
                summary_topk_by_method_frames.append(topk_export)

    if summary_consensus_frames:
        summary_consensus = pd.concat(summary_consensus_frames, ignore_index=True)
        summary_consensus.to_csv(root_dir / "customer_univariate_consensus_topk.csv", index=False)

    if summary_topk_by_method_frames:
        summary_topk_by_method = pd.concat(summary_topk_by_method_frames, ignore_index=True)
        summary_topk_by_method.to_csv(
            root_dir / "customer_univariate_topk_by_method.csv",
        index=False,
        )


def prepare_features(features):
    features = clean_features(features)
    features = features.copy()
    features.index = pd.to_datetime(features.index)
    features.index.name = "date"
    timestamps = pd.Series(features.index.astype(str), index=pd.RangeIndex(len(features)))

    return features, timestamps


def run_variant(
        variant_name: str,
        csv_path: Path,
        output_dir: Path,
        method_groups: Dict[str, List[str]],
        top_k: int,
        catch_params: dict,
) -> Dict[str, Dict[str, "pd.DataFrame"]]:
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    method_log_dir = variant_dir / "method_logs"

    print(f"Variant {variant_name} startet", flush=True)

    features_raw, _ = load_timeseries(csv_path)
    features, timestamps = prepare_features(features_raw)

    univariate_methods = []
    multivariate_methods = []
    for group_name, group_methods in method_groups.items():
        target = univariate_methods if group_name in UNIVARIATE_GROUPS else multivariate_methods
        for method in group_methods:
            if method not in target:
                target.append(method)

    if univariate_methods:
        (
            univariate_scores,
            univariate_channel_scores,
        ) = compute_univariate_scores(
            features=features,
            methods=univariate_methods,
            log_dir=method_log_dir,
        )
    else:
        univariate_scores = pd.DataFrame(index=features.index)
        univariate_channel_scores = pd.DataFrame(index=features.index)

    if multivariate_methods:
        multivariate_scores = compute_multivariate_scores(
            features=features,
            methods=multivariate_methods,
            catch_params=catch_params,
            log_dir=method_log_dir,
        )
    else:
        multivariate_scores = pd.DataFrame(index=features.index)

    combined_scores = univariate_scores.copy()
    duplicate_methods = set(univariate_scores.columns).intersection(multivariate_scores.columns)
    if duplicate_methods:
        combined_scores = combined_scores.rename(
            columns={method: f"{method}_univariate" for method in duplicate_methods}
        )
    for method in multivariate_scores.columns:
        if method in duplicate_methods:
            combined_scores[f"{method}_multivariate"] = multivariate_scores[method]
        else:
            combined_scores[method] = multivariate_scores[method]

    if combined_scores.empty:
        raise RuntimeError("No method completed successfully; cannot build rankings.")

    normalized_scores = normalize_scores(combined_scores)

    if not univariate_channel_scores.empty:
        build_customer_univariate_outputs(
            features=features,
            timestamps=timestamps,
            univariate_channel_scores=univariate_channel_scores,
            method_groups=method_groups,
            top_k=top_k,
            variant_dir=variant_dir,
            variant_name=variant_name,
        )

    all_outputs = build_candidate_tables(
        timestamps=timestamps,
        raw_scores=combined_scores,
        normalized_scores=normalized_scores,
        top_k=top_k,
    )
    all_outputs["consensus_topk"].to_csv(
        variant_dir / "all_methods_consensus_topk.csv",
        index=False,
    )

    group_outputs: Dict[str, Dict[str, "pd.DataFrame"]] = {}
    for group_name, group_methods in method_groups.items():
        if group_name in UNIVARIATE_GROUPS:
            score_frame = univariate_scores
        else:
            score_frame = multivariate_scores

        available_methods = [m for m in group_methods if m in score_frame.columns]
        if not available_methods:
            print(f"Gruppe ohne Ergebnisse: {group_name}", flush=True)
            continue

        group_raw_scores = score_frame[available_methods]
        group_normalized_scores = normalize_scores(group_raw_scores)
        outputs = build_candidate_tables(
            timestamps=timestamps,
            raw_scores=group_raw_scores,
            normalized_scores=group_normalized_scores,
            top_k=top_k,
        )
        group_outputs[group_name] = outputs

        group_dir = variant_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        outputs["consensus_topk"].to_csv(group_dir / "consensus_topk.csv", index=False)
        outputs["topk_by_method"].to_csv(group_dir / "topk_by_method.csv", index=False)

    print(f"Variant {variant_name} fertig", flush=True)
    return group_outputs



def compare_raw_vs_log1p(
        results: Dict[str, Dict[str, Dict[str, "pd.DataFrame"]]],
        output_dir: Path,
        top_k: int,
) -> None:
    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if "raw" not in results or "log1p" not in results:
        print("raw/log1p Vergleich wird übersprungen.", flush=True)
        return

    for group_name in ANOMALY_TYPE_METHODS:
        raw_outputs = results["raw"].get(group_name)
        log_outputs = results["log1p"].get(group_name)
        if not raw_outputs or not log_outputs:
            continue

        raw_top = raw_outputs["consensus_topk"].copy()
        log_top = log_outputs["consensus_topk"].copy()
        raw_set = set(raw_top["timestamp"])
        log_set = set(log_top["timestamp"])
        intersection = raw_set & log_set
        union = raw_set | log_set

        rows.append(
            {
                "group": group_name,
                "top_k": top_k,
                "raw_count": len(raw_set),
                "log1p_count": len(log_set),
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else 0.0,
            }
        )

        shared = raw_top[raw_top["timestamp"].isin(intersection)].merge(
            log_top,
            on="timestamp",
            suffixes=("_raw", "_log1p"),
        )
        if not shared.empty:
            shared = shared.sort_values(
                ["consensus_rank_raw", "consensus_rank_log1p", "timestamp"]
            )
        shared.to_csv(comparison_dir / f"shared_raw_log1p_{group_name}.csv", index=False)

    overlap = pd.DataFrame(rows)
    overlap.to_csv(comparison_dir / "raw_vs_log1p_consensus_overlap.csv", index=False)



def compare_customer_univariate_raw_vs_log1p(
        output_dir: Path,
        top_k: int,
) -> None:

    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw" / "customer_univariate" / "customer_univariate_consensus_topk.csv"
    log_path = output_dir / "log1p" / "customer_univariate" / "customer_univariate_consensus_topk.csv"

    if not raw_path.exists() or not log_path.exists():
        print("Customer raw/log1p Vergleich wird übersprungen.", flush=True)
        return
    raw = pd.read_csv(raw_path)
    log = pd.read_csv(log_path)

    if raw.empty or log.empty:
        print("Customer raw/log1p Vergleich wird übersprungen.", flush=True)
        return

    rows = []
    shared_frames = []
    group_cols = ["customer", "anomaly_type"]

    for keys, raw_group in raw.groupby(group_cols):
        customer, anomaly_type = keys
        log_group = log[
            (log["customer"].astype(str) == str(customer))
            & (log["anomaly_type"].astype(str) == str(anomaly_type))
            ]

        raw_top = raw_group.head(top_k).copy()
        log_top = log_group.head(top_k).copy()

        raw_set = set(raw_top["timestamp"])
        log_set = set(log_top["timestamp"])
        intersection = raw_set & log_set
        union = raw_set | log_set

        rows.append(
            {
                "customer": customer,
                "anomaly_type": anomaly_type,
                "top_k": top_k,
                "raw_count": len(raw_set),
                "log1p_count": len(log_set),
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else 0.0,
            }
        )

        if intersection:
            shared = raw_top[raw_top["timestamp"].isin(intersection)].merge(
                log_top,
                on=["customer", "anomaly_type", "timestamp"],
                suffixes=("_raw", "_log1p"),
            )
            if not shared.empty:
                shared = shared.sort_values(
                    [
                        "customer",
                        "anomaly_type",
                        "consensus_rank_raw",
                        "consensus_rank_log1p",
                        "timestamp",
                    ],
                )
                shared_frames.append(shared)

    pd.DataFrame(rows).to_csv(
        comparison_dir / "customer_univariate_raw_vs_log1p_overlap.csv",
        index=False,
    )

    if shared_frames:
        pd.concat(shared_frames, ignore_index=True).to_csv(
            comparison_dir / "shared_customer_univariate_raw_log1p.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(
            comparison_dir / "shared_customer_univariate_raw_log1p.csv",
            index=False,
        )


def run_candidate_study(
    raw_csv: Path,
    log1p_csv: Path,
    output_dir: Path,
    top_k: int = TOP_K,
) -> Path:
    raw_csv = Path(raw_csv)
    log1p_csv = Path(log1p_csv)
    output_dir = Path(output_dir)

    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)
    if not log1p_csv.exists():
        raise FileNotFoundError(log1p_csv)

    method_groups = {group: list(methods) for group, methods in ANOMALY_TYPE_METHODS.items()}

    print(f"Step A startet -> {output_dir}", flush=True)
    results: Dict[str, Dict[str, Dict[str, "pd.DataFrame"]]] = {}
    results["raw"] = run_variant(
        variant_name="raw",
        csv_path=raw_csv,
        output_dir=output_dir,
        method_groups=method_groups,
        top_k=top_k,
        catch_params=CATCH_PARAMS,
    )
    print("raw fertig", flush=True)
    results["log1p"] = run_variant(
        variant_name="log1p",
        csv_path=log1p_csv,
        output_dir=output_dir,
        method_groups=method_groups,
        top_k=top_k,
        catch_params=CATCH_PARAMS,
    )
    print("log1p fertig", flush=True)
    compare_raw_vs_log1p(results, output_dir, top_k)
    compare_customer_univariate_raw_vs_log1p(output_dir, top_k)
    print(f"Step A fertig -> {output_dir}", flush=True)
    return output_dir


def main() -> None:
    run_candidate_study(
        Path("/Users/timadmin/Dokumente/TAB-MAC/dataset/anomaly_detect/data/atruvia_data/Systemanmeldung/Systemanmeldung_day_raw_stacked.csv"),
        Path("/Users/timadmin/Dokumente/TAB-MAC/dataset/anomaly_detect/data/atruvia_data/Systemanmeldung/Systemanmeldung_day_log1p_stacked.csv"),
        Path("/Users/timadmin/Dokumente/TAB-MAC/result/anomaly_candidate_study/systemanmeldung_cluster04"),
        TOP_K,
    )


if __name__ == "__main__":
    main()
