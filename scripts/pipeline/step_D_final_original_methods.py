from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


from ts_benchmark.baselines.self_impl.CATCH.CATCH import CATCH
from ts_benchmark.models import get_models

RAW_STACKED = Path("/Users/timadmin/Dokumente/TAB-MAC/dataset/anomaly_detect/data/atruvia_data/Systemanmeldung/Systemanmeldung_day_raw_stacked.csv")
LOG1P_STACKED = Path("/Users/timadmin/Dokumente/TAB-MAC/dataset/anomaly_detect/data/atruvia_data/Systemanmeldung/Systemanmeldung_day_log1p_stacked.csv")
OUTPUT_DIR = Path("/Users/timadmin/Dokumente/TAB-MAC/results/systemanmeldung_cluster04_20260618_211652/04_final_original_methods")
PLOT_DIR = Path("/Users/timadmin/Dokumente/TAB-MAC/results/plots/systemanmeldung_final_original_methods")
TOP_K_BY_ANOMALY_TYPE = {
    "global_point": 25,
    "contextual_point": 50,
    "subsequence_pattern": 50,
    "multivariate_dependency": 25,
}

EXPECTED_CUSTOMERS = [
    "161838000",
    "161836000",
    "161835000",
    "3296000",
    "13401000",
    "161832000",
    "2968000",
    "330387000",
    "13400000",
    "620000",
    "633000",
]

ANOMALY_TYPES = [
    "global_point",
    "contextual_point",
    "subsequence_pattern",
    "multivariate_dependency",
]

MODEL_CONFIGS = {
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

METHOD_COLORS = {
    "units": "#d62728",
    "calf": "#ff7f0e",
    "catch": "#1f77b4",
}

METHOD_NAMES = {
    "units": "UniTS",
    "calf": "CALF",
    "catch": "CATCH",
}


def load_stacked(path):
    data = pd.read_csv(path)
    data["_date"] = pd.to_datetime(data["date"], errors="coerce")
    wide = data.pivot(index="_date", columns="cols", values="data")
    wide = wide.sort_index()
    wide.index.name = "date"
    wide.columns = wide.columns.astype(str)

    if len(wide) != 609:
        print(f"Warnung: {path} hat {len(wide)} Tage.", flush=True)
    if set(wide.columns) != set(EXPECTED_CUSTOMERS):
        print(f"Warnung: {path} enthaelt nicht exakt die erwarteten Kunden.", flush=True)

    return wide


def clean_features(data):
    features = data.copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.interpolate(limit_direction="both").ffill().bfill()
    keep = features.nunique(dropna=True)
    return features[list(keep[keep > 1].index)]


def align_scores(scores, length):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) == length:
        return scores
    if len(scores) > length:
        return scores[:length]
    return np.pad(scores, (0, length - len(scores)), mode="constant", constant_values=np.nan)


def to_score_array(score_output):
    if isinstance(score_output, dict):
        score_output = next(iter(score_output.values()))
    if isinstance(score_output, (tuple, list)):
        score_output = score_output[0]
    return np.asarray(score_output, dtype=float).reshape(-1)


def normalize_scores(scores):
    normalized = pd.DataFrame(index=scores.index)
    for column in scores.columns:
        series = pd.to_numeric(scores[column], errors="coerce")
        finite = np.isfinite(series.to_numpy(dtype=float))
        percentile = pd.Series(np.nan, index=scores.index, dtype=float)
        if finite.any():
            percentile.loc[series.index[finite]] = series.loc[series.index[finite]].rank(
                method="average",
                pct=True,
                ascending=True,
            )
        normalized[column] = percentile
    return normalized


def run_tab_model(features, method):
    model_config = dict(MODEL_CONFIGS[method])
    model_config["model_hyper_params"] = dict(model_config["model_hyper_params"])
    model = get_models({"models": [model_config]})[0]()

    if hasattr(model, "detect_fit"):
        model.detect_fit(features, features)
    else:
        model.fit(features, None)

    scores = to_score_array(model.detect_score(features))
    return align_scores(scores, len(features))


def run_catch(features):
    params = dict(CATCH_PARAMS)
    params["seq_len"] = min(int(params["seq_len"]), max(3, int(len(features) * 0.6)))
    params["batch_size"] = min(int(params["batch_size"]), max(1, len(features)))

    model = CATCH(**params)
    model.device = torch.device("cpu")
    model.detect_fit(features, None)
    scores, _ = model.detect_score(features)
    return align_scores(scores, len(features))


def run_univariate(features, method):
    result = {}
    for customer in features.columns:
        print(f"{method} {customer}", flush=True)
        try:
            result[customer] = run_tab_model(features[[customer]], method)
        except Exception as exc:
            print(f"{method} {customer} fehlgeschlagen", flush=True)
            print(exc, flush=True)
    return pd.DataFrame(result, index=features.index)


def build_topk(scores, variant, anomaly_type, method):
    top_k = TOP_K_BY_ANOMALY_TYPE[anomaly_type]
    normalized = normalize_scores(scores)
    rows = []
    for customer in normalized.columns:
        table = pd.DataFrame(
            {
                "variant": variant,
                "anomaly_type": anomaly_type,
                "method": method,
                "customer": customer,
                "date": normalized.index,
                "score": scores[customer],
                "score_pct": normalized[customer],
            }
        ).reset_index(drop=True)
        table = table.dropna(subset=["score_pct"])
        table = table.sort_values(["score_pct", "score", "date"], ascending=[False, False, True])
        table = table.head(top_k).copy()
        table["rank"] = range(1, len(table) + 1)
        rows.append(table)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["variant", "anomaly_type", "method", "customer", "rank", "date", "score", "score_pct"])


def compare_variants(raw_candidates, log1p_candidates):
    rows = []
    shared_rows = []
    for anomaly_type in ANOMALY_TYPES:
        raw_part = raw_candidates[raw_candidates["anomaly_type"] == anomaly_type]
        log_part = log1p_candidates[log1p_candidates["anomaly_type"] == anomaly_type]
        raw_set = set(zip(raw_part["customer"].astype(str), raw_part["date"].astype(str)))
        log_set = set(zip(log_part["customer"].astype(str), log_part["date"].astype(str)))
        intersection = raw_set & log_set
        union = raw_set | log_set
        rows.append(
            {
                "anomaly_type": anomaly_type,
                "top_k": TOP_K_BY_ANOMALY_TYPE[anomaly_type],
                "raw_count": len(raw_set),
                "log1p_count": len(log_set),
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else 0,
            }
        )
        for customer, date in sorted(intersection):
            shared_rows.append({"anomaly_type": anomaly_type, "customer": customer, "date": date})
    return pd.DataFrame(rows), pd.DataFrame(shared_rows)


def run_variant(variant, data):
    print(f"Starte {variant}", flush=True)
    features = clean_features(data)
    candidates = []

    units_scores = run_univariate(features, "units")
    calf_scores = run_univariate(features, "calf")

    candidates.append(build_topk(units_scores, variant, "global_point", "units"))
    candidates.append(build_topk(calf_scores, variant, "contextual_point", "calf"))
    candidates.append(build_topk(calf_scores, variant, "subsequence_pattern", "calf"))

    try:
        print("catch", flush=True)
        catch_scores = run_catch(features)
        catch_scores = pd.DataFrame({"ALL": catch_scores}, index=features.index)
        candidates.append(build_topk(catch_scores, variant, "multivariate_dependency", "catch"))
    except Exception as exc:
        print("catch fehlgeschlagen", flush=True)
        print(exc, flush=True)

    candidates = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()

    variant_dir = OUTPUT_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(variant_dir / "final_candidates.csv", index=False)
    multivariate = candidates[candidates["customer"].astype(str) == "ALL"].copy()
    if not multivariate.empty:
        multivariate.to_csv(variant_dir / "final_multivariate_candidates.csv", index=False)

    return candidates


def plot_univariate(data, candidates, variant):
    for anomaly_type in ["global_point", "contextual_point", "subsequence_pattern"]:
        output_dir = PLOT_DIR / variant / anomaly_type
        output_dir.mkdir(parents=True, exist_ok=True)
        part = candidates[candidates["anomaly_type"] == anomaly_type].copy()
        title = METHOD_NAMES[part["method"].iloc[0]] if not part.empty else anomaly_type
        for customer in data.columns:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(data.index, data[customer], color="steelblue", width=1.0)
            customer_part = part[part["customer"].astype(str) == str(customer)].copy()
            for method, method_part in customer_part.groupby("method"):
                dates = pd.to_datetime(method_part["date"])
                values = data.reindex(dates)[customer]
                ax.scatter(dates, values, color=METHOD_COLORS.get(method, "red"), edgecolors="black", linewidths=0.4, s=45, label=method, zorder=3)
            ax.set_title(title)
            ax.set_xlabel("Datum")
            ax.set_ylabel("Anzahl")
            if not customer_part.empty:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
            fig.tight_layout()
            fig.savefig(output_dir / f"{customer}.png", dpi=150)
            plt.close(fig)


def plot_multivariate(data, candidates, variant):
    part = candidates[candidates["anomaly_type"] == "multivariate_dependency"].copy()
    if part.empty:
        return
    dates = pd.to_datetime(part["date"])
    fig, axes = plt.subplots(len(data.columns), 1, figsize=(16, 18), sharex=True)
    for ax, customer in zip(axes, data.columns):
        ax.bar(data.index, data[customer], color="#b7c2cc", width=1.0)
        values = data.reindex(dates)[customer]
        ax.scatter(dates, values, color=METHOD_COLORS["catch"], edgecolors="black", linewidths=0.4, s=38, label="catch", zorder=3)
        ax.set_ylabel(customer, rotation=0, ha="right", va="center", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Datum")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1)
    fig.suptitle("CATCH")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = PLOT_DIR / variant
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "multivariate_dependency.png", dpi=150)
    plt.close(fig)


def main():
    print("Step D startet", flush=True)
    raw = load_stacked(RAW_STACKED)
    log1p = load_stacked(LOG1P_STACKED)

    raw_candidates = run_variant("raw", raw)
    log_candidates = run_variant("log1p", log1p)

    comparisons = OUTPUT_DIR / "comparisons"
    comparisons.mkdir(parents=True, exist_ok=True)
    overlap, shared = compare_variants(raw_candidates, log_candidates)
    overlap.to_csv(comparisons / "raw_vs_log1p_final_overlap.csv", index=False)
    shared.to_csv(comparisons / "raw_log1p_shared_final_candidates.csv", index=False)

    plot_univariate(raw, raw_candidates, "raw")
    plot_univariate(log1p, log_candidates, "log1p")
    plot_multivariate(raw, raw_candidates, "raw")
    plot_multivariate(log1p, log_candidates, "log1p")

    print(f"CSV gespeichert in: {OUTPUT_DIR}", flush=True)
    print(f"Plots gespeichert in: {PLOT_DIR}", flush=True)
    print("Step D fertig", flush=True)


if __name__ == "__main__":
    main()
