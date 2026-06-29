import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from step_A_candidate_study import run_candidate_study


LABEL_TO_RESULT_GROUPS = {
    "global_peak": ["global_point"],
    "contextual_peak": ["contextual_point"],
    "drop_to_zero": ["contextual_point", "subsequence_pattern"],
    "activity_block": ["subsequence_pattern"],
    "multivariate_group_peak": ["multivariate_dependency"],
}

ALL_RESULT_GROUPS = [
    "contextual_point",
    "global_point",
    "multivariate_dependency",
    "subsequence_pattern",
]

CUSTOMER_UNIVARIATE_GROUPS = {"global_point", "contextual_point", "subsequence_pattern"}
SEGMENT_ANOMALIES = {"drop_to_zero", "activity_block"}
EVALUATION_VARIANTS = ("raw", "log1p", "combined")


def normalize_customer_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def add_candidate_columns(
    frame: pd.DataFrame,
    *,
    variant: str,
    result_group: str,
    method: str,
    rank_col: str,
    score_col: Optional[str],
) -> pd.DataFrame:
    columns = ["variant", "result_group", "method", "timestamp", "rank", "score", "customer"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    out = frame.copy()
    out["variant"] = variant
    out["result_group"] = result_group
    out["method"] = method if method != "__from_column__" else out["method"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.normalize()
    out["rank"] = pd.to_numeric(out[rank_col], errors="coerce")
    if score_col and score_col in out.columns:
        out["score"] = pd.to_numeric(out[score_col], errors="coerce")
    else:
        out["score"] = pd.NA
    if "customer" not in out.columns:
        out["customer"] = ""
    out["customer"] = out["customer"].apply(normalize_customer_id)
    out = out.dropna(subset=["timestamp", "rank"])
    return out[columns]


def load_candidates_for_group(run_output: Path, variant: str, result_group: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    if result_group in CUSTOMER_UNIVARIATE_GROUPS:
        base = run_output / variant / "customer_univariate"
        consensus_path = base / "customer_univariate_consensus_topk.csv"
        method_path = base / "customer_univariate_topk_by_method.csv"

        if consensus_path.exists():
            consensus = pd.read_csv(consensus_path)
            if "anomaly_type" in consensus.columns:
                consensus = consensus[consensus["anomaly_type"].astype(str) == result_group]
                frames.append(
                    add_candidate_columns(
                        consensus,
                        variant=variant,
                        result_group=result_group,
                        method="consensus",
                        rank_col="consensus_rank",
                        score_col="consensus_score",
                    )
                )

        if method_path.exists():
            by_method = pd.read_csv(method_path)
            if "anomaly_type" in by_method.columns:
                by_method = by_method[by_method["anomaly_type"].astype(str) == result_group]
                frames.append(
                    add_candidate_columns(
                        by_method,
                        variant=variant,
                        result_group=result_group,
                        method="__from_column__",
                        rank_col="method_rank",
                        score_col="score",
                    )
                )
    else:
        base = run_output / variant / result_group
        consensus_path = base / "consensus_topk.csv"
        method_path = base / "topk_by_method.csv"

        if consensus_path.exists():
            frames.append(
                add_candidate_columns(
                    pd.read_csv(consensus_path),
                    variant=variant,
                    result_group=result_group,
                    method="consensus",
                    rank_col="consensus_rank",
                    score_col="consensus_score",
                )
            )

        if method_path.exists():
            frames.append(
                add_candidate_columns(
                    pd.read_csv(method_path),
                    variant=variant,
                    result_group=result_group,
                    method="__from_column__",
                    rank_col="method_rank",
                    score_col="score",
                )
            )

    if not frames:
        return pd.DataFrame()
    candidates = pd.concat(frames, ignore_index=True)
    candidates["method"] = candidates["method"].astype(str)
    candidates["result_group"] = result_group
    return candidates


def combine_raw_log1p_candidates(raw: pd.DataFrame, log1p: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (raw, log1p) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["variant"] = "combined"
    combined = combined.sort_values(["method", "customer", "timestamp", "rank"], kind="mergesort")
    combined = combined.drop_duplicates(
        ["method", "customer", "result_group", "timestamp"],
        keep="first",
    )
    return combined.reset_index(drop=True)


def candidate_count(candidates: pd.DataFrame, k: int) -> int:
    if candidates.empty:
        return 0
    frame = candidates[candidates["rank"] <= k]
    if frame.empty:
        return 0
    return int(frame[["timestamp", "customer", "method", "result_group"]].drop_duplicates().shape[0])


def filter_candidates_for_label(
    candidates: pd.DataFrame,
    label: pd.Series,
    result_group: str,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    filtered = candidates.copy()
    if result_group in CUSTOMER_UNIVARIATE_GROUPS:
        label_customer = normalize_customer_id(label.get("customer", ""))
        filtered = filtered[filtered["customer"].astype(str) == label_customer]
    return filtered


def matched_candidates(
    candidates: pd.DataFrame,
    label: pd.Series,
    result_group: str,
    k: int,
    segment_tolerance_days: int,
) -> pd.DataFrame:
    filtered = filter_candidates_for_label(candidates, label, result_group)
    if filtered.empty:
        return filtered
    filtered = filtered[filtered["rank"] <= k].copy()
    if filtered.empty:
        return filtered

    anomaly_type = str(label["anomaly_type"])
    if anomaly_type in SEGMENT_ANOMALIES:
        start = pd.Timestamp(label["start_date"]).normalize() - pd.Timedelta(days=segment_tolerance_days)
        end = pd.Timestamp(label["end_date"]).normalize() + pd.Timedelta(days=segment_tolerance_days)
        mask = (filtered["timestamp"] >= start) & (filtered["timestamp"] <= end)
    else:
        date_value = label["date"] if "date" in label else label["start_date"]
        target = pd.Timestamp(date_value).normalize()
        mask = filtered["timestamp"] == target
    return filtered[mask].sort_values(["rank", "timestamp"], kind="mergesort")


def match_one_label(
    label: pd.Series,
    candidates: pd.DataFrame,
    result_group: str,
    method: str,
    k: int,
    segment_tolerance_days: int,
) -> Dict[str, Any]:
    method_candidates = candidates[candidates["method"].astype(str) == str(method)].copy()
    method_candidates = filter_candidates_for_label(method_candidates, label, result_group)
    pool_size = candidate_count(method_candidates, k)
    matches = matched_candidates(method_candidates, label, result_group, k, segment_tolerance_days)
    if matches.empty:
        return {
            "matched": False,
            "matched_timestamp": "",
            "matched_rank": "",
            "candidate_count": pool_size,
        }

    first = matches.iloc[0]
    return {
        "matched": True,
        "matched_timestamp": first["timestamp"].strftime("%Y-%m-%d"),
        "matched_rank": float(first["rank"]),
        "candidate_count": pool_size,
    }


def empty_detail_row(
    run_id: str,
    label: pd.Series,
    variant: str,
    result_group: str,
    method: str,
    k: int,
    candidate_count_value: int = 0,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "anomaly_id": label["anomaly_id"],
        "anomaly_type": label["anomaly_type"],
        "intensity": label["intensity"],
        "variant": variant,
        "result_group": result_group,
        "method": method,
        "k": k,
        "matched": False,
        "matched_timestamp": "",
        "matched_rank": "",
        "candidate_count": candidate_count_value,
    }


def evaluate_run(
    run_id: str,
    labels_csv: Path,
    run_output: Path,
    eval_k: Sequence[int],
    segment_tolerance_days: int,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)
    details: List[Dict[str, Any]] = []
    candidate_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    for variant in ("raw", "log1p"):
        for result_group in ALL_RESULT_GROUPS:
            candidate_cache[(variant, result_group)] = load_candidates_for_group(
                run_output,
                variant,
                result_group,
            )

    for result_group in ALL_RESULT_GROUPS:
        candidate_cache[("combined", result_group)] = combine_raw_log1p_candidates(
            candidate_cache.get(("raw", result_group), pd.DataFrame()),
            candidate_cache.get(("log1p", result_group), pd.DataFrame()),
        )

    for _, label in labels.iterrows():
        anomaly_type = str(label["anomaly_type"])
        result_groups = LABEL_TO_RESULT_GROUPS.get(anomaly_type, [])

        for variant in EVALUATION_VARIANTS:
            for result_group in result_groups:
                candidates = candidate_cache.get((variant, result_group), pd.DataFrame())
                if candidates.empty:
                    for k in eval_k:
                        details.append(empty_detail_row(run_id, label, variant, result_group, "consensus", k))
                    continue

                candidates_for_label = filter_candidates_for_label(candidates, label, result_group)
                methods = sorted(candidates_for_label["method"].astype(str).unique())
                if not methods:
                    for k in eval_k:
                        details.append(empty_detail_row(run_id, label, variant, result_group, "consensus", k))
                    continue

                for method in methods:
                    for k in eval_k:
                        match = match_one_label(
                            label,
                            candidates,
                            result_group,
                            method,
                            k,
                            segment_tolerance_days,
                        )
                        details.append(
                            {
                                "run_id": run_id,
                                "anomaly_id": label["anomaly_id"],
                                "anomaly_type": anomaly_type,
                                "intensity": label["intensity"],
                                "variant": variant,
                                "result_group": result_group,
                                "method": method,
                                "k": k,
                                "matched": bool(match["matched"]),
                                "matched_timestamp": match["matched_timestamp"],
                                "matched_rank": match["matched_rank"],
                                "candidate_count": match["candidate_count"],
                            }
                        )

    return pd.DataFrame(details)


def summarize_details(details: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    columns = list(group_cols) + [
        "hit_at_k",
        "recall_at_k",
        "precision_at_k",
        "mean_rank",
        "median_rank",
        "detected_count",
        "label_count",
        "candidate_count",
    ]
    if details.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []
    for keys, group in details.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label_key_cols = ["run_id", "anomaly_id"] if "run_id" in group.columns else ["anomaly_id"]
        label_count = int(group[label_key_cols].drop_duplicates().shape[0])
        detected_count = int(
            group.loc[group["matched"].astype(bool), label_key_cols]
            .drop_duplicates()
            .shape[0]
        )
        candidate_count_value = int(pd.to_numeric(group["candidate_count"], errors="coerce").fillna(0).sum())
        rank_frame = group.loc[group["matched"].astype(bool), ["anomaly_id", "matched_rank"]]
        ranks = pd.to_numeric(rank_frame.drop_duplicates()["matched_rank"], errors="coerce").dropna()
        row = {}
        for col, value in zip(group_cols, keys):
            row[col] = value
        row.update(
            {
                "hit_at_k": detected_count,
                "recall_at_k": detected_count / label_count if label_count else math.nan,
                "precision_at_k": (
                    detected_count / candidate_count_value if candidate_count_value else math.nan
                ),
                "mean_rank": float(ranks.mean()) if not ranks.empty else math.nan,
                "median_rank": float(ranks.median()) if not ranks.empty else math.nan,
                "detected_count": detected_count,
                "label_count": label_count,
                "candidate_count": candidate_count_value,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def aggregate_over_runs(summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["anomaly_type", "intensity", "variant", "method", "result_group", "k"]
    columns = group_cols + [
        "recall_mean",
        "recall_std",
        "precision_mean",
        "precision_std",
        "mean_rank",
        "median_rank",
        "detected_total",
        "label_total",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []
    for keys, group in summary.groupby(group_cols, dropna=False):
        recall = pd.to_numeric(group["recall_at_k"], errors="coerce").dropna()
        precision = pd.to_numeric(group["precision_at_k"], errors="coerce").dropna()
        mean_rank = pd.to_numeric(group["mean_rank"], errors="coerce").dropna()
        median_rank = pd.to_numeric(group["median_rank"], errors="coerce").dropna()
        row = {}
        for col, value in zip(group_cols, keys):
            row[col] = value
        row.update(
            {
                "recall_mean": float(recall.mean()) if not recall.empty else math.nan,
                "recall_std": float(recall.std(ddof=0)) if not recall.empty else math.nan,
                "precision_mean": float(precision.mean()) if not precision.empty else math.nan,
                "precision_std": float(precision.std(ddof=0)) if not precision.empty else math.nan,
                "mean_rank": float(mean_rank.mean()) if not mean_rank.empty else math.nan,
                "median_rank": float(median_rank.median()) if not median_rank.empty else math.nan,
                "detected_total": int(group["detected_count"].sum()),
                "label_total": int(group["label_count"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def write_outputs(output_dir: Path, detail_frames: Sequence[pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    summary = summarize_details(
        details,
        ["run_id", "anomaly_type", "intensity", "variant", "method", "result_group", "k"],
    )
    summary_by_run = summarize_details(details, ["run_id", "variant", "method", "k"])
    summary_by_type = summarize_details(details, ["anomaly_type", "intensity", "variant", "method", "k"])
    summary_by_method = summarize_details(details, ["method", "variant", "result_group", "k"])
    overall = aggregate_over_runs(summary)

    details.to_csv(output_dir / "synthetic_label_match_details.csv", index=False)
    summary.to_csv(output_dir / "synthetic_eval_summary.csv", index=False)
    summary_by_run.to_csv(output_dir / "synthetic_eval_by_run.csv", index=False)
    summary_by_type.to_csv(output_dir / "synthetic_eval_by_type.csv", index=False)
    summary_by_method.to_csv(output_dir / "synthetic_eval_by_method.csv", index=False)
    overall.to_csv(output_dir / "synthetic_eval_aggregate.csv", index=False)


def run_method_evaluation(
    synthetic_dir: Path,
    output_dir: Path,
    method_top_k: int = 50,
    eval_k: Sequence[int] = (10, 25, 50),
    segment_tolerance_days: int = 0,
) -> Path:
    synthetic_dir = Path(synthetic_dir)
    output_dir = Path(output_dir)
    eval_values = []
    for k in eval_k:
        value = int(k)
        if value not in eval_values:
            eval_values.append(value)
    eval_k = sorted(eval_values)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = []
    for path in synthetic_dir.glob("run_*"):
        if path.is_dir():
            run_dirs.append(path)
    run_dirs = sorted(run_dirs)
    if not run_dirs:
        raise ValueError(f"No synthetic run directories found in {synthetic_dir}.")

    detail_frames: List[pd.DataFrame] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        raw_csv = run_dir / "raw_stacked.csv"
        log1p_csv = run_dir / "log1p_stacked.csv"
        labels_csv = run_dir / "labels.csv"
        run_output = output_dir / run_id

        print(f"Step C startet: {run_id}", flush=True)
        run_output.mkdir(parents=True, exist_ok=True)
        run_candidate_study(
            raw_csv=raw_csv,
            log1p_csv=log1p_csv,
            output_dir=run_output,
            top_k=method_top_k,
        )
        details = evaluate_run(run_id, labels_csv, run_output, eval_k, segment_tolerance_days)
        detail_frames.append(details)
        print(f"Step C fertig: {run_id}", flush=True)

    write_outputs(output_dir, detail_frames)

    print(f"Step C Ergebnis: {output_dir}", flush=True)
    print(f"  {output_dir / 'synthetic_label_match_details.csv'}", flush=True)
    print(f"  {output_dir / 'synthetic_eval_summary.csv'}", flush=True)
    return output_dir


def main() -> None:
    run_method_evaluation(
        synthetic_dir=Path("/Users/timadmin/Dokumente/TAB-MAC/results/systemanmeldung_cluster04_20260618_211652/02_synthetic_data"),
        output_dir=Path("/Users/timadmin/Dokumente/TAB-MAC/results/systemanmeldung_cluster04_20260618_211652/03_method_runs"),
        method_top_k=50,
        eval_k=(10, 25, 50),
        segment_tolerance_days=0,
    )


if __name__ == "__main__":
    main()
