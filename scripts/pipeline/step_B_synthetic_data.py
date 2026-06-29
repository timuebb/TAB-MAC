import json
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


RAW_STACKED = Path(
    "/Users/timadmin/Dokumente/TAB-MAC/dataset/anomaly_detect/data/atruvia_data/Systemanmeldung/"
    "Systemanmeldung_day_raw_stacked.csv"
)
SEED = 42
NUM_REPEATS = 5
EDGE_MARGIN_DAYS = 50
CANDIDATE_BUFFER_DAYS = 5
PHASE_BUFFER_DAYS = 7
TOP_K_CANDIDATES = 10
MIN_INJECTION_GAP_DAYS = 7
INJECTIONS_PER_TYPE = 1

DEFAULT_ANOMALY_TYPES = (
    "global_peak",
    "contextual_peak",
    "drop_to_zero",
    "activity_block",
    "multivariate_group_peak",
)
DEFAULT_INTENSITIES = ("medium", "strong")
SEGMENT_ANOMALIES = {"drop_to_zero", "activity_block"}

LOCAL_WINDOW_DAYS = 14
GLOBAL_ALPHA_MEDIUM = 6.0
GLOBAL_ALPHA_STRONG = 8.0
CONTEXTUAL_BETA_MEDIUM = 4.0
CONTEXTUAL_BETA_STRONG = 6.0
DROP_DURATION_MEDIUM = 3
DROP_DURATION_STRONG = 5
ACTIVITY_DURATION_MEDIUM = 5
ACTIVITY_DURATION_STRONG = 7
ACTIVITY_FACTOR_MEDIUM = 1.5
ACTIVITY_FACTOR_STRONG = 2.0
MULTIVARIATE_GROUP_MIN = 3
MULTIVARIATE_GROUP_MAX = 5


def format_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def to_json(value: Any) -> str:
    def convert(obj: Any) -> Any:
        if isinstance(obj, (pd.Timestamp, datetime)):
            return format_date(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=convert)


def clean_customer_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.isna().any().any():
        numeric = numeric.fillna(0.0)
    numeric.columns = [str(col) for col in numeric.columns]
    return numeric.sort_index().sort_index(axis=1)


def load_raw_stacked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)
    if raw.duplicated(["date", "cols"]).any():
        duplicates = raw.loc[raw.duplicated(["date", "cols"], keep=False), ["date", "cols"]]
        raise ValueError(
            "Stacked input contains duplicate date/customer rows: "
            f"{duplicates.head(10).to_dict(orient='records')}"
        )

    raw["_date"] = pd.to_datetime(raw["date"], errors="coerce")
    if raw["_date"].isna().any():
        raise ValueError(f"Could not parse all dates in {path}.")

    values = raw.pivot(index="_date", columns="cols", values="data")
    values.index.name = "date"
    return clean_customer_columns(values)


def robust_mad(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return 0.0
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def customer_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for customer in raw.columns:
        series = raw[customer].astype(float)
        active = series[series > 0]
        rows.append(
            {
                "customer": str(customer),
                "sum_events": float(series.sum()),
                "active_days": int((series > 0).sum()),
                "zero_days": int((series == 0).sum()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "active_median": float(active.median()) if len(active) else 0.0,
                "max": float(series.max()),
                "p75": float(series.quantile(0.75)),
                "p90": float(series.quantile(0.90)),
                "p95": float(series.quantile(0.95)),
                "p99": float(series.quantile(0.99)),
                "MAD": robust_mad(series),
            }
        )

    stats = pd.DataFrame(rows).sort_values(["sum_events", "customer"], ascending=[False, True])
    splits = np.array_split(stats.index.to_numpy(), 3)
    strata = {}
    for label, index_values in zip(("high_activity", "medium_activity", "low_activity"), splits):
        for idx in index_values:
            strata[idx] = label
    stats["activity_stratum"] = stats.index.map(strata)
    return stats.sort_values("customer").reset_index(drop=True)


def top_k_candidate_sources(candidate_dir: Path) -> List[Dict[str, Any]]:
    sources = [
        {
            "path": candidate_dir / "raw" / "all_methods_consensus_topk.csv",
            "variant": "raw",
            "default_anomaly_type": "",
        },
        {
            "path": candidate_dir / "log1p" / "all_methods_consensus_topk.csv",
            "variant": "log1p",
            "default_anomaly_type": "",
        },
        {
            "path": candidate_dir / "raw" / "multivariate_dependency" / "consensus_topk.csv",
            "variant": "raw",
            "default_anomaly_type": "multivariate_dependency",
        },
        {
            "path": candidate_dir / "log1p" / "multivariate_dependency" / "consensus_topk.csv",
            "variant": "log1p",
            "default_anomaly_type": "multivariate_dependency",
        },
    ]
    return [source for source in sources if source["path"].exists()]


def shared_candidate_sources(candidate_dir: Path) -> List[Dict[str, Any]]:
    sources = [
        {
            "path": candidate_dir / "comparisons" / "shared_customer_univariate_raw_log1p.csv",
            "variant": "raw_log1p",
            "default_anomaly_type": "",
        }
    ]
    for path in sorted((candidate_dir / "comparisons").glob("shared_raw_log1p_*.csv")):
        sources.append(
            {
                "path": path,
                "variant": "raw_log1p",
                "default_anomaly_type": path.stem.replace("shared_raw_log1p_", ""),
            }
        )
    return [source for source in sources if source["path"].exists()]


def first_value(row: pd.Series, names: Sequence[str]) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return ""


def extract_candidate_date(row: pd.Series, time_index: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    value = first_value(row, ("date", "timestamp", "selected_date", "timestamp_raw", "timestamp_log1p"))
    if value != "":
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            return pd.Timestamp(parsed).normalize()

    row_number = first_value(row, ("row", "row_raw", "row_log1p"))
    if row_number != "":
        try:
            idx = int(row_number)
        except Exception as exc:
            print(exc)
            return None
        if 0 <= idx < len(time_index):
            return pd.Timestamp(time_index[idx]).normalize()
    return None


def candidate_source_summary(candidate_dir: Path, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    rank_columns = ("consensus_rank", "consensus_rank_raw", "consensus_rank_log1p")

    source_groups = [
        (top_k_candidate_sources(candidate_dir), True),
        (shared_candidate_sources(candidate_dir), False),
    ]

    for sources, limit_top_k in source_groups:
        for source in sources:
            path = source["path"]
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                print(exc)
                continue
            if frame.empty:
                continue

            rank_col = next((col for col in rank_columns if col in frame.columns), None)
            if rank_col is not None:
                frame = frame.sort_values(rank_col, kind="mergesort")
            if limit_top_k:
                frame = frame.head(TOP_K_CANDIDATES)

            for _, row in frame.iterrows():
                candidate_date = extract_candidate_date(row, time_index)
                anomaly_type = (
                    str(row["anomaly_type"])
                    if "anomaly_type" in row and not pd.isna(row["anomaly_type"])
                    else str(source.get("default_anomaly_type", ""))
                )
                customer = ""
                if "customer" in row and not pd.isna(row["customer"]):
                    customer = str(row["customer"])
                elif "cols" in row and not pd.isna(row["cols"]):
                    customer = str(row["cols"])

                records.append(
                    {
                        "date": format_date(candidate_date) if candidate_date is not None else "",
                        "variant": str(row["variant"]) if "variant" in row and not pd.isna(row["variant"]) else source["variant"],
                        "anomaly_type": anomaly_type,
                        "customer": customer,
                        "consensus_rank": first_value(row, ("consensus_rank", "consensus_rank_raw", "consensus_rank_log1p")),
                        "consensus_score": first_value(row, ("consensus_score", "consensus_score_raw", "consensus_score_log1p")),
                        "available_method_count": first_value(row, ("available_method_count", "available_method_count_raw", "available_method_count_log1p")),
                        "missing_method_count": first_value(row, ("missing_method_count", "missing_method_count_raw", "missing_method_count_log1p")),
                    }
                )

    columns = [
        "date",
        "variant",
        "anomaly_type",
        "customer",
        "consensus_rank",
        "consensus_score",
        "available_method_count",
        "missing_method_count",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(records, columns=columns)
        .drop_duplicates()
        .sort_values(["date", "variant", "anomaly_type", "customer"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_exclusion_calendar(dates: pd.DatetimeIndex, candidates: pd.DataFrame) -> pd.DataFrame:
    normalized_dates = pd.DatetimeIndex(pd.to_datetime(dates).normalize())
    work = candidates.copy()
    work["_candidate_date"] = pd.to_datetime(work.get("date", pd.Series(dtype=str)), errors="coerce").dt.normalize()
    work = work.dropna(subset=["_candidate_date"]).copy()

    records: List[Dict[str, Any]] = []
    for idx, current_date in enumerate(normalized_dates):
        is_edge = idx < EDGE_MARGIN_DAYS or idx >= len(normalized_dates) - EDGE_MARGIN_DAYS
        is_candidate = False
        is_phase = False
        nearest_candidate = ""
        nearest_distance = ""

        if not work.empty:
            distances_days = (work["_candidate_date"] - current_date).abs().dt.days
            nearest_idx = distances_days.idxmin()
            nearest = pd.Timestamp(work.loc[nearest_idx, "_candidate_date"])
            nearest_candidate = format_date(nearest)
            nearest_distance = int(distances_days.loc[nearest_idx])
            candidate_hits = distances_days <= CANDIDATE_BUFFER_DAYS
            phase_candidates = work["anomaly_type"].isin(["subsequence_pattern", "multivariate_dependency"])
            phase_hits = phase_candidates & (distances_days <= PHASE_BUFFER_DAYS)
            is_candidate = bool(candidate_hits.any())
            is_phase = bool(phase_hits.any())

        reasons = []
        if is_edge:
            reasons.append("edge_margin")
        if is_candidate:
            reasons.append("candidate_buffer")
        if is_phase:
            reasons.append("phase_buffer")

        records.append(
            {
                "date": format_date(current_date),
                "is_edge_excluded": bool(is_edge),
                "is_candidate_excluded": bool(is_candidate),
                "is_phase_excluded": bool(is_phase),
                "excluded_reason": ";".join(reasons),
                "nearest_candidate_date": nearest_candidate,
                "nearest_candidate_distance_days": nearest_distance,
            }
        )

    return pd.DataFrame(records)


def eligible_dates_from_calendar(calendar: pd.DataFrame) -> Set[pd.Timestamp]:
    eligible = calendar[
        ~calendar["is_edge_excluded"]
        & ~calendar["is_candidate_excluded"]
        & ~calendar["is_phase_excluded"]
    ]
    return set(pd.to_datetime(eligible["date"]).dt.normalize())


def stats_map(customer_stats: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    return {str(row["customer"]): row.to_dict() for _, row in customer_stats.iterrows()}


def window_dates(index: pd.DatetimeIndex, start_pos: int, duration: int) -> List[pd.Timestamp]:
    return [pd.Timestamp(index[pos]).normalize() for pos in range(start_pos, start_pos + duration)]


def interval_distance_days(candidate: Dict[str, Any], existing: Dict[str, Any]) -> int:
    candidate_start = pd.Timestamp(candidate["start_date"]).normalize()
    candidate_end = pd.Timestamp(candidate["end_date"]).normalize()
    existing_start = pd.Timestamp(existing["start_date"]).normalize()
    existing_end = pd.Timestamp(existing["end_date"]).normalize()
    if candidate_end < existing_start:
        return int((existing_start - candidate_end).days)
    if existing_end < candidate_start:
        return int((candidate_start - existing_end).days)
    return 0


def event_record(
    anomaly_type: str,
    intensity: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    customers: Sequence[str],
    selection_reason: str,
    pool_size: int,
) -> Dict[str, Any]:
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    return {
        "anomaly_type": anomaly_type,
        "intensity": intensity,
        "start_date": start_date,
        "end_date": end_date,
        "selected_date": start_date,
        "duration_days": int((end_date - start_date).days) + 1,
        "customers": tuple(str(customer) for customer in customers),
        "selection_reason": selection_reason,
        "selection_pool_size": int(pool_size),
    }


def build_global_peak_pool(
    raw: pd.DataFrame,
    customer_stats: Dict[str, Dict[str, Any]],
    eligible_dates: Set[pd.Timestamp],
    intensity: str,
) -> List[Dict[str, Any]]:
    pool = []
    for date, row in raw.iterrows():
        normalized = pd.Timestamp(date).normalize()
        if normalized not in eligible_dates:
            continue
        for customer in raw.columns:
            stats = customer_stats[str(customer)]
            value = float(row[customer])
            if value <= float(stats["p90"]):
                pool.append(
                    event_record(
                        "global_peak",
                        intensity,
                        normalized,
                        normalized,
                        [str(customer)],
                        "eligible_day_value_below_customer_p90",
                        0,
                    )
                )
    return pool


def build_contextual_peak_pool(
    raw: pd.DataFrame,
    customer_stats: Dict[str, Dict[str, Any]],
    eligible_dates: Set[pd.Timestamp],
    intensity: str,
) -> List[Dict[str, Any]]:
    pool = []
    for pos, date in enumerate(raw.index):
        normalized = pd.Timestamp(date).normalize()
        if normalized not in eligible_dates:
            continue
        left = max(0, pos - LOCAL_WINDOW_DAYS)
        right = min(len(raw), pos + LOCAL_WINDOW_DAYS + 1)
        for customer in raw.columns:
            stats = customer_stats[str(customer)]
            value = float(raw.iloc[pos][customer])
            window = raw.iloc[left:right][customer].astype(float)
            local_median = float(window.median())
            local_mad = robust_mad(window)
            if value <= float(stats["p90"]) and local_median <= float(stats["p75"]) and local_mad <= max(float(stats["MAD"]), 1.0):
                pool.append(
                    event_record(
                        "contextual_peak",
                        intensity,
                        normalized,
                        normalized,
                        [str(customer)],
                        "eligible_day_quiet_local_window_value_below_customer_p90",
                        0,
                    )
                )
    return pool


def build_drop_to_zero_pool(
    raw: pd.DataFrame,
    customer_stats: Dict[str, Dict[str, Any]],
    eligible_dates: Set[pd.Timestamp],
    intensity: str,
) -> List[Dict[str, Any]]:
    duration = DROP_DURATION_STRONG if intensity == "strong" else DROP_DURATION_MEDIUM
    pool = []
    for start_pos in range(0, len(raw) - duration + 1):
        dates = window_dates(raw.index, start_pos, duration)
        if not all(day in eligible_dates for day in dates):
            continue
        for customer in raw.columns:
            stats = customer_stats[str(customer)]
            window = raw.iloc[start_pos : start_pos + duration][customer].astype(float)
            if (window > 0).all() and float(window.mean()) >= float(stats["active_median"]):
                pool.append(
                    event_record(
                        "drop_to_zero",
                        intensity,
                        dates[0],
                        dates[-1],
                        [str(customer)],
                        "eligible_window_all_positive_mean_above_active_median",
                        0,
                    )
                )
    return pool


def build_activity_block_pool(
    raw: pd.DataFrame,
    customer_stats: Dict[str, Dict[str, Any]],
    eligible_dates: Set[pd.Timestamp],
    intensity: str,
) -> List[Dict[str, Any]]:
    duration = ACTIVITY_DURATION_STRONG if intensity == "strong" else ACTIVITY_DURATION_MEDIUM
    pool = []
    for start_pos in range(0, len(raw) - duration + 1):
        dates = window_dates(raw.index, start_pos, duration)
        if not all(day in eligible_dates for day in dates):
            continue
        for customer in raw.columns:
            stats = customer_stats[str(customer)]
            window = raw.iloc[start_pos : start_pos + duration][customer].astype(float)
            if (window == 0).sum() < duration / 2 and float(window.max()) <= float(stats["p99"]):
                pool.append(
                    event_record(
                        "activity_block",
                        intensity,
                        dates[0],
                        dates[-1],
                        [str(customer)],
                        "eligible_window_without_long_zero_phase_or_extreme_peak",
                        0,
                    )
                )
    return pool


def customer_groups(customer_stats: pd.DataFrame) -> List[Tuple[str, ...]]:
    customers_by_stratum: Dict[str, List[str]] = {}
    for _, row in customer_stats.sort_values(["activity_stratum", "customer"]).iterrows():
        customers_by_stratum.setdefault(str(row["activity_stratum"]), []).append(str(row["customer"]))

    all_customers = sorted(customer_stats["customer"].astype(str).tolist())
    effective_group_max = min(MULTIVARIATE_GROUP_MAX, len(all_customers) - 1)
    if effective_group_max < MULTIVARIATE_GROUP_MIN:
        print("multivariate_group_peak disabled because too few customers are available")
        return []
    group_size = min(effective_group_max, max(MULTIVARIATE_GROUP_MIN, len(all_customers)))

    groups: Set[Tuple[str, ...]] = set()
    strata = ["high_activity", "medium_activity", "low_activity"]
    base_lists = [customers_by_stratum.get(stratum, []) for stratum in strata]
    for high in base_lists[0]:
        for medium in base_lists[1]:
            for low in base_lists[2]:
                groups.add(tuple(sorted([high, medium, low])))

    if group_size > 3:
        for group in list(groups):
            remaining = [customer for customer in all_customers if customer not in group]
            groups.add(tuple(sorted(list(group) + remaining[: group_size - 3])))

    groups = {
        group
        for group in groups
        if MULTIVARIATE_GROUP_MIN <= len(group) <= effective_group_max and len(group) < len(all_customers)
    }

    if not groups:
        groups.update(tuple(combo) for combo in combinations(all_customers, group_size))
    return sorted(groups)


def build_multivariate_group_peak_pool(
    raw: pd.DataFrame,
    customer_stats_df: pd.DataFrame,
    customer_stats: Dict[str, Dict[str, Any]],
    eligible_dates: Set[pd.Timestamp],
    intensity: str,
) -> List[Dict[str, Any]]:
    groups = customer_groups(customer_stats_df)
    pool = []
    for date, row in raw.iterrows():
        normalized = pd.Timestamp(date).normalize()
        if normalized not in eligible_dates:
            continue
        for group in groups:
            if all(float(row[customer]) <= float(customer_stats[customer]["p90"]) for customer in group):
                pool.append(
                    event_record(
                        "multivariate_group_peak",
                        intensity,
                        normalized,
                        normalized,
                        list(group),
                        "eligible_day_subgroup_values_below_customer_p90_non_selected_customers_unchanged",
                        0,
                    )
                )
    return pool


def event_sort_key(event: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        event["anomaly_type"],
        event["intensity"],
        format_date(event["start_date"]),
        ",".join(event["customers"]),
    )


def build_pools(
    raw: pd.DataFrame,
    customer_stats_df: pd.DataFrame,
    exclusion_calendar: pd.DataFrame,
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    eligible_dates = eligible_dates_from_calendar(exclusion_calendar)
    stat_map = stats_map(customer_stats_df)
    pools: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for anomaly_type in DEFAULT_ANOMALY_TYPES:
        for intensity in DEFAULT_INTENSITIES:
            if anomaly_type == "global_peak":
                pool = build_global_peak_pool(raw, stat_map, eligible_dates, intensity)
            elif anomaly_type == "contextual_peak":
                pool = build_contextual_peak_pool(raw, stat_map, eligible_dates, intensity)
            elif anomaly_type == "drop_to_zero":
                pool = build_drop_to_zero_pool(raw, stat_map, eligible_dates, intensity)
            elif anomaly_type == "activity_block":
                pool = build_activity_block_pool(raw, stat_map, eligible_dates, intensity)
            else:
                pool = build_multivariate_group_peak_pool(raw, customer_stats_df, stat_map, eligible_dates, intensity)
            for event in pool:
                event["selection_pool_size"] = len(pool)
            pools[(anomaly_type, intensity)] = sorted(pool, key=event_sort_key)
    return pools


def select_injection_plan(
    pools: Dict[Tuple[str, str], List[Dict[str, Any]]],
    run_id: str,
    run_seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(run_seed)
    selected: List[Dict[str, Any]] = []
    occupied: Set[Tuple[pd.Timestamp, str]] = set()

    anomaly_counter = 1
    for anomaly_type in DEFAULT_ANOMALY_TYPES:
        for intensity in DEFAULT_INTENSITIES:
            pool = pools.get((anomaly_type, intensity), [])
            for draw_idx in range(INJECTIONS_PER_TYPE):
                non_overlapping = []
                for event in pool:
                    dates = pd.date_range(event["start_date"], event["end_date"], freq="D")
                    cells = {
                        (pd.Timestamp(day).normalize(), str(customer))
                        for day in dates
                        for customer in event["customers"]
                    }
                    if cells.isdisjoint(occupied):
                        non_overlapping.append((event, cells))

                valid = []
                for event, cells in non_overlapping:
                    distances = [interval_distance_days(event, existing) for existing in selected]
                    if not distances or min(distances) >= MIN_INJECTION_GAP_DAYS:
                        valid.append((event, cells))
                if not valid:
                    print(f"{run_id}: keine passende Position fuer {anomaly_type} {intensity}", flush=True)
                    continue

                chosen_event, chosen_cells = valid[int(rng.integers(0, len(valid)))]
                chosen = dict(chosen_event)
                chosen["run_id"] = run_id
                chosen["run_seed"] = run_seed
                chosen["anomaly_id"] = f"{run_id}_a{anomaly_counter:03d}"
                chosen["excluded_candidate_buffer_days"] = CANDIDATE_BUFFER_DAYS
                chosen["edge_margin_days"] = EDGE_MARGIN_DAYS
                chosen["min_injection_gap_days"] = MIN_INJECTION_GAP_DAYS
                anomaly_counter += 1
                selected.append(chosen)
                occupied.update(chosen_cells)

    return selected


def target_peak_value(current: float, stats: Dict[str, Any], alpha: float, fallback_quantile: str) -> float:
    mad = float(stats["MAD"])
    median = float(stats["median"])
    if mad > 0:
        target = median + alpha * mad
    else:
        target = max(current + 1.0, float(stats[fallback_quantile]), median + 1.0)
    return max(current, target)


def apply_event(
    data: pd.DataFrame,
    base_raw: pd.DataFrame,
    event: Dict[str, Any],
    customer_stats: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    anomaly_type = event["anomaly_type"]
    intensity = event["intensity"]
    customers = list(event["customers"])
    dates = pd.date_range(event["start_date"], event["end_date"], freq="D")

    original_values: Dict[str, Dict[str, float]] = {}
    injected_values: Dict[str, Dict[str, float]] = {}
    for customer in customers:
        original_values[customer] = {}
        injected_values[customer] = {}
        for day in dates:
            day = pd.Timestamp(day).normalize()
            current = float(data.loc[day, customer])
            original_values[customer][format_date(day)] = current
            stats = customer_stats[str(customer)]

            if anomaly_type == "global_peak":
                alpha = GLOBAL_ALPHA_STRONG if intensity == "strong" else GLOBAL_ALPHA_MEDIUM
                new_value = target_peak_value(current, stats, alpha, "p99")
            elif anomaly_type == "contextual_peak":
                pos = base_raw.index.get_loc(day)
                left = max(0, pos - LOCAL_WINDOW_DAYS)
                right = min(len(base_raw), pos + LOCAL_WINDOW_DAYS + 1)
                window = base_raw.iloc[left:right][customer].astype(float)
                local_median = float(window.median())
                local_mad = robust_mad(window)
                beta = CONTEXTUAL_BETA_STRONG if intensity == "strong" else CONTEXTUAL_BETA_MEDIUM
                if local_mad > 0:
                    new_value = max(current, local_median + beta * local_mad)
                else:
                    new_value = max(current + 1.0, float(stats["p75"]))
            elif anomaly_type == "drop_to_zero":
                new_value = 0.0
            elif anomaly_type == "activity_block":
                factor = ACTIVITY_FACTOR_STRONG if intensity == "strong" else ACTIVITY_FACTOR_MEDIUM
                new_value = max(current * factor, float(stats["p90"]))
            elif anomaly_type == "multivariate_group_peak":
                alpha = GLOBAL_ALPHA_STRONG if intensity == "strong" else GLOBAL_ALPHA_MEDIUM
                new_value = max(current, float(stats["p95"]), target_peak_value(current, stats, alpha, "p95"))
            else:
                raise ValueError(f"Unsupported anomaly type: {anomaly_type}")

            new_value = float(max(0.0, math.ceil(new_value)))
            data.loc[day, customer] = new_value
            injected_values[customer][format_date(day)] = new_value

    if anomaly_type == "multivariate_group_peak":
        scope = "multivariate_point"
    elif anomaly_type in SEGMENT_ANOMALIES:
        scope = "segment"
    else:
        scope = "point"

    return {
        "run_id": event["run_id"],
        "anomaly_id": event["anomaly_id"],
        "anomaly_type": anomaly_type,
        "intensity": intensity,
        "scope": scope,
        "customer": customers[0] if len(customers) == 1 else "",
        "customers": to_json(customers),
        "start_date": format_date(event["start_date"]),
        "end_date": format_date(event["end_date"]),
        "date": format_date(event["selected_date"]),
        "duration_days": int(event["duration_days"]),
        "original_values": to_json(original_values),
        "injected_values": to_json(injected_values),
        "seed": int(event["run_seed"]),
    }


def integerize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    rounded = raw.clip(lower=0).round().astype(int)
    rounded.index = pd.DatetimeIndex(pd.to_datetime(rounded.index).normalize(), name="date")
    rounded.columns = [str(col) for col in rounded.columns]
    return rounded


def wide_for_export(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out.insert(0, "date", [format_date(day) for day in out.index])
    return out


def stacked_for_export(raw: pd.DataFrame) -> pd.DataFrame:
    wide = wide_for_export(raw)
    stacked = wide.melt(id_vars=["date"], var_name="cols", value_name="data")
    return stacked[["date", "data", "cols"]].sort_values(["date", "cols"], kind="mergesort")


def write_run_outputs(run_dir: Path, raw: pd.DataFrame, labels: pd.DataFrame) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log1p = np.log1p(raw)
    wide_for_export(raw).to_csv(run_dir / "raw_wide.csv", index=False)
    wide_for_export(log1p).to_csv(run_dir / "log1p_wide.csv", index=False)
    stacked_for_export(raw).to_csv(run_dir / "raw_stacked.csv", index=False)
    stacked_for_export(log1p).to_csv(run_dir / "log1p_stacked.csv", index=False)
    labels.to_csv(run_dir / "labels.csv", index=False)


def run_synthetic_data(candidate_result_dir: Path, output_dir: Path) -> Path:
    candidate_result_dir = Path(candidate_result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = integerize_raw(load_raw_stacked(RAW_STACKED))
    customer_stats_df = customer_statistics(raw)
    candidates = candidate_source_summary(candidate_result_dir, pd.DatetimeIndex(raw.index))
    exclusion_calendar = build_exclusion_calendar(pd.DatetimeIndex(raw.index), candidates)
    eligible = exclusion_calendar[exclusion_calendar["excluded_reason"] == ""].copy()
    pools = build_pools(raw, customer_stats_df, exclusion_calendar)

    if eligible.empty:
        print("Warning: no eligible background days remain")

    candidates.to_csv(output_dir / "candidate_source_summary.csv", index=False)
    exclusion_calendar.to_csv(output_dir / "exclusion_calendar.csv", index=False)
    eligible.to_csv(output_dir / "eligible_background_days.csv", index=False)
    customer_stats_df.to_csv(output_dir / "customer_strata.csv", index=False)

    selected_by_run: Dict[str, List[Dict[str, Any]]] = {}
    for run_number in range(1, NUM_REPEATS + 1):
        run_id = f"run_{run_number:03d}"
        run_seed = SEED + run_number - 1
        selected = select_injection_plan(pools, run_id, run_seed)
        selected_by_run[run_id] = selected

    stat_map = stats_map(customer_stats_df)
    label_columns = [
        "run_id",
        "anomaly_id",
        "anomaly_type",
        "intensity",
        "scope",
        "customer",
        "customers",
        "start_date",
        "end_date",
        "date",
        "duration_days",
        "original_values",
        "injected_values",
        "seed",
    ]

    for run_id, events in selected_by_run.items():
        synthetic_raw = raw.copy()
        label_records = [apply_event(synthetic_raw, raw, event, stat_map) for event in events]
        synthetic_raw = integerize_raw(synthetic_raw)
        labels = pd.DataFrame(label_records, columns=label_columns)
        write_run_outputs(output_dir / run_id, synthetic_raw, labels)

    print(f"Wrote synthetic Systemanmeldung anomaly data to {output_dir}")
    return output_dir


def main() -> None:
    run_synthetic_data(
        Path("/Users/timadmin/Dokumente/TAB-MAC/result/anomaly_candidate_study/systemanmeldung_cluster04"),
        Path("/Users/timadmin/Dokumente/TAB-MAC/synthetic_runs/systemanmeldung_cluster04"),
    )


if __name__ == "__main__":
    main()
