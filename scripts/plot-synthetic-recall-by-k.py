import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scripts.pipeline.step_D_final_original_methods import clean_features, run_catch, run_univariate

SYNTHETIC_DIR = Path("/Users/timadmin/Dokumente/TAB-MAC/results_/systemanmeldung_cluster04_20260618_211652/02_synthetic_data")
OUTPUT_DIR = Path("/Users/timadmin/Dokumente/TAB-MAC/results_/plots/systemanmeldung_synthetic_recall_by_k")
TOP_K = 200

ANOMALY_TYPES = [
    "global_peak",
    "contextual_peak",
    "drop_to_zero",
    "activity_block",
    "multivariate_group_peak",
]


def normalize_customer(value):
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def load_wide(path):
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date")
    data.columns = data.columns.astype(str)
    return data


def topk(scores, anomaly_type, method):
    rows = []
    for customer in scores.columns:
        table = pd.DataFrame(
            {
                "anomaly_type": anomaly_type,
                "method": method,
                "customer": customer,
                "date": scores.index,
                "score": scores[customer],
            }
        )
        table["rank"] = table["score"].rank(method="first", ascending=False)
        table = table[table["rank"] <= TOP_K].copy()
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def build_candidates(data):
    features = clean_features(data)

    units = run_univariate(features, "units")
    calf = run_univariate(features, "calf")
    catch = run_catch(features)
    catch = pd.DataFrame({"ALL": catch}, index=features.index)

    candidates = [
        topk(units, "global_peak", "units"),
        topk(calf, "contextual_peak", "calf"),
        topk(calf, "drop_to_zero", "calf"),
        topk(calf, "activity_block", "calf"),
        topk(catch, "multivariate_group_peak", "catch"),
    ]
    return pd.concat(candidates, ignore_index=True)


def label_customers(label):
    if label["anomaly_type"] == "multivariate_group_peak":
        return ["ALL"]
    if str(label.get("customer", "")).strip():
        return [normalize_customer(label["customer"])]
    return [normalize_customer(value) for value in json.loads(label["customers"])]


def label_hit(label, candidates, k):
    part = candidates[candidates["anomaly_type"] == label["anomaly_type"]].copy()
    part = part[part["rank"] <= k]
    if part.empty:
        return False

    customers = label_customers(label)
    part = part[part["customer"].astype(str).isin(customers)]
    if part.empty:
        return False

    dates = pd.to_datetime(part["date"]).dt.normalize()
    start = pd.Timestamp(label["start_date"]).normalize()
    end = pd.Timestamp(label["end_date"]).normalize()
    return bool(((dates >= start) & (dates <= end)).any())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_by_run = {}
    candidates_by_run = {}

    for run_dir in sorted(SYNTHETIC_DIR.glob("run_*")):
        print(run_dir.name, flush=True)
        labels = pd.read_csv(run_dir / "labels.csv", dtype=str, keep_default_na=False)
        labels_by_run[run_dir.name] = labels

        raw_candidates = build_candidates(load_wide(run_dir / "raw_wide.csv"))
        log1p_candidates = build_candidates(load_wide(run_dir / "log1p_wide.csv"))
        candidates_by_run[run_dir.name] = pd.concat([raw_candidates, log1p_candidates], ignore_index=True)

    rows = []
    for anomaly_type in ANOMALY_TYPES:
        for k in range(1, TOP_K + 1):
            label_count = 0
            hit_count = 0
            for run_id, labels in labels_by_run.items():
                candidates = candidates_by_run[run_id]
                labels_part = labels[labels["anomaly_type"] == anomaly_type]
                for _, label in labels_part.iterrows():
                    label_count += 1
                    if label_hit(label, candidates, k):
                        hit_count += 1
            recall = hit_count / label_count if label_count else 0
            rows.append({"anomaly_type": anomaly_type, "k": k, "recall": recall, "hits": hit_count, "labels": label_count})

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "synthetic_recall_by_k.csv", index=False)

    plt.figure(figsize=(13, 6))
    for anomaly_type in ANOMALY_TYPES:
        data = result[result["anomaly_type"] == anomaly_type]
        plt.plot(data["k"], data["recall"], label=anomaly_type)
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "synthetic_recall_by_k.png", dpi=150)
    plt.close()

    print(f"Dateien gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
