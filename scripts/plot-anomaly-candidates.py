from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path("/Users/timadmin/Dokumente/TAB-MAC/dataset/anomaly_detect/data/atruvia_data/Systemanmeldung/Systemanmeldung_day_wide.csv")
OUTPUT_DIR = Path("/Users/timadmin/Dokumente/TAB-MAC/results/plots/systemanmeldung_anomaly_candidates")


def load_candidates(variant):
    path = Path("/Users/timadmin/Dokumente/TAB-MAC/results/systemanmeldung_cluster04_20260618_211652/01_candidates") / variant / "customer_univariate" / "customer_univariate_consensus_topk.csv"
    candidates = pd.read_csv(path)
    candidates["customer"] = candidates["customer"].astype(str)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"])
    return candidates


def plot_customers(data, candidates, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "global_point": "red",
        "contextual_point": "orange",
        "subsequence_pattern": "purple",
    }

    for anomaly_type in colors:
        anomaly_dir = output_dir / anomaly_type
        anomaly_dir.mkdir(parents=True, exist_ok=True)

        for customer in data.columns[1:]:
            customer_candidates = candidates[candidates["customer"] == str(customer)]
            dates = customer_candidates[customer_candidates["anomaly_type"] == anomaly_type]["timestamp"]
            points = data[data["date"].isin(dates)]

            plt.figure(figsize=(12, 4))
            plt.bar(data["date"], data[customer], color="steelblue")

            if not points.empty:
                plt.scatter(
                    points["date"],
                    points[customer],
                    color=colors[anomaly_type],
                    s=45,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.4,
                    label=anomaly_type,
                    zorder=3,
                )

            plt.title(f"Verlauf {customer} - {anomaly_type}")
            plt.xlabel("Datum")
            plt.ylabel("Anzahl")
            if not points.empty:
                plt.legend()
            plt.tight_layout()
            plt.savefig(anomaly_dir / f"{customer}.png")
            plt.close()


def main():
    raw_data = pd.read_csv(DATA_FILE)
    raw_data["date"] = pd.to_datetime(raw_data["date"])

    log1p_data = raw_data.copy()
    log1p_data.iloc[:, 1:] = np.log1p(log1p_data.iloc[:, 1:])

    raw_candidates = load_candidates("raw")
    log1p_candidates = load_candidates("log1p")

    plot_customers(raw_data, raw_candidates, OUTPUT_DIR / "raw")
    plot_customers(log1p_data, log1p_candidates, OUTPUT_DIR / "log1p")

    print(f"Plots gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
