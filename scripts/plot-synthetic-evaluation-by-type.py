from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("/Users/timadmin/Dokumente/TAB-MAC/results_/systemanmeldung_cluster04_20260618_211652/03_method_runs/synthetic_eval_summary.csv")
OUTPUT_DIR = Path("/Users/timadmin/Dokumente/TAB-MAC/results_/plots/systemanmeldung_synthetic_eval_by_type")
K_VALUES = [10, 25, 50]

TITLE_SIZE = 22
SUBTITLE_SIZE = 18
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 15


def main():
    data = pd.read_csv(INPUT_FILE)
    data = data[data["variant"] == "combined"].copy()

    table = data.groupby(["anomaly_type", "method", "result_group", "k"], as_index=False).agg(
        detected_count=("detected_count", "sum"),
        label_count=("label_count", "sum"),
        candidate_count=("candidate_count", "sum"),
    )

    table["recall_at_k"] = table["detected_count"] / table["label_count"]
    table["precision_at_k"] = table["detected_count"] / table["candidate_count"]
    table["f1_at_k"] = 2 * table["precision_at_k"] * table["recall_at_k"] / (
        table["precision_at_k"] + table["recall_at_k"]
    )
    table["f1_at_k"] = table["f1_at_k"].fillna(0)

    csv_dir = OUTPUT_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_dir / "all_anomaly_types.csv", index=False)

    for anomaly_type in sorted(table["anomaly_type"].unique()):
        anomaly_data = table[table["anomaly_type"] == anomaly_type].copy()
        anomaly_data.to_csv(csv_dir / f"{anomaly_type}.csv", index=False)

        fig, axes = plt.subplots(3, 1, figsize=(16, 13.5))
        groups = anomaly_data["result_group"].nunique()

        for axis, k in zip(axes, K_VALUES):
            second_axis = axis.twinx()
            plot_data = anomaly_data[anomaly_data["k"] == k].copy()
            plot_data = plot_data.sort_values(["recall_at_k", "precision_at_k"], ascending=False)
            plot_data["name"] = plot_data["method"]
            if groups > 1:
                plot_data["name"] = plot_data["method"] + "\n" + plot_data["result_group"]

            x = list(range(len(plot_data)))
            recall_bars = axis.bar([value - 0.25 for value in x], plot_data["recall_at_k"], width=0.25, label="Recall@k")
            precision_bars = second_axis.bar(x, plot_data["precision_at_k"], width=0.25, color="tab:orange", label="Precision@k")
            f1_bars = second_axis.bar([value + 0.25 for value in x], plot_data["f1_at_k"], width=0.25, color="tab:green", label="F1@k")
            axis.set_title(f"k = {k}", fontsize=SUBTITLE_SIZE)
            axis.set_ylim(0, 1)
            second_axis.set_ylim(0, max(0.1, plot_data[["precision_at_k", "f1_at_k"]].max().max() * 1.2))
            axis.set_ylabel("Recall@k", fontsize=AXIS_LABEL_SIZE)
            second_axis.set_ylabel("Precision@k / F1@k", fontsize=AXIS_LABEL_SIZE)
            axis.set_xticks(x)
            axis.set_xticklabels(plot_data["name"], rotation=65, ha="right", fontsize=TICK_LABEL_SIZE)
            axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
            second_axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
            axis.grid(axis="y", alpha=0.3)

        axes[0].legend([recall_bars, precision_bars, f1_bars], ["Recall@k", "Precision@k", "F1@k"], loc="upper right", fontsize=LEGEND_SIZE)
        fig.suptitle(anomaly_type, fontsize=TITLE_SIZE)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(OUTPUT_DIR / f"{anomaly_type}.png", dpi=150)
        plt.close(fig)

    print(f"Dateien gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
