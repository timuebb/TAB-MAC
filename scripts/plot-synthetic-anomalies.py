from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


SYNTHETIC_DIR = Path("/Users/timadmin/Dokumente/TAB-MAC/results/systemanmeldung_cluster04_20260618_211652/02_synthetic_data")
OUTPUT_DIR = Path("/Users/timadmin/Dokumente/TAB-MAC/results") / "plots" / "systemanmeldung_synthetic_anomalies_by_run"

COLORS = {
    "global_peak": "#e41a1c",
    "contextual_peak": "#ff7f00",
    "drop_to_zero": "#984ea3",
    "activity_block": "#4daf4a",
    "multivariate_group_peak": "#ffff33",
}


def plot_run(run_dir, variant):
    labels = pd.read_csv(run_dir / "labels.csv")
    labels["start_date"] = pd.to_datetime(labels["start_date"])
    labels["end_date"] = pd.to_datetime(labels["end_date"])

    data = pd.read_csv(run_dir / f"{variant}_wide.csv")
    data["date"] = pd.to_datetime(data["date"])
    customers = list(data.columns[1:])

    fig, axes = plt.subplots(len(customers), 1, figsize=(14, 2 * len(customers) + 1.5), sharex=True)
    if len(customers) == 1:
        axes = [axes]

    for axis, customer in zip(axes, customers):
        offset = max(data[customer].max() * 0.03, 0.05)
        axis.bar(data["date"], data[customer], color="steelblue", width=1.0)
        axis.set_ylabel(customer)
        axis.set_ylim(0, data[customer].max() + offset * 5)

        for _, event in labels.iterrows():
            if pd.notna(event["customer"]):
                event_customers = [str(int(event["customer"]))]
            else:
                event_customers = [str(value) for value in json.loads(event["customers"])]

            if str(customer) not in event_customers:
                continue

            dates = pd.date_range(event["start_date"], event["end_date"])
            points = data[data["date"].isin(dates)]
            axis.scatter(
                points["date"],
                points[customer] + offset,
                color=COLORS[event["anomaly_type"]],
                s=80,
                edgecolors="black",
                linewidths=0.7,
                zorder=3,
            )

    legend = [
        Line2D([0], [0], marker="o", color="white", label=name, markerfacecolor=color, markeredgecolor="black")
        for name, color in COLORS.items()
    ]

    axes[-1].set_xlabel("Datum")
    fig.suptitle(f"{run_dir.name} - {variant}")
    fig.legend(handles=legend, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{run_dir.name}_{variant}.png", dpi=150)
    plt.close(fig)


def main():
    for run_dir in sorted(SYNTHETIC_DIR.glob("run_*")):
        plot_run(run_dir, "raw")
        plot_run(run_dir, "log1p")

    print(f"Plots gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
