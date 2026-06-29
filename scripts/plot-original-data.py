from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path("/Users/timadmin/Dokumente/TAB-MAC/dataset/anomaly_detect/data/atruvia_data/Systemanmeldung/Systemanmeldung_day_wide.csv")
OUTPUT_DIR = Path("/Users/timadmin/Dokumente/TAB-MAC/results/plots/systemanmeldung_original")


def plot_customers(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for customer in data.columns[1:]:
        plt.figure(figsize=(12, 4))
        plt.bar(data["date"], data[customer])
        plt.title(f"Verlauf {customer}")
        plt.xlabel("Datum")
        plt.ylabel("Anzahl")
        plt.tight_layout()
        plt.savefig(output_dir / f"{customer}.png")
        plt.close()


def main():
    raw_data = pd.read_csv(DATA_FILE)
    raw_data["date"] = pd.to_datetime(raw_data["date"])

    log1p_data = raw_data.copy()
    log1p_data.iloc[:, 1:] = np.log1p(log1p_data.iloc[:, 1:])

    plot_customers(raw_data, OUTPUT_DIR / "raw")
    plot_customers(log1p_data, OUTPUT_DIR / "log1p")

    print(f"Plots gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
