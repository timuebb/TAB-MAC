from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


RESULT_DIR = Path("/Users/timadmin/Dokumente/TAB-MAC/results_/systemanmeldung_cluster04_20260618_211652")
DATA_FILE = Path(
    "/Users/timadmin/Dokumente/TAB-MAC/dataset/anomaly_detect/data/atruvia_data/Systemanmeldung/customer_clustering/daily_systemanmeldung_log1p.csv"
)
THRESHOLD = 0.4


def main():
    data = pd.read_csv(DATA_FILE)
    data["date"] = pd.to_datetime(data["date"])
    wide = data.set_index("date")

    corr = wide.corr()
    distance = 1 - corr
    distance = distance.clip(lower=0)
    np.fill_diagonal(distance.values, 0)

    links = linkage(squareform(distance.values), method="average")

    plt.figure(figsize=(18, 8))
    dendrogram(
        links,
        labels=wide.columns.astype(str).tolist(),
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=THRESHOLD,
    )
    plt.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1.2, label=f"Schwelle {THRESHOLD}")
    plt.title("Dendrogramm der Systemanmeldung-Kunden")
    plt.ylabel("Distanz = 1 - Pearson-Korrelation")
    plt.legend()
    plt.tight_layout()

    output_dir = RESULT_DIR / "plots"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "dendrogram_alle_kunden_schwelle_0_4.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Gespeichert: {output_file}")


if __name__ == "__main__":
    main()
