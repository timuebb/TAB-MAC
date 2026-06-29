from datetime import datetime
from pathlib import Path

from step_A_candidate_study import run_candidate_study
from step_B_synthetic_data import run_synthetic_data
from step_C_method_runs import run_method_evaluation

RAW_CSV = Path("/Users/timadmin/Dokumente/TAB-MAC/dataset/anomaly_detect/data/atruvia_data/Systemanmeldung/Systemanmeldung_day_raw_stacked.csv")
LOG1P_CSV = Path("/Users/timadmin/Dokumente/TAB-MAC/dataset/anomaly_detect/data/atruvia_data/Systemanmeldung/Systemanmeldung_day_log1p_stacked.csv")
RESULTS_ROOT = Path("/Users/timadmin/Dokumente/TAB-MAC/results")

CANDIDATE_TOP_K = 25
METHOD_TOP_K = 50
EVAL_K = [10, 25, 50]


def run_pipeline():
    run_id = f"systemanmeldung_cluster04_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_dir = RESULTS_ROOT / run_id
    candidate_dir = result_dir / "01_candidates"
    synthetic_dir = result_dir / "02_synthetic_data"
    method_runs_dir = result_dir / "03_method_runs"

    if result_dir.exists():
        raise FileExistsError(result_dir)
    result_dir.mkdir(parents=True)

    print(f"Run ID: {run_id}", flush=True)
    print(f"Result: {result_dir}", flush=True)

    print("START step_A_candidate_study", flush=True)
    run_candidate_study(RAW_CSV, LOG1P_CSV, candidate_dir, CANDIDATE_TOP_K)
    print(f"OK step_A_candidate_study -> {candidate_dir}", flush=True)

    print("START step_B_synthetic_data", flush=True)
    run_synthetic_data(
        candidate_result_dir=candidate_dir,
        output_dir=synthetic_dir,
    )
    print(f"OK step_B_synthetic_data -> {synthetic_dir}", flush=True)

    print("START step_C_method_runs", flush=True)
    run_method_evaluation(
        synthetic_dir=synthetic_dir,
        output_dir=method_runs_dir,
        method_top_k=METHOD_TOP_K,
        eval_k=EVAL_K,
        segment_tolerance_days=0,
    )
    print(f"OK step_C_method_runs -> {method_runs_dir}", flush=True)

    print("Pipeline finished.", flush=True)
    print(f"Result: {result_dir}", flush=True)
    return result_dir


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
