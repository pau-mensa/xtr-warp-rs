#!/usr/bin/env python3

import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

DATASETS = [
    "scifact",
    "nfcorpus",
    "arguana",
    "quora",
    "scidocs",
    "fiqa",
    "trec-covid",
    "webis-touche2020",
]


def get_yaml_files(experiments_dir):
    yaml_files = list(experiments_dir.glob("*.yaml"))
    return sorted(yaml_files)


def run_benchmark(config_path, dataset):
    cmd = [
        "uv",
        "run",
        "benchmark/benchmark.py",
        str(config_path),
        "--dataset",
        dataset,
    ]

    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset}")
    print(f"Config: {config_path.name}")
    print(f"{'=' * 80}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    return elapsed, result.returncode == 0


def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_progress(
    current_exp,
    total_exp,
    elapsed_time,
    dataset,
    config_name,
    failed_experiments,
):
    remaining_exp = total_exp - current_exp

    if current_exp > 1:
        avg_time_per_exp = elapsed_time / current_exp
        estimated_remaining = avg_time_per_exp * remaining_exp
        estimated_total = avg_time_per_exp * total_exp
        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        eta_str = eta.strftime("%H:%M:%S")
    else:
        avg_time_per_exp = 0
        estimated_remaining = 0
        estimated_total = 0
        eta_str = "calculating..."

    print(f"\n{'=' * 80}")
    print(f"📊 PROGRESS REPORT")
    print(f"{'=' * 80}")
    print(
        f"Experiments: {current_exp}/{total_exp} ({current_exp / total_exp * 100:.1f}%)"
    )
    print(f"Remaining:    {remaining_exp}")
    print(f"Time elapsed: {format_timedelta(timedelta(seconds=elapsed_time))}")
    print(f"ETA:          {eta_str}")
    print(f"Avg time:     {avg_time_per_exp:.1f}s per experiment")
    print(f"Est. total:   {format_timedelta(timedelta(seconds=estimated_total))}")
    print(f"Est. remain:  {format_timedelta(timedelta(seconds=estimated_remaining))}")
    print(f"\nCurrent:")
    print(f"  Dataset:  {dataset}")
    print(f"  Config:   {config_name}")
    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    print(f"{'=' * 80}\n")


def main():
    script_dir = Path(__file__).parent.resolve()
    experiments_dir = script_dir / "experiments"

    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        sys.exit(1)

    yaml_files = get_yaml_files(experiments_dir)

    if not yaml_files:
        print(f"Error: No YAML files found in {experiments_dir}")
        sys.exit(1)

    total_experiments = len(DATASETS) * len(yaml_files)
    current_experiment = 0
    failed_experiments = []
    start_time = time.time()

    print(f"\n{'=' * 80}")
    print(f"🚀 BENCHMARK RUNNER")
    print(f"{'=' * 80}")
    print(f"Datasets: {DATASETS}")
    print(f"Configs:  {len(yaml_files)}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'=' * 80}\n")

    for dataset in DATASETS:
        for config_path in yaml_files:
            current_experiment += 1

            elapsed_time = time.time() - start_time

            config_name = config_path.stem
            full_exp_name = f"{dataset}_{config_name}"

            print_progress(
                current_experiment,
                total_experiments,
                elapsed_time,
                dataset,
                config_name,
                failed_experiments,
            )

            exp_start = time.time()
            elapsed, success = run_benchmark(config_path, dataset)
            exp_time = time.time() - exp_start

            if not success:
                failed_experiments.append(f"{full_exp_name} (error)")
                print(f"\n❌ Experiment failed: {full_exp_name}")
            else:
                print(f"\n✅ Experiment completed: {full_exp_name} ({exp_time:.1f}s)")

    total_time = time.time() - start_time
    success_count = total_experiments - len(failed_experiments)

    print(f"\n{'=' * 80}")
    print(f"🎉 BENCHMARK COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total time:       {format_timedelta(timedelta(seconds=total_time))}")
    print(f"Experiments run:  {total_experiments}")
    print(f"Successful:       {success_count}")
    print(f"Failed:           {len(failed_experiments)}")
    print(f"Avg per exp:     {total_time / total_experiments:.1f}s")
    if failed_experiments:
        print(f"\n❌ Failed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
