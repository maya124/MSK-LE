# File: synthesize_results.py
# Aggregates results from all experiments (metrics_eval_best_weights.json) into a parent folder
# Original code provided by Stanford's CS230 (See https://github.com/cs230-stanford/cs230-code-examples for the full code)

import argparse
import json
import os

from tabulate import tabulate


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments',
                    help='Directory containing results of experiments')


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.

    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`

    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, 'metrics_val_best_weights.json')
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def metrics_to_table(metrics):
    
    # Get the headers from the first subdir. Puts '-1' for metrics that don't exist
    headers = metrics[list(metrics.keys())[0]].keys()
    table = []
    for subdir, values in metrics.items():
        table.append([subdir] + [values[h] if h in values else -1 for h in headers])
    print(table)
    res = tabulate(table, headers, tablefmt='pipe', floatfmt=".2f")

    return res


if __name__ == "__main__":
    args = parser.parse_args()

    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(args.parent_dir, metrics)
    table = metrics_to_table(metrics)

    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = os.path.join(args.parent_dir, "results.md")
    print("Now saving to %s" % save_file)
    with open(save_file, 'w') as f:
        f.write(table)
