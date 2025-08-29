#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from relbench.base import EntityTask, RecommendationTask

# Import relbench components
from relbench.tasks import get_task, get_task_names


def discover_ratebeer_tasks() -> Tuple[List[str], List[str]]:
    """Dynamically discover and categorize RateBeer tasks."""
    dataset_name = "rel-ratebeer"
    task_names = get_task_names(dataset_name)

    entity_tasks = []
    recommendation_tasks = []

    print(f"Discovering tasks for {dataset_name}...")
    for task_name in task_names:
        try:
            task = get_task(dataset_name, task_name, download=False)
            if isinstance(task, EntityTask):
                entity_tasks.append(task_name)
                print(f"  {task_name} -> EntityTask")
            elif isinstance(task, RecommendationTask):
                recommendation_tasks.append(task_name)
                print(f"  {task_name} -> RecommendationTask")
            else:
                print(f"  {task_name} -> Unknown task type: {type(task)}")
        except Exception as e:
            print(f"  Error loading {task_name}: {e}")

    return entity_tasks, recommendation_tasks


def setup_environment() -> str:
    """Setup environment and return activation command."""
    # Check if we're in the right directory
    if not os.path.exists("examples/gnn_entity.py"):
        raise RuntimeError("Must run from my-relbench directory")

    # Use absolute path for virtual environment
    venv_path = "/lfs/ampere1/0/justingu/venv2/bin/activate"
    if not os.path.exists(venv_path):
        raise RuntimeError(f"Virtual environment not found at {venv_path}")

    # Build activation command prefix
    activation_cmd = f"source {venv_path}"
    return activation_cmd


def check_cuda_devices() -> List[int]:
    """Check CUDA devices using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        devices = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(", ")
                idx, free_mem, total_mem = int(parts[0]), int(parts[1]), int(parts[2])
                # Consider device available if >80% memory is free
                if free_mem > 0.8 * total_mem:
                    devices.append(idx)
                print(
                    f"  GPU {idx}: {free_mem}MB free / {total_mem}MB total {'(available)' if idx in devices else '(busy)'}"
                )
        return devices
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not check CUDA devices")
        return []


def execute_command(
    cmd_args: List[str], cuda_device: Optional[int] = None, timeout: int = 3600
) -> Tuple[str, str, bool]:
    """Execute command with proper environment setup."""

    # Build environment
    env = os.environ.copy()
    env["HOME"] = "/lfs/ampere1/0/justingu/my-relbench/final"
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # Build full command with activation
    activation_cmd = setup_environment()
    cmd_str = " ".join(cmd_args)
    full_cmd = f"{activation_cmd} && cd /lfs/ampere1/0/justingu/my-relbench && python3 {cmd_str}"

    print(f"Executing: {cmd_str}")
    if cuda_device is not None:
        print(f"Using CUDA device: {cuda_device}")

    try:
        result = subprocess.run(
            ["bash", "-c", full_cmd],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/lfs/ampere1/0/justingu/my-relbench",
        )

        return result.stdout, result.stderr, result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout} seconds", False
    except Exception as e:
        return "", f"Error executing command: {e}", False


def parse_metrics_output(stdout: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Parse metrics from model output."""
    lines = stdout.strip().split("\n")

    val_metrics = None
    test_metrics = None

    for line in lines:
        line_lower = line.lower().strip()

        # Look for "Best Val metrics:", "Val:", or similar patterns
        if ("val:" in line_lower or "best val metrics:" in line_lower) and "{" in line:
            val_metrics = extract_metrics_from_line(line)

        # Look for test metrics
        elif (
            "test:" in line_lower or "best test metrics:" in line_lower
        ) and "{" in line:
            test_metrics = extract_metrics_from_line(line)

    return val_metrics, test_metrics


def extract_metrics_from_line(line: str) -> Optional[Dict]:
    """Extract metrics dictionary from a single line."""
    try:
        # Find dictionary-like content
        dict_match = re.search(r"\{[^}]+\}", line)
        if dict_match:
            dict_str = dict_match.group(0)

            # Clean up the string for JSON parsing
            # Handle np.float64() and similar numpy types
            dict_str = re.sub(r"np\.float64\(([\d.e-]+)\)", r"\1", dict_str)
            dict_str = re.sub(r"np\.int64\((\d+)\)", r"\1", dict_str)

            # Add quotes to keys
            dict_str = re.sub(r"(\w+):", r'"\1":', dict_str)

            # Replace single quotes with double quotes
            dict_str = re.sub(r"'([^']*)':", r'"\1":', dict_str)

            return json.loads(dict_str)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: manual parsing with numpy handling
    metrics = {}

    # Pattern to match: metric_name: value or metric_name: np.float64(value)
    patterns = [
        r"(\w+):\s*np\.float64\(([\d.e-]+)\)",  # np.float64() format
        r"(\w+):\s*np\.int64\((\d+)\)",  # np.int64() format
        r"(\w+):\s*([\d.e-]+)",  # regular float format
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, line):
            metric_name, value = match.groups()
            try:
                metrics[metric_name] = float(value)
            except ValueError:
                continue

    return metrics if metrics else None


def run_single_experiment(
    task_name: str,
    model_script: str,
    seed: int,
    cuda_device: Optional[int] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Run a single experiment with retry logic."""

    base_cmd = [
        f"examples/{model_script}.py",
        "--dataset",
        "rel-ratebeer",
        "--task",
        task_name,
        "--seed",
        str(seed),
    ]

    # Try different batch sizes for GNN models
    batch_sizes = [512, 256, 128, 64, 32, 16] if "gnn" in model_script else [None]

    for attempt, batch_size in enumerate(batch_sizes):
        if attempt >= max_retries:
            break

        cmd = base_cmd.copy()
        if batch_size is not None:
            cmd.extend(["--batch_size", str(batch_size)])

        stdout, stderr, success = execute_command(cmd, cuda_device)

        if success:
            val_metrics, test_metrics = parse_metrics_output(stdout)
            if val_metrics and test_metrics:
                return {
                    "task": task_name,
                    "model": model_script.replace(".py", ""),
                    "seed": seed,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "batch_size": batch_size,
                    "success": True,
                    "stdout": stdout[-1000:],  # Keep last 1000 chars for debugging
                    "timestamp": time.time(),
                }
            else:
                print(f"Warning: Could not parse metrics from output")
                print(f"Last few lines of stdout:")
                for line in stdout.split("\n")[-5:]:
                    if line.strip():
                        print(f"  {line}")

        # Check for OOM and retry with smaller batch size
        if "out of memory" in stderr.lower() or "cuda oom" in stderr.lower():
            print(
                f"OOM with batch_size={batch_size}, retrying with smaller batch size..."
            )
            continue
        else:
            # Other error, log and return failure
            print(f"Error: {stderr[-500:]}")  # Show last 500 chars of error
            return {
                "task": task_name,
                "model": model_script.replace(".py", ""),
                "seed": seed,
                "error": stderr[-500:],
                "success": False,
                "timestamp": time.time(),
            }

    return {
        "task": task_name,
        "model": model_script.replace(".py", ""),
        "seed": seed,
        "error": "Failed after all retries",
        "success": False,
        "timestamp": time.time(),
    }


def compute_aggregated_stats(results: List[Dict]) -> List[Dict]:
    """Compute mean/std statistics across seeds."""
    # Group by task and model
    grouped = defaultdict(list)

    for result in results:
        if result["success"]:
            key = (result["task"], result["model"])
            grouped[key].append(result)

    aggregated = []
    for (task, model), task_results in grouped.items():
        if len(task_results) >= 1:  # At least 1 run

            # Aggregate val metrics
            val_metrics_agg = {}
            test_metrics_agg = {}

            # Get all metric names
            all_val_metrics = set()
            all_test_metrics = set()
            for r in task_results:
                all_val_metrics.update(r["val_metrics"].keys())
                all_test_metrics.update(r["test_metrics"].keys())

            # Compute stats for each metric
            for metric in all_val_metrics:
                values = [
                    r["val_metrics"][metric]
                    for r in task_results
                    if metric in r["val_metrics"]
                ]
                if values:
                    val_metrics_agg[metric] = {
                        "mean": float(np.mean(values)),
                        "std": (
                            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        ),
                        "values": values,
                    }

            for metric in all_test_metrics:
                values = [
                    r["test_metrics"][metric]
                    for r in task_results
                    if metric in r["test_metrics"]
                ]
                if values:
                    test_metrics_agg[metric] = {
                        "mean": float(np.mean(values)),
                        "std": (
                            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        ),
                        "values": values,
                    }

            aggregated.append(
                {
                    "task": task,
                    "model": model,
                    "num_runs": len(task_results),
                    "seeds": [r["seed"] for r in task_results],
                    "val_metrics": val_metrics_agg,
                    "test_metrics": test_metrics_agg,
                }
            )

    return aggregated


def run_task_benchmark(
    task_name: str, seeds: List[int] = None, models: List[str] = None
):
    """Run benchmark for a specific task."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]  # Default 5 seeds

    print(f"=== Benchmarking Task: {task_name} ===")

    # Discover tasks to determine task type
    entity_tasks, recommendation_tasks = discover_ratebeer_tasks()

    # Determine task type and appropriate models
    if task_name in entity_tasks:
        task_type = "entity"
        default_models = ["baseline_entity", "lightgbm_entity", "gnn_entity"]
    elif task_name in recommendation_tasks:
        task_type = "recommendation"
        default_models = [
            "baseline_recommendation",
            "lightgbm_recommendation",
            "idgnn_recommendation",
        ]
    else:
        print(f"Error: Task '{task_name}' not found in RateBeer tasks!")
        print(f"Available entity tasks: {entity_tasks}")
        print(f"Available recommendation tasks: {recommendation_tasks}")
        return

    if models is None:
        models = default_models

    print(f"Task type: {task_type}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")

    # Check CUDA devices
    print(f"\nChecking CUDA devices...")
    cuda_devices = check_cuda_devices()
    print(f"Available CUDA devices: {cuda_devices}")

    # Check if we need CUDA for any of the models
    cuda_requiring_models = [m for m in models if "gnn" in m or "idgnn" in m]
    if cuda_requiring_models and not cuda_devices:
        print(
            f"\nError: No CUDA devices available, but the following models require GPU:"
        )
        for model in cuda_requiring_models:
            print(f"  - {model}")
        print(f"\nOptions:")
        print(f"1. Wait for GPU availability")
        print(
            f"2. Run without GPU models: --models {' '.join([m for m in models if m not in cuda_requiring_models])}"
        )
        print(f"3. Use a different machine with available GPUs")
        print(f"\nExiting...")
        return

    if cuda_requiring_models and cuda_devices:
        print(f"GPU models will use CUDA device: {cuda_devices[0]}")

    results = []

    # Run experiments
    for model in models:
        for seed in seeds:
            if model.startswith("baseline") and seed > 0:
                continue  # Baseline only needs to run once

            print(f"\n--- Running {task_name}-{model}-seed{seed} ---")

            # Assign CUDA device for GPU models, None for CPU models
            if "gnn" in model or "idgnn" in model:
                cuda_device = cuda_devices[0] if cuda_devices else None
                if cuda_device is None:
                    print(f"Skipping {model} - no CUDA device available")
                    continue
            else:
                cuda_device = None  # CPU models don't need GPU

            result = run_single_experiment(task_name, model, seed, cuda_device)
            results.append(result)

            if result["success"]:
                print(f"✓ Success!")
                print(f"  Val metrics: {result['val_metrics']}")
                print(f"  Test metrics: {result['test_metrics']}")
            else:
                print(f"✗ Failed: {result.get('error', 'Unknown error')}")

    # Save results to task-specific files
    print(f"\nSaving results...")

    # Individual results
    individual_file = f"results_{task_name}_individual.json"
    with open(individual_file, "w") as f:
        json.dump(results, f, indent=2)

    # Compute aggregated statistics
    aggregated = compute_aggregated_stats(results)

    # Aggregated results
    aggregated_file = f"results_{task_name}_aggregated.json"
    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Results saved to:")
    print(f"  Individual: {individual_file}")
    print(f"  Aggregated: {aggregated_file}")

    # Print summary
    print(f"\n=== Summary for {task_name} ===")
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Successful runs: {len(successful)}/{len(results)}")
    if failed:
        print("Failed runs:")
        for f in failed:
            print(f"  {f['model']}-seed{f['seed']}: {f.get('error', 'Unknown')}")

    if aggregated:
        print("\nAggregated Results:")
        for agg in aggregated:
            print(f"  {agg['model']} ({agg['num_runs']} runs):")
            for metric, stats in agg["test_metrics"].items():
                mean, std = stats["mean"], stats["std"]
                print(f"    {metric}: {mean:.3f} ± {std:.3f}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Benchmark RateBeer tasks")
    parser.add_argument(
        "task", nargs="?", help="Task name to benchmark (e.g., beer-rating-churn)"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Seeds to run (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to run (auto-detected based on task type if not specified)",
    )
    parser.add_argument(
        "--list-tasks", action="store_true", help="List all available tasks and exit"
    )

    args = parser.parse_args()

    if args.list_tasks:
        print("Available RateBeer tasks:")
        entity_tasks, recommendation_tasks = discover_ratebeer_tasks()
        print("\nEntity tasks:")
        for task in entity_tasks:
            print(f"  {task}")
        print("\nRecommendation tasks:")
        for task in recommendation_tasks:
            print(f"  {task}")
        return

    if not args.task:
        parser.error("Task name is required unless --list-tasks is used")

    run_task_benchmark(args.task, args.seeds, args.models)


if __name__ == "__main__":
    main()
