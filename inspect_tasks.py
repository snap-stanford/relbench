"""
Interactive inspection of relbench tasks.

This script:
1. Initializes tasks from the relbench package
2. Displays the heads of train/val/test tables for each
3. Performs relationship validation checks to ensure data integrity

Tasks inspected:
- rel-f1 / driver-top3: EntityTask (binary classification) - predicts if driver qualifies top-3
- rel-f1 / driver-race-compete: RecommendationTask (link prediction) - predicts which races a driver competes in
- rel-arxiv / co-citation: RecommendationTask (link prediction) - predicts which papers are co-cited together
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from relbench.datasets import get_dataset
from relbench.tasks import get_task

AVAILABLE_DATASETS = ["rel-f1", "rel-arxiv"]

# Global variable to store log file path
LOG_FILE_PATH: Path | None = None


def setup_logging(output_dir: str | Path = ".") -> Path:
    """Configure logging to output to both console and file.
    
    Args:
        output_dir: Directory to save the log file.
        
    Returns:
        Path to the log file.
    """
    global LOG_FILE_PATH
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"inspect_tasks_{timestamp}.log"
    log_file_path = output_dir / log_filename
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    # Console formatter without timestamp (cleaner output)
    console_formatter = logging.Formatter("%(message)s")
    # File formatter with timestamp for traceability
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    LOG_FILE_PATH = log_file_path
    return log_file_path


def print_separator(title: str, char: str = "=") -> None:
    """Print a visual separator with a title."""
    logging.info("\n" + char * 70)
    logging.info(f" {title}")
    logging.info(char * 70)


def display_table_head(table, split_name: str, n_rows: int = 10) -> None:
    """Display the head of a task table with metadata.
    
    Args:
        table: The relbench Table object
        split_name: Name of the split (train/val/test)
        n_rows: Number of rows to display
    """
    print_separator(f"{split_name.upper()} TABLE", "-")
    
    df = table.df
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"\nForeign key mappings: {table.fkey_col_to_pkey_table}")
    logging.info(f"Primary key column: {table.pkey_col}")
    logging.info(f"Time column: {table.time_col}")
    
    logging.info(f"\nFirst {n_rows} rows:")
    logging.info(df.head(n_rows).to_string())
    
    # Show date range
    if table.time_col and table.time_col in df.columns:
        logging.info(f"\nDate range: {df[table.time_col].min()} to {df[table.time_col].max()}")
        logging.info(f"Unique timestamps: {df[table.time_col].nunique()}")


def check_driver_id_validity(
    task_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    split_name: str,
) -> dict:
    """Check that all driverIds in the task table exist in the drivers table."""
    valid_driver_ids = set(drivers_df["driverId"].unique())
    task_driver_ids = set(task_df["driverId"].unique())
    
    missing_drivers = task_driver_ids - valid_driver_ids
    
    return {
        "split": split_name,
        "entity": "driverId",
        "total_unique": len(task_driver_ids),
        "valid_count": len(task_driver_ids - missing_drivers),
        "missing_count": len(missing_drivers),
        "missing_ids": list(missing_drivers)[:10] if missing_drivers else [],
        "passed": len(missing_drivers) == 0,
    }


def check_race_id_validity(
    task_df: pd.DataFrame,
    races_df: pd.DataFrame,
    split_name: str,
) -> dict:
    """Check that all raceIds in the task table exist in the races table.
    
    For RecommendationTask, raceId is a list column.
    """
    valid_race_ids = set(races_df["raceId"].unique())
    
    # Flatten the list column to get all race IDs
    all_race_ids = set()
    for race_list in task_df["raceId"]:
        if isinstance(race_list, list):
            all_race_ids.update(race_list)
    
    missing_races = all_race_ids - valid_race_ids
    
    return {
        "split": split_name,
        "entity": "raceId",
        "total_unique": len(all_race_ids),
        "valid_count": len(all_race_ids - missing_races),
        "missing_count": len(missing_races),
        "missing_ids": list(missing_races)[:10] if missing_races else [],
        "passed": len(missing_races) == 0,
    }


def check_circuit_id_validity(
    task_df: pd.DataFrame,
    circuits_df: pd.DataFrame,
    split_name: str,
) -> dict:
    """Check that all circuitIds in the task table exist in the circuits table.
    
    For RecommendationTask, circuitId is a list column.
    """
    valid_circuit_ids = set(circuits_df["circuitId"].unique())
    
    # Flatten the list column to get all circuit IDs
    all_circuit_ids = set()
    for circuit_list in task_df["circuitId"]:
        if isinstance(circuit_list, list):
            all_circuit_ids.update(circuit_list)
    
    missing_circuits = all_circuit_ids - valid_circuit_ids
    
    return {
        "split": split_name,
        "entity": "circuitId",
        "total_unique": len(all_circuit_ids),
        "valid_count": len(all_circuit_ids - missing_circuits),
        "missing_count": len(missing_circuits),
        "missing_ids": list(missing_circuits)[:10] if missing_circuits else [],
        "passed": len(missing_circuits) == 0,
    }


def check_temporal_constraints(
    task_df: pd.DataFrame,
    split_name: str,
    val_timestamp: pd.Timestamp,
    test_timestamp: pd.Timestamp,
) -> dict:
    """Check that timestamps in each split are in the expected range."""
    dates = task_df["date"]
    min_date = dates.min()
    max_date = dates.max()
    
    if split_name == "train":
        expected_condition = max_date < val_timestamp
        constraint_desc = f"all dates < val_timestamp ({val_timestamp})"
    elif split_name == "val":
        expected_condition = (min_date >= val_timestamp) and (max_date < test_timestamp)
        constraint_desc = f"val_timestamp <= dates < test_timestamp"
    else:  # test
        expected_condition = min_date >= test_timestamp
        constraint_desc = f"all dates >= test_timestamp ({test_timestamp})"
    
    return {
        "split": split_name,
        "check": "temporal_constraints",
        "min_date": str(min_date),
        "max_date": str(max_date),
        "expected": constraint_desc,
        "passed": expected_condition,
    }


def check_target_column_validity(
    task_df: pd.DataFrame,
    split_name: str,
    target_col: str,
) -> dict:
    """Check that the target column has valid values for binary classification."""
    target_values = task_df[target_col]
    unique_values = set(target_values.unique())
    expected_values = {0, 1}
    
    invalid_values = unique_values - expected_values
    null_count = target_values.isna().sum()
    
    return {
        "split": split_name,
        "check": "target_column_validity",
        "target_col": target_col,
        "unique_values": sorted(unique_values),
        "invalid_values": list(invalid_values) if invalid_values else [],
        "null_count": int(null_count),
        "passed": len(invalid_values) == 0 and null_count == 0,
    }


def check_qualifying_table_relationship(
    task_df: pd.DataFrame,
    qualifying_df: pd.DataFrame,
    split_name: str,
) -> dict:
    """Check that drivers in the task exist in the qualifying table."""
    task_driver_ids = set(task_df["driverId"].unique())
    qualifying_driver_ids = set(qualifying_df["driverId"].unique())
    
    missing_from_qualifying = task_driver_ids - qualifying_driver_ids
    
    return {
        "split": split_name,
        "check": "qualifying_table_relationship",
        "task_drivers": len(task_driver_ids),
        "qualifying_drivers": len(qualifying_driver_ids),
        "missing_from_qualifying": len(missing_from_qualifying),
        "missing_ids": list(missing_from_qualifying)[:10] if missing_from_qualifying else [],
        "passed": len(missing_from_qualifying) == 0,
    }


def check_paper_id_validity(
    task_df: pd.DataFrame,
    papers_df: pd.DataFrame,
    split_name: str,
    paper_col: str = "Paper_ID",
) -> dict:
    """Check that all Paper_IDs in the task table exist in the papers table."""
    valid_paper_ids = set(papers_df["Paper_ID"].unique())
    task_paper_ids = set(task_df[paper_col].unique())
    
    missing_papers = task_paper_ids - valid_paper_ids
    
    return {
        "split": split_name,
        "entity": paper_col,
        "total_unique": len(task_paper_ids),
        "valid_count": len(task_paper_ids - missing_papers),
        "missing_count": len(missing_papers),
        "missing_ids": list(missing_papers)[:10] if missing_papers else [],
        "passed": len(missing_papers) == 0,
    }


def check_co_cited_paper_validity(
    task_df: pd.DataFrame,
    papers_df: pd.DataFrame,
    split_name: str,
) -> dict:
    """Check that all co_cited Paper_IDs (list column) exist in the papers table."""
    valid_paper_ids = set(papers_df["Paper_ID"].unique())
    
    # Flatten the list column to get all co-cited paper IDs
    all_co_cited_ids = set()
    for paper_list in task_df["co_cited"]:
        if isinstance(paper_list, list):
            all_co_cited_ids.update(paper_list)
    
    missing_papers = all_co_cited_ids - valid_paper_ids
    
    return {
        "split": split_name,
        "entity": "co_cited",
        "total_unique": len(all_co_cited_ids),
        "valid_count": len(all_co_cited_ids - missing_papers),
        "missing_count": len(missing_papers),
        "missing_ids": list(missing_papers)[:10] if missing_papers else [],
        "passed": len(missing_papers) == 0,
    }


def check_class_balance(
    task_df: pd.DataFrame,
    split_name: str,
    target_col: str,
) -> dict:
    """Check class balance for binary classification."""
    target_values = task_df[target_col]
    total = len(target_values)
    positives = (target_values == 1).sum()
    negatives = (target_values == 0).sum()
    
    return {
        "split": split_name,
        "check": "class_balance",
        "total": total,
        "positives": int(positives),
        "negatives": int(negatives),
        "positive_rate": round(positives / total * 100, 2) if total > 0 else 0,
        "passed": True,  # Informational check
    }


def check_list_column_structure(
    task_df: pd.DataFrame,
    split_name: str,
    list_col: str,
) -> dict:
    """Check that a column contains lists as expected for link prediction."""
    col_values = task_df[list_col]
    
    all_lists = all(isinstance(x, list) for x in col_values)
    list_lengths = [len(x) if isinstance(x, list) else 0 for x in col_values]
    empty_lists = sum(1 for length in list_lengths if length == 0)
    
    return {
        "split": split_name,
        "check": "list_column_structure",
        "column": list_col,
        "all_are_lists": all_lists,
        "total_rows": len(task_df),
        "empty_lists": empty_lists,
        "min_list_length": min(list_lengths) if list_lengths else 0,
        "max_list_length": max(list_lengths) if list_lengths else 0,
        "avg_list_length": round(sum(list_lengths) / len(list_lengths), 2) if list_lengths else 0,
        "passed": all_lists and empty_lists == 0,
    }


def print_check_result(result: dict) -> None:
    """Print the result of a validation check."""
    status = "PASSED" if result.get("passed") else "FAILED"
    color_code = "\033[92m" if result.get("passed") else "\033[91m"
    reset_code = "\033[0m"
    
    check_name = result.get("entity", result.get("check", "unknown"))
    logging.info(f"\n{color_code}[{status}]{reset_code} {result.get('split', 'N/A').upper()} - {check_name}")
    
    for key, value in result.items():
        if key not in ("passed", "split", "entity", "check"):
            logging.info(f"  {key}: {value}")


def plot_timestamp_histogram(
    train_table,
    val_table,
    test_table,
    task_name: str,
    time_col: str = "date",
    num_bins: int = 150,
    save_path: str | None = None,
) -> None:
    """Plot a stacked histogram of timestamps showing train/val/test distribution.
    
    Args:
        train_table: The training split table
        val_table: The validation split table
        test_table: The test split table
        task_name: Name of the task for the plot title
        time_col: Name of the time column in the tables
        num_bins: Number of bins for the histogram
        save_path: Optional path to save the plot (if None, displays interactively)
    """
    train_timestamps = train_table.df[time_col]
    val_timestamps = val_table.df[time_col]
    test_timestamps = test_table.df[time_col]
    
    # Combine all timestamps to determine bin edges
    all_timestamps = pd.concat([train_timestamps, val_timestamps, test_timestamps])
    min_time = all_timestamps.min()
    max_time = all_timestamps.max()
    
    # Create bin edges based on all data
    bin_edges = pd.date_range(start=min_time, end=max_time, periods=num_bins + 1)
    
    # Compute histogram counts for each split
    train_counts, _ = np.histogram(train_timestamps, bins=bin_edges)
    val_counts, _ = np.histogram(val_timestamps, bins=bin_edges)
    test_counts, _ = np.histogram(test_timestamps, bins=bin_edges)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate bin centers for plotting (as matplotlib date numbers)
    bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    bin_centers_mpl = mdates.date2num(bin_centers.to_pydatetime())
    
    # Calculate bar width in matplotlib date units
    bar_width_mpl = mdates.date2num(bin_edges[1].to_pydatetime()) - mdates.date2num(bin_edges[0].to_pydatetime())
    bar_width_mpl *= 0.9  # Slightly smaller for visual separation
    
    # Plot stacked bars
    colors = {
        "train": "#3498db",  # Blue
        "val": "#f39c12",    # Orange
        "test": "#e74c3c",   # Red
    }
    
    ax.bar(
        bin_centers_mpl,
        train_counts,
        width=bar_width_mpl,
        label=f"Train ({len(train_timestamps):,} rows)",
        color=colors["train"],
        alpha=0.8,
    )
    ax.bar(
        bin_centers_mpl,
        val_counts,
        width=bar_width_mpl,
        bottom=train_counts,
        label=f"Val ({len(val_timestamps):,} rows)",
        color=colors["val"],
        alpha=0.8,
    )
    ax.bar(
        bin_centers_mpl,
        test_counts,
        width=bar_width_mpl,
        bottom=train_counts + val_counts,
        label=f"Test ({len(test_timestamps):,} rows)",
        color=colors["test"],
        alpha=0.8,
    )
    
    # Plot vertical lines for all unique timestamps in each split
    unique_train_timestamps = train_timestamps.unique()
    unique_val_timestamps = val_timestamps.unique()
    unique_test_timestamps = test_timestamps.unique()
    
    logging.info(f"Unique timestamps - Train: {len(unique_train_timestamps)}, Val: {len(unique_val_timestamps)}, Test: {len(unique_test_timestamps)}")
    
    line_alpha = 0.4
    line_width = 0.5
    
    # Train timestamps - dashed lines
    for ts in unique_train_timestamps:
        ax.axvline(
            x=mdates.date2num(pd.Timestamp(ts).to_pydatetime()),
            color=colors["train"],
            linestyle="--",
            linewidth=line_width,
            alpha=line_alpha,
        )
    
    # Validation timestamps - dashed lines
    for ts in unique_val_timestamps:
        ax.axvline(
            x=mdates.date2num(pd.Timestamp(ts).to_pydatetime()),
            color=colors["val"],
            linestyle="--",
            linewidth=line_width,
            alpha=line_alpha,
        )
    
    # Test timestamps - dashed lines
    for ts in unique_test_timestamps:
        ax.axvline(
            x=mdates.date2num(pd.Timestamp(ts).to_pydatetime()),
            color=colors["test"],
            linestyle="--",
            linewidth=line_width,
            alpha=line_alpha,
        )
    
    # Format the x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")
    
    # Labels and title
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Number of Rows", fontsize=12)
    ax.set_title(f"Timestamp Distribution by Split: {task_name}", fontsize=14, fontweight="bold")
    
    # Legend with vertical line indicators
    from matplotlib.lines import Line2D
    custom_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors["train"], alpha=0.8, label=f"Train ({len(train_timestamps):,} rows)"),
        plt.Rectangle((0, 0), 1, 1, fc=colors["val"], alpha=0.8, label=f"Val ({len(val_timestamps):,} rows)"),
        plt.Rectangle((0, 0), 1, 1, fc=colors["test"], alpha=0.8, label=f"Test ({len(test_timestamps):,} rows)"),
        Line2D([0], [0], color=colors["train"], linestyle="--", linewidth=1, alpha=0.7, label=f"Train timestamps ({len(unique_train_timestamps)})"),
        Line2D([0], [0], color=colors["val"], linestyle="--", linewidth=1, alpha=0.7, label=f"Val timestamps ({len(unique_val_timestamps)})"),
        Line2D([0], [0], color=colors["test"], linestyle="--", linewidth=1, alpha=0.7, label=f"Test timestamps ({len(unique_test_timestamps)})"),
    ]
    ax.legend(handles=custom_handles, loc="upper left", fontsize=9)
    
    # Grid for readability
    ax.grid(axis="y", alpha=0.3)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"\nPlot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def inspect_driver_top3(dataset, db):
    """Inspect the driver-top3 EntityTask."""
    print_separator("DRIVER-TOP3 TASK (EntityTask - Binary Classification)")
    
    logging.info("\nLoading task... (download=False)")
    task = get_task("rel-f1", "driver-top3", download=False)
    
    logging.info(f"Task: {task}")
    logging.info(f"Task type: {task.task_type}")
    logging.info(f"Entity column: {task.entity_col}")
    logging.info(f"Entity table: {task.entity_table}")
    logging.info(f"Target column: {task.target_col}")
    logging.info(f"Time column: {task.time_col}")
    logging.info(f"Time delta: {task.timedelta}")
    logging.info(f"Num eval timestamps: {task.num_eval_timestamps}")
    logging.info(f"Metrics: {[m.__name__ for m in task.metrics]}")
    
    drivers_df = db.table_dict["drivers"].df
    qualifying_df = db.table_dict["qualifying"].df
    
    logging.info(f"\nDrivers table: {len(drivers_df)} rows")
    logging.info(f"Qualifying table: {len(qualifying_df)} rows")
    
    # Load all splits
    logging.info("\nLoading train/val/test tables...")
    train_table = task.get_table("train", mask_input_cols=False)
    val_table = task.get_table("val", mask_input_cols=False)
    test_table = task.get_table("test", mask_input_cols=False)
    
    # Display table heads
    display_table_head(train_table, "train")
    display_table_head(val_table, "val")
    display_table_head(test_table, "test")
    
    # Run validation checks
    print_separator("VALIDATION CHECKS", "-")
    
    all_checks = []
    
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        
        # Check driver ID validity
        driver_check = check_driver_id_validity(df, drivers_df, split_name)
        all_checks.append(driver_check)
        print_check_result(driver_check)
        
        # Check temporal constraints
        temporal_check = check_temporal_constraints(
            df, split_name, dataset.val_timestamp, dataset.test_timestamp
        )
        all_checks.append(temporal_check)
        print_check_result(temporal_check)
        
        # Check target column validity
        target_check = check_target_column_validity(df, split_name, task.target_col)
        all_checks.append(target_check)
        print_check_result(target_check)
        
        # Check qualifying table relationship
        qualifying_check = check_qualifying_table_relationship(df, qualifying_df, split_name)
        all_checks.append(qualifying_check)
        print_check_result(qualifying_check)
        
        # Check class balance
        balance_check = check_class_balance(df, split_name, task.target_col)
        all_checks.append(balance_check)
        print_check_result(balance_check)
    
    # Summary
    print_separator("VALIDATION SUMMARY", "-")
    print_validation_summary(all_checks)
    
    # Statistics
    print_separator("TASK STATISTICS", "-")
    print_entity_task_statistics(task, train_table, val_table, test_table)
    
    # Plot timestamp histogram
    print_separator("TIMESTAMP DISTRIBUTION PLOT", "-")
    plot_timestamp_histogram(
        train_table,
        val_table,
        test_table,
        task_name="driver-top3",
        time_col=task.time_col,
        save_path="driver_top3_timestamp_histogram.png",
    )
    
    return task, train_table, val_table, test_table, all_checks


def inspect_driver_race_compete(dataset, db):
    """Inspect the driver-race-compete RecommendationTask."""
    print_separator("DRIVER-RACE-COMPETE TASK (RecommendationTask - Link Prediction)")
    
    logging.info("\nLoading task... (download=False)")
    task = get_task("rel-f1", "driver-race-compete", download=False)
    
    logging.info(f"Task: {task}")
    logging.info(f"Task type: {task.task_type}")
    logging.info(f"Source entity: {task.src_entity_col} -> {task.src_entity_table}")
    logging.info(f"Destination entity: {task.dst_entity_col} -> {task.dst_entity_table}")
    logging.info(f"Time column: {task.time_col}")
    logging.info(f"Time delta: {task.timedelta}")
    logging.info(f"Eval k: {task.eval_k}")
    logging.info(f"Metrics: {[m.__name__ for m in task.metrics]}")
    
    drivers_df = db.table_dict["drivers"].df
    races_df = db.table_dict["races"].df
    qualifying_df = db.table_dict["qualifying"].df
    
    logging.info(f"\nDrivers table: {len(drivers_df)} rows")
    logging.info(f"Races table: {len(races_df)} rows")
    logging.info(f"Qualifying table: {len(qualifying_df)} rows")
    
    # Load all splits
    logging.info("\nLoading train/val/test tables...")
    train_table = task.get_table("train", mask_input_cols=False)
    val_table = task.get_table("val", mask_input_cols=False)
    test_table = task.get_table("test", mask_input_cols=False)
    
    # Display table heads
    display_table_head(train_table, "train")
    display_table_head(val_table, "val")
    display_table_head(test_table, "test")
    
    # Run validation checks
    print_separator("VALIDATION CHECKS", "-")
    
    all_checks = []
    
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        
        # Check driver ID validity
        driver_check = check_driver_id_validity(df, drivers_df, split_name)
        all_checks.append(driver_check)
        print_check_result(driver_check)
        
        # Check race ID validity (list column)
        race_check = check_race_id_validity(df, races_df, split_name)
        all_checks.append(race_check)
        print_check_result(race_check)
        
        # Check temporal constraints
        temporal_check = check_temporal_constraints(
            df, split_name, dataset.val_timestamp, dataset.test_timestamp
        )
        all_checks.append(temporal_check)
        print_check_result(temporal_check)
        
        # Check list column structure
        list_check = check_list_column_structure(df, split_name, task.dst_entity_col)
        all_checks.append(list_check)
        print_check_result(list_check)
        
        # Check qualifying table relationship (task uses qualifying table)
        qualifying_check = check_qualifying_table_relationship(df, qualifying_df, split_name)
        all_checks.append(qualifying_check)
        print_check_result(qualifying_check)
    
    # Summary
    print_separator("VALIDATION SUMMARY", "-")
    print_validation_summary(all_checks)
    
    # Statistics
    print_separator("TASK STATISTICS", "-")
    print_recommendation_task_statistics(task, train_table, val_table, test_table)
    
    # Plot timestamp histogram
    print_separator("TIMESTAMP DISTRIBUTION PLOT", "-")
    plot_timestamp_histogram(
        train_table,
        val_table,
        test_table,
        task_name="driver-race-compete",
        time_col=task.time_col,
        save_path="driver_race_compete_timestamp_histogram.png",
    )
    
    return task, train_table, val_table, test_table, all_checks


def inspect_driver_circuit_compete(dataset, db):
    """Inspect the driver-circuit-compete RecommendationTask."""
    print_separator("DRIVER-CIRCUIT-COMPETE TASK (RecommendationTask - Link Prediction)")
    
    logging.info("\nLoading task... (download=False)")
    task = get_task("rel-f1", "driver-circuit-compete", download=False)
    
    logging.info(f"Task: {task}")
    logging.info(f"Task type: {task.task_type}")
    logging.info(f"Source entity: {task.src_entity_col} -> {task.src_entity_table}")
    logging.info(f"Destination entity: {task.dst_entity_col} -> {task.dst_entity_table}")
    logging.info(f"Time column: {task.time_col}")
    logging.info(f"Time delta: {task.timedelta}")
    logging.info(f"Eval k: {task.eval_k}")
    logging.info(f"Metrics: {[m.__name__ for m in task.metrics]}")
    
    drivers_df = db.table_dict["drivers"].df
    circuits_df = db.table_dict["circuits"].df
    races_df = db.table_dict["races"].df
    results_df = db.table_dict["results"].df
    
    logging.info(f"\nDrivers table: {len(drivers_df)} rows")
    logging.info(f"Circuits table: {len(circuits_df)} rows")
    logging.info(f"Races table: {len(races_df)} rows")
    logging.info(f"Results table: {len(results_df)} rows")
    
    # Load all splits
    logging.info("\nLoading train/val/test tables...")
    train_table = task.get_table("train", mask_input_cols=False)
    val_table = task.get_table("val", mask_input_cols=False)
    test_table = task.get_table("test", mask_input_cols=False)
    
    # Display table heads
    display_table_head(train_table, "train")
    display_table_head(val_table, "val")
    display_table_head(test_table, "test")
    
    # Run validation checks
    print_separator("VALIDATION CHECKS", "-")
    
    all_checks = []
    
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        
        # Check driver ID validity
        driver_check = check_driver_id_validity(df, drivers_df, split_name)
        all_checks.append(driver_check)
        print_check_result(driver_check)
        
        # Check circuit ID validity (list column)
        circuit_check = check_circuit_id_validity(df, circuits_df, split_name)
        all_checks.append(circuit_check)
        print_check_result(circuit_check)
        
        # Check temporal constraints
        temporal_check = check_temporal_constraints(
            df, split_name, dataset.val_timestamp, dataset.test_timestamp
        )
        all_checks.append(temporal_check)
        print_check_result(temporal_check)
        
        # Check list column structure
        list_check = check_list_column_structure(df, split_name, task.dst_entity_col)
        all_checks.append(list_check)
        print_check_result(list_check)
    
    # Summary
    print_separator("VALIDATION SUMMARY", "-")
    print_validation_summary(all_checks)
    
    # Statistics
    print_separator("TASK STATISTICS", "-")
    print_circuit_compete_task_statistics(task, train_table, val_table, test_table, circuits_df)
    
    # Plot timestamp histogram
    print_separator("TIMESTAMP DISTRIBUTION PLOT", "-")
    plot_timestamp_histogram(
        train_table,
        val_table,
        test_table,
        task_name="driver-circuit-compete",
        time_col=task.time_col,
        save_path="driver_circuit_compete_timestamp_histogram.png",
    )
    
    return task, train_table, val_table, test_table, all_checks


def print_validation_summary(all_checks: list) -> None:
    """Print summary of validation checks."""
    passed_count = sum(1 for check in all_checks if check.get("passed"))
    total_count = len(all_checks)
    
    logging.info(f"Total checks: {total_count}")
    logging.info(f"Passed: {passed_count}")
    logging.info(f"Failed: {total_count - passed_count}")
    
    if passed_count == total_count:
        logging.info("\n\033[92mAll validation checks passed!\033[0m")
    else:
        logging.info("\n\033[91mSome validation checks failed. Review the details above.\033[0m")
        logging.info("\nFailed checks:")
        for check in all_checks:
            if not check.get("passed"):
                split = check.get("split", "N/A")
                check_name = check.get("entity", check.get("check", "unknown"))
                logging.info(f"  - {split.upper()}: {check_name}")


def print_entity_task_statistics(task, train_table, val_table, test_table) -> None:
    """Print statistics for an EntityTask."""
    logging.info("\nPer-split statistics:")
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        logging.info(f"\n{split_name.upper()}:")
        logging.info(f"  Rows: {len(df)}")
        logging.info(f"  Unique drivers: {df['driverId'].nunique()}")
        logging.info(f"  Unique timestamps: {df['date'].nunique()}")
        logging.info(f"  Positives (top-3): {(df[task.target_col] == 1).sum()}")
        logging.info(f"  Negatives: {(df[task.target_col] == 0).sum()}")
        logging.info(f"  Positive rate: {(df[task.target_col] == 1).mean() * 100:.2f}%")
    
    # Driver overlap analysis
    logging.info("\nDriver overlap analysis:")
    train_drivers = set(train_table.df["driverId"].unique())
    val_drivers = set(val_table.df["driverId"].unique())
    test_drivers = set(test_table.df["driverId"].unique())
    
    logging.info(f"  Train drivers: {len(train_drivers)}")
    logging.info(f"  Val drivers: {len(val_drivers)}")
    logging.info(f"  Test drivers: {len(test_drivers)}")
    logging.info(f"  Train ∩ Val: {len(train_drivers & val_drivers)}")
    logging.info(f"  Train ∩ Test: {len(train_drivers & test_drivers)}")
    logging.info(f"  Val ∩ Test: {len(val_drivers & test_drivers)}")
    
    test_only = test_drivers - train_drivers - val_drivers
    logging.info(f"  Test-only (cold-start): {len(test_only)}")


def print_recommendation_task_statistics(task, train_table, val_table, test_table) -> None:
    """Print statistics for a RecommendationTask."""
    logging.info("\nPer-split statistics:")
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        logging.info(f"\n{split_name.upper()}:")
        logging.info(f"  Rows: {len(df)}")
        logging.info(f"  Unique drivers: {df['driverId'].nunique()}")
        logging.info(f"  Unique timestamps: {df['date'].nunique()}")
        
        total_races = sum(len(race_list) for race_list in df["raceId"])
        unique_races = len(set(race for race_list in df["raceId"] for race in race_list))
        logging.info(f"  Total race entries: {total_races}")
        logging.info(f"  Unique races: {unique_races}")
        logging.info(f"  Avg races per driver-timestamp: {round(total_races / len(df), 2) if len(df) > 0 else 0}")
    
    # Driver overlap analysis
    logging.info("\nDriver overlap analysis:")
    train_drivers = set(train_table.df["driverId"].unique())
    val_drivers = set(val_table.df["driverId"].unique())
    test_drivers = set(test_table.df["driverId"].unique())
    
    logging.info(f"  Train drivers: {len(train_drivers)}")
    logging.info(f"  Val drivers: {len(val_drivers)}")
    logging.info(f"  Test drivers: {len(test_drivers)}")
    logging.info(f"  Train ∩ Val: {len(train_drivers & val_drivers)}")
    logging.info(f"  Train ∩ Test: {len(train_drivers & test_drivers)}")
    logging.info(f"  Val ∩ Test: {len(val_drivers & test_drivers)}")
    
    test_only = test_drivers - train_drivers - val_drivers
    logging.info(f"  Test-only (cold-start): {len(test_only)}")
    
    # Race overlap analysis
    logging.info("\nRace overlap analysis:")
    train_races = set(race for race_list in train_table.df["raceId"] for race in race_list)
    val_races = set(race for race_list in val_table.df["raceId"] for race in race_list)
    test_races = set(race for race_list in test_table.df["raceId"] for race in race_list)
    
    logging.info(f"  Train races: {len(train_races)}")
    logging.info(f"  Val races: {len(val_races)}")
    logging.info(f"  Test races: {len(test_races)}")
    logging.info(f"  Train ∩ Test: {len(train_races & test_races)}")
    
    test_only_races = test_races - train_races - val_races
    logging.info(f"  Test-only races (unseen): {len(test_only_races)}")


def print_circuit_compete_task_statistics(task, train_table, val_table, test_table, circuits_df) -> None:
    """Print statistics for the driver-circuit-compete RecommendationTask with link balance analysis."""
    total_circuits_in_db = len(circuits_df)
    
    logging.info("\nPer-split statistics:")
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        logging.info(f"\n{split_name.upper()}:")
        logging.info(f"  Rows: {len(df)}")
        logging.info(f"  Unique drivers: {df['driverId'].nunique()}")
        logging.info(f"  Unique timestamps: {df['date'].nunique()}")
        
        total_circuits = sum(len(circuit_list) for circuit_list in df["circuitId"])
        unique_circuits = len(set(circuit for circuit_list in df["circuitId"] for circuit in circuit_list))
        list_lengths = [len(circuit_list) for circuit_list in df["circuitId"]]
        logging.info(f"  Total circuit entries (positive links): {total_circuits}")
        logging.info(f"  Unique circuits: {unique_circuits}")
        logging.info(f"  Avg circuits per driver-timestamp: {round(total_circuits / len(df), 2) if len(df) > 0 else 0}")
        logging.info(f"  Min/Max circuits per driver-timestamp: {min(list_lengths)}/{max(list_lengths)}")
    
    # Driver overlap analysis
    logging.info("\nDriver overlap analysis:")
    train_drivers = set(train_table.df["driverId"].unique())
    val_drivers = set(val_table.df["driverId"].unique())
    test_drivers = set(test_table.df["driverId"].unique())
    
    logging.info(f"  Train drivers: {len(train_drivers)}")
    logging.info(f"  Val drivers: {len(val_drivers)}")
    logging.info(f"  Test drivers: {len(test_drivers)}")
    logging.info(f"  Train ∩ Val: {len(train_drivers & val_drivers)}")
    logging.info(f"  Train ∩ Test: {len(train_drivers & test_drivers)}")
    logging.info(f"  Val ∩ Test: {len(val_drivers & test_drivers)}")
    
    test_only = test_drivers - train_drivers - val_drivers
    logging.info(f"  Test-only (cold-start): {len(test_only)}")
    
    # Circuit overlap analysis
    logging.info("\nCircuit overlap analysis:")
    train_circuits = set(circuit for circuit_list in train_table.df["circuitId"] for circuit in circuit_list)
    val_circuits = set(circuit for circuit_list in val_table.df["circuitId"] for circuit in circuit_list)
    test_circuits = set(circuit for circuit_list in test_table.df["circuitId"] for circuit in circuit_list)
    
    logging.info(f"  Total circuits in database: {total_circuits_in_db}")
    logging.info(f"  Train circuits: {len(train_circuits)}")
    logging.info(f"  Val circuits: {len(val_circuits)}")
    logging.info(f"  Test circuits: {len(test_circuits)}")
    logging.info(f"  Train ∩ Val: {len(train_circuits & val_circuits)}")
    logging.info(f"  Train ∩ Test: {len(train_circuits & test_circuits)}")
    logging.info(f"  Val ∩ Test: {len(val_circuits & test_circuits)}")
    
    test_only_circuits = test_circuits - train_circuits - val_circuits
    logging.info(f"  Test-only circuits (unseen): {len(test_only_circuits)}")
    
    # Link balance analysis
    logging.info("\nLink balance analysis (positive vs potential links):")
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        num_drivers = df["driverId"].nunique()
        unique_circuits = len(set(circuit for circuit_list in df["circuitId"] for circuit in circuit_list))
        total_positive_links = sum(len(circuit_list) for circuit_list in df["circuitId"])
        
        # Potential links = num_drivers * num_circuits (for each timestamp)
        # But since we're looking at per-driver basis, it's simpler to look at coverage
        potential_links_per_driver = total_circuits_in_db
        avg_positive_per_driver = total_positive_links / num_drivers if num_drivers > 0 else 0
        coverage_ratio = avg_positive_per_driver / potential_links_per_driver if potential_links_per_driver > 0 else 0
        
        logging.info(f"\n  {split_name.upper()}:")
        logging.info(f"    Total positive links: {total_positive_links}")
        logging.info(f"    Unique drivers: {num_drivers}")
        logging.info(f"    Unique circuits in split: {unique_circuits}")
        logging.info(f"    Avg positive links per driver: {round(avg_positive_per_driver, 2)}")
        logging.info(f"    Potential circuits per driver: {potential_links_per_driver}")
        logging.info(f"    Coverage ratio (pos/potential): {round(coverage_ratio * 100, 2)}%")


def print_co_citation_task_statistics(task, train_table, val_table, test_table) -> None:
    """Print statistics for the co-citation RecommendationTask."""
    logging.info("\nPer-split statistics:")
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        logging.info(f"\n{split_name.upper()}:")
        logging.info(f"  Rows: {len(df)}")
        logging.info(f"  Unique source papers: {df['Paper_ID'].nunique()}")
        logging.info(f"  Unique timestamps: {df['date'].nunique()}")
        
        total_co_cited = sum(len(paper_list) for paper_list in df["co_cited"])
        unique_co_cited = len(set(paper for paper_list in df["co_cited"] for paper in paper_list))
        list_lengths = [len(paper_list) for paper_list in df["co_cited"]]
        logging.info(f"  Total co-cited entries: {total_co_cited}")
        logging.info(f"  Unique co-cited papers: {unique_co_cited}")
        logging.info(f"  Avg co-cited per paper-timestamp: {round(total_co_cited / len(df), 2) if len(df) > 0 else 0}")
        logging.info(f"  Min/Max co-cited list length: {min(list_lengths)}/{max(list_lengths)}")
    
    # Paper overlap analysis
    logging.info("\nSource paper overlap analysis:")
    train_papers = set(train_table.df["Paper_ID"].unique())
    val_papers = set(val_table.df["Paper_ID"].unique())
    test_papers = set(test_table.df["Paper_ID"].unique())
    
    logging.info(f"  Train papers: {len(train_papers)}")
    logging.info(f"  Val papers: {len(val_papers)}")
    logging.info(f"  Test papers: {len(test_papers)}")
    logging.info(f"  Train ∩ Val: {len(train_papers & val_papers)}")
    logging.info(f"  Train ∩ Test: {len(train_papers & test_papers)}")
    logging.info(f"  Val ∩ Test: {len(val_papers & test_papers)}")
    
    test_only = test_papers - train_papers - val_papers
    logging.info(f"  Test-only (cold-start): {len(test_only)}")
    
    # Co-cited paper overlap analysis
    logging.info("\nCo-cited paper overlap analysis:")
    train_co_cited = set(paper for paper_list in train_table.df["co_cited"] for paper in paper_list)
    val_co_cited = set(paper for paper_list in val_table.df["co_cited"] for paper in paper_list)
    test_co_cited = set(paper for paper_list in test_table.df["co_cited"] for paper in paper_list)
    
    logging.info(f"  Train co-cited papers: {len(train_co_cited)}")
    logging.info(f"  Val co-cited papers: {len(val_co_cited)}")
    logging.info(f"  Test co-cited papers: {len(test_co_cited)}")
    logging.info(f"  Train ∩ Test: {len(train_co_cited & test_co_cited)}")
    
    test_only_co_cited = test_co_cited - train_co_cited - val_co_cited
    logging.info(f"  Test-only co-cited (unseen): {len(test_only_co_cited)}")


def inspect_co_citation(dataset, db):
    """Inspect the co-citation RecommendationTask from rel-arxiv."""
    print_separator("CO-CITATION TASK (RecommendationTask - Link Prediction)")
    
    logging.info("\nLoading task... (download=False)")
    task = get_task("rel-arxiv", "co-citation", download=False)
    
    logging.info(f"Task: {task}")
    logging.info(f"Task type: {task.task_type}")
    logging.info(f"Source entity: {task.src_entity_col} -> {task.src_entity_table}")
    logging.info(f"Destination entity: {task.dst_entity_col} -> {task.dst_entity_table}")
    logging.info(f"Time column: {task.time_col}")
    logging.info(f"Time delta: {task.timedelta}")
    logging.info(f"Eval k: {task.eval_k}")
    logging.info(f"Metrics: {[m.__name__ for m in task.metrics]}")
    
    papers_df = db.table_dict["papers"].df
    citations_df = db.table_dict["citations"].df
    
    logging.info(f"\nPapers table: {len(papers_df)} rows")
    logging.info(f"Citations table: {len(citations_df)} rows")
    
    # Load all splits
    logging.info("\nLoading train/val/test tables...")
    train_table = task.get_table("train", mask_input_cols=False)
    val_table = task.get_table("val", mask_input_cols=False)
    test_table = task.get_table("test", mask_input_cols=False)
    
    # Display table heads
    display_table_head(train_table, "train")
    display_table_head(val_table, "val")
    display_table_head(test_table, "test")
    
    # Run validation checks
    print_separator("VALIDATION CHECKS", "-")
    
    all_checks = []
    
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        
        # Check source paper ID validity
        paper_check = check_paper_id_validity(df, papers_df, split_name)
        all_checks.append(paper_check)
        print_check_result(paper_check)
        
        # Check co-cited paper ID validity (list column)
        co_cited_check = check_co_cited_paper_validity(df, papers_df, split_name)
        all_checks.append(co_cited_check)
        print_check_result(co_cited_check)
        
        # Check temporal constraints
        temporal_check = check_temporal_constraints(
            df, split_name, dataset.val_timestamp, dataset.test_timestamp
        )
        all_checks.append(temporal_check)
        print_check_result(temporal_check)
        
        # Check list column structure
        list_check = check_list_column_structure(df, split_name, task.dst_entity_col)
        all_checks.append(list_check)
        print_check_result(list_check)
    
    # Summary
    print_separator("VALIDATION SUMMARY", "-")
    print_validation_summary(all_checks)
    
    # Statistics
    print_separator("TASK STATISTICS", "-")
    print_co_citation_task_statistics(task, train_table, val_table, test_table)
    
    # Plot timestamp histogram
    print_separator("TIMESTAMP DISTRIBUTION PLOT", "-")
    plot_timestamp_histogram(
        train_table,
        val_table,
        test_table,
        task_name="co-citation",
        time_col=task.time_col,
        save_path="co_citation_timestamp_histogram.png",
    )
    
    return task, train_table, val_table, test_table, all_checks


def main():
    """Main function to load dataset and inspect both tasks."""
    print_separator("LOADING REL-F1 DATASET")
    
    logging.info("Loading dataset... (download=True)")
    dataset = get_dataset("rel-f1", download=True)
    
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Val timestamp: {dataset.val_timestamp}")
    logging.info(f"Test timestamp: {dataset.test_timestamp}")
    
    db = dataset.get_db(upto_test_timestamp=False)
    logging.info(f"\nDatabase tables: {list(db.table_dict.keys())}")
    for table_name, table in db.table_dict.items():
        logging.info(f"  {table_name}: {len(table.df)} rows")
    
    # Store results for interactive exploration
    results = {}
    
    # Inspect driver-top3 task
    logging.info("\n")
    task1, train1, val1, test1, checks1 = inspect_driver_top3(dataset, db)
    results["driver_top3"] = {
        "task": task1,
        "train": train1,
        "val": val1,
        "test": test1,
        "checks": checks1,
    }
    
    # Inspect driver-race-compete task
    logging.info("\n")
    task2, train2, val2, test2, checks2 = inspect_driver_race_compete(dataset, db)
    results["driver_race_compete"] = {
        "task": task2,
        "train": train2,
        "val": val2,
        "test": test2,
        "checks": checks2,
    }
    
    # Inspect driver-circuit-compete task
    logging.info("\n")
    task3, train3, val3, test3, checks3 = inspect_driver_circuit_compete(dataset, db)
    results["driver_circuit_compete"] = {
        "task": task3,
        "train": train3,
        "val": val3,
        "test": test3,
        "checks": checks3,
    }
    
    # Final summary
    print_separator("FINAL SUMMARY")
    
    all_checks = checks1 + checks2 + checks3
    passed = sum(1 for c in all_checks if c.get("passed"))
    total = len(all_checks)
    
    logging.info(f"Total checks across all tasks: {total}")
    logging.info(f"Passed: {passed}")
    logging.info(f"Failed: {total - passed}")
    
    if passed == total:
        logging.info("\n\033[92mAll validation checks passed for all tasks!\033[0m")
    else:
        logging.info("\n\033[91mSome checks failed. Review details above.\033[0m")
    
    return dataset, db, results


def main_arxiv():
    """Main function to load arxiv dataset and inspect co-citation task."""
    print_separator("LOADING REL-ARXIV DATASET")
    
    logging.info("Loading dataset... (download=True)")
    dataset = get_dataset("rel-arxiv", download=True)
    
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Val timestamp: {dataset.val_timestamp}")
    logging.info(f"Test timestamp: {dataset.test_timestamp}")
    
    db = dataset.get_db(upto_test_timestamp=False)
    logging.info(f"\nDatabase tables: {list(db.table_dict.keys())}")
    for table_name, table in db.table_dict.items():
        logging.info(f"  {table_name}: {len(table.df)} rows")
    
    # Store results for interactive exploration
    results = {}
    
    # Inspect co-citation task
    logging.info("\n")
    task, train, val, test, checks = inspect_co_citation(dataset, db)
    results["co_citation"] = {
        "task": task,
        "train": train,
        "val": val,
        "test": test,
        "checks": checks,
    }
    
    # Final summary
    print_separator("FINAL SUMMARY")
    
    passed = sum(1 for c in checks if c.get("passed"))
    total = len(checks)
    
    logging.info(f"Total checks: {total}")
    logging.info(f"Passed: {passed}")
    logging.info(f"Failed: {total - passed}")
    
    if passed == total:
        logging.info("\n\033[92mAll validation checks passed!\033[0m")
    else:
        logging.info("\n\033[91mSome checks failed. Review details above.\033[0m")
    
    return dataset, db, results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive inspection of relbench tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_tasks.py                    # Inspect all datasets (default)
  python inspect_tasks.py rel-f1             # Inspect only rel-f1 tasks
  python inspect_tasks.py rel-arxiv          # Inspect only rel-arxiv tasks
  python inspect_tasks.py rel-f1 rel-arxiv   # Inspect both datasets
        """,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=AVAILABLE_DATASETS + [[]],
        help=f"Dataset(s) to inspect. Choices: {AVAILABLE_DATASETS}. Default: all datasets.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Use all datasets when none specified (nargs="*" returns [] not default)
    datasets_to_inspect = args.datasets if args.datasets else AVAILABLE_DATASETS
    
    # Set up logging to both console and file
    log_file_path = setup_logging(output_dir=".")
    logging.info(f"Logging to: {log_file_path.resolve()}")
    
    all_results = {}
    
    if "rel-f1" in datasets_to_inspect:
        dataset_f1, db_f1, results_f1 = main()
        all_results["rel-f1"] = {
            "dataset": dataset_f1,
            "db": db_f1,
            "results": results_f1,
        }
    
    if "rel-arxiv" in datasets_to_inspect:
        dataset_arxiv, db_arxiv, results_arxiv = main_arxiv()
        all_results["rel-arxiv"] = {
            "dataset": dataset_arxiv,
            "db": db_arxiv,
            "results": results_arxiv,
        }
    
    # Print available variables for interactive exploration
    logging.info("\n" + "=" * 70)
    logging.info(" Variables available for interactive exploration:")
    logging.info("   - all_results: Dict with results for each inspected dataset")
    for dataset_name in datasets_to_inspect:
        logging.info(f"   - all_results['{dataset_name}']: dataset, db, results")
    logging.info("=" * 70)
    
    # Print log file location at the end
    logging.info("\n" + "=" * 70)
    logging.info(" LOG FILE SAVED")
    logging.info(f"   Output log: {log_file_path.resolve()}")
    logging.info("=" * 70)
