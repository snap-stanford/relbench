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

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from relbench.tasks import get_task
from relbench.datasets import get_dataset


def print_separator(title: str, char: str = "=") -> None:
    """Print a visual separator with a title."""
    print("\n" + char * 70)
    print(f" {title}")
    print(char * 70)


def display_table_head(table, split_name: str, n_rows: int = 10) -> None:
    """Display the head of a task table with metadata.
    
    Args:
        table: The relbench Table object
        split_name: Name of the split (train/val/test)
        n_rows: Number of rows to display
    """
    print_separator(f"{split_name.upper()} TABLE", "-")
    
    df = table.df
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nForeign key mappings: {table.fkey_col_to_pkey_table}")
    print(f"Primary key column: {table.pkey_col}")
    print(f"Time column: {table.time_col}")
    
    print(f"\nFirst {n_rows} rows:")
    print(df.head(n_rows).to_string())
    
    # Show date range
    if table.time_col and table.time_col in df.columns:
        print(f"\nDate range: {df[table.time_col].min()} to {df[table.time_col].max()}")
        print(f"Unique timestamps: {df[table.time_col].nunique()}")


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
    print(f"\n{color_code}[{status}]{reset_code} {result.get('split', 'N/A').upper()} - {check_name}")
    
    for key, value in result.items():
        if key not in ("passed", "split", "entity", "check"):
            print(f"  {key}: {value}")


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
    
    print(f"Unique timestamps - Train: {len(unique_train_timestamps)}, Val: {len(unique_val_timestamps)}, Test: {len(unique_test_timestamps)}")
    
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
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def inspect_driver_top3(dataset, db):
    """Inspect the driver-top3 EntityTask."""
    print_separator("DRIVER-TOP3 TASK (EntityTask - Binary Classification)")
    
    print("\nLoading task... (download=False)")
    task = get_task("rel-f1", "driver-top3", download=False)
    
    print(f"Task: {task}")
    print(f"Task type: {task.task_type}")
    print(f"Entity column: {task.entity_col}")
    print(f"Entity table: {task.entity_table}")
    print(f"Target column: {task.target_col}")
    print(f"Time column: {task.time_col}")
    print(f"Time delta: {task.timedelta}")
    print(f"Num eval timestamps: {task.num_eval_timestamps}")
    print(f"Metrics: {[m.__name__ for m in task.metrics]}")
    
    drivers_df = db.table_dict["drivers"].df
    qualifying_df = db.table_dict["qualifying"].df
    
    print(f"\nDrivers table: {len(drivers_df)} rows")
    print(f"Qualifying table: {len(qualifying_df)} rows")
    
    # Load all splits
    print("\nLoading train/val/test tables...")
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
    
    print("\nLoading task... (download=False)")
    task = get_task("rel-f1", "driver-race-compete", download=False)
    
    print(f"Task: {task}")
    print(f"Task type: {task.task_type}")
    print(f"Source entity: {task.src_entity_col} -> {task.src_entity_table}")
    print(f"Destination entity: {task.dst_entity_col} -> {task.dst_entity_table}")
    print(f"Time column: {task.time_col}")
    print(f"Time delta: {task.timedelta}")
    print(f"Eval k: {task.eval_k}")
    print(f"Metrics: {[m.__name__ for m in task.metrics]}")
    
    drivers_df = db.table_dict["drivers"].df
    races_df = db.table_dict["races"].df
    qualifying_df = db.table_dict["qualifying"].df
    
    print(f"\nDrivers table: {len(drivers_df)} rows")
    print(f"Races table: {len(races_df)} rows")
    print(f"Qualifying table: {len(qualifying_df)} rows")
    
    # Load all splits
    print("\nLoading train/val/test tables...")
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


def print_validation_summary(all_checks: list) -> None:
    """Print summary of validation checks."""
    passed_count = sum(1 for check in all_checks if check.get("passed"))
    total_count = len(all_checks)
    
    print(f"Total checks: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    
    if passed_count == total_count:
        print("\n\033[92mAll validation checks passed!\033[0m")
    else:
        print("\n\033[91mSome validation checks failed. Review the details above.\033[0m")
        print("\nFailed checks:")
        for check in all_checks:
            if not check.get("passed"):
                split = check.get("split", "N/A")
                check_name = check.get("entity", check.get("check", "unknown"))
                print(f"  - {split.upper()}: {check_name}")


def print_entity_task_statistics(task, train_table, val_table, test_table) -> None:
    """Print statistics for an EntityTask."""
    print("\nPer-split statistics:")
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        print(f"\n{split_name.upper()}:")
        print(f"  Rows: {len(df)}")
        print(f"  Unique drivers: {df['driverId'].nunique()}")
        print(f"  Unique timestamps: {df['date'].nunique()}")
        print(f"  Positives (top-3): {(df[task.target_col] == 1).sum()}")
        print(f"  Negatives: {(df[task.target_col] == 0).sum()}")
        print(f"  Positive rate: {(df[task.target_col] == 1).mean() * 100:.2f}%")
    
    # Driver overlap analysis
    print("\nDriver overlap analysis:")
    train_drivers = set(train_table.df["driverId"].unique())
    val_drivers = set(val_table.df["driverId"].unique())
    test_drivers = set(test_table.df["driverId"].unique())
    
    print(f"  Train drivers: {len(train_drivers)}")
    print(f"  Val drivers: {len(val_drivers)}")
    print(f"  Test drivers: {len(test_drivers)}")
    print(f"  Train ∩ Val: {len(train_drivers & val_drivers)}")
    print(f"  Train ∩ Test: {len(train_drivers & test_drivers)}")
    print(f"  Val ∩ Test: {len(val_drivers & test_drivers)}")
    
    test_only = test_drivers - train_drivers - val_drivers
    print(f"  Test-only (cold-start): {len(test_only)}")


def print_recommendation_task_statistics(task, train_table, val_table, test_table) -> None:
    """Print statistics for a RecommendationTask."""
    print("\nPer-split statistics:")
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        print(f"\n{split_name.upper()}:")
        print(f"  Rows: {len(df)}")
        print(f"  Unique drivers: {df['driverId'].nunique()}")
        print(f"  Unique timestamps: {df['date'].nunique()}")
        
        total_races = sum(len(race_list) for race_list in df["raceId"])
        unique_races = len(set(race for race_list in df["raceId"] for race in race_list))
        print(f"  Total race entries: {total_races}")
        print(f"  Unique races: {unique_races}")
        print(f"  Avg races per driver-timestamp: {round(total_races / len(df), 2) if len(df) > 0 else 0}")
    
    # Driver overlap analysis
    print("\nDriver overlap analysis:")
    train_drivers = set(train_table.df["driverId"].unique())
    val_drivers = set(val_table.df["driverId"].unique())
    test_drivers = set(test_table.df["driverId"].unique())
    
    print(f"  Train drivers: {len(train_drivers)}")
    print(f"  Val drivers: {len(val_drivers)}")
    print(f"  Test drivers: {len(test_drivers)}")
    print(f"  Train ∩ Val: {len(train_drivers & val_drivers)}")
    print(f"  Train ∩ Test: {len(train_drivers & test_drivers)}")
    print(f"  Val ∩ Test: {len(val_drivers & test_drivers)}")
    
    test_only = test_drivers - train_drivers - val_drivers
    print(f"  Test-only (cold-start): {len(test_only)}")
    
    # Race overlap analysis
    print("\nRace overlap analysis:")
    train_races = set(race for race_list in train_table.df["raceId"] for race in race_list)
    val_races = set(race for race_list in val_table.df["raceId"] for race in race_list)
    test_races = set(race for race_list in test_table.df["raceId"] for race in race_list)
    
    print(f"  Train races: {len(train_races)}")
    print(f"  Val races: {len(val_races)}")
    print(f"  Test races: {len(test_races)}")
    print(f"  Train ∩ Test: {len(train_races & test_races)}")
    
    test_only_races = test_races - train_races - val_races
    print(f"  Test-only races (unseen): {len(test_only_races)}")


def print_co_citation_task_statistics(task, train_table, val_table, test_table) -> None:
    """Print statistics for the co-citation RecommendationTask."""
    print("\nPer-split statistics:")
    for split_name, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        df = table.df
        print(f"\n{split_name.upper()}:")
        print(f"  Rows: {len(df)}")
        print(f"  Unique source papers: {df['Paper_ID'].nunique()}")
        print(f"  Unique timestamps: {df['date'].nunique()}")
        
        total_co_cited = sum(len(paper_list) for paper_list in df["co_cited"])
        unique_co_cited = len(set(paper for paper_list in df["co_cited"] for paper in paper_list))
        list_lengths = [len(paper_list) for paper_list in df["co_cited"]]
        print(f"  Total co-cited entries: {total_co_cited}")
        print(f"  Unique co-cited papers: {unique_co_cited}")
        print(f"  Avg co-cited per paper-timestamp: {round(total_co_cited / len(df), 2) if len(df) > 0 else 0}")
        print(f"  Min/Max co-cited list length: {min(list_lengths)}/{max(list_lengths)}")
    
    # Paper overlap analysis
    print("\nSource paper overlap analysis:")
    train_papers = set(train_table.df["Paper_ID"].unique())
    val_papers = set(val_table.df["Paper_ID"].unique())
    test_papers = set(test_table.df["Paper_ID"].unique())
    
    print(f"  Train papers: {len(train_papers)}")
    print(f"  Val papers: {len(val_papers)}")
    print(f"  Test papers: {len(test_papers)}")
    print(f"  Train ∩ Val: {len(train_papers & val_papers)}")
    print(f"  Train ∩ Test: {len(train_papers & test_papers)}")
    print(f"  Val ∩ Test: {len(val_papers & test_papers)}")
    
    test_only = test_papers - train_papers - val_papers
    print(f"  Test-only (cold-start): {len(test_only)}")
    
    # Co-cited paper overlap analysis
    print("\nCo-cited paper overlap analysis:")
    train_co_cited = set(paper for paper_list in train_table.df["co_cited"] for paper in paper_list)
    val_co_cited = set(paper for paper_list in val_table.df["co_cited"] for paper in paper_list)
    test_co_cited = set(paper for paper_list in test_table.df["co_cited"] for paper in paper_list)
    
    print(f"  Train co-cited papers: {len(train_co_cited)}")
    print(f"  Val co-cited papers: {len(val_co_cited)}")
    print(f"  Test co-cited papers: {len(test_co_cited)}")
    print(f"  Train ∩ Test: {len(train_co_cited & test_co_cited)}")
    
    test_only_co_cited = test_co_cited - train_co_cited - val_co_cited
    print(f"  Test-only co-cited (unseen): {len(test_only_co_cited)}")


def inspect_co_citation(dataset, db):
    """Inspect the co-citation RecommendationTask from rel-arxiv."""
    print_separator("CO-CITATION TASK (RecommendationTask - Link Prediction)")
    
    print("\nLoading task... (download=False)")
    task = get_task("rel-arxiv", "co-citation", download=False)
    
    print(f"Task: {task}")
    print(f"Task type: {task.task_type}")
    print(f"Source entity: {task.src_entity_col} -> {task.src_entity_table}")
    print(f"Destination entity: {task.dst_entity_col} -> {task.dst_entity_table}")
    print(f"Time column: {task.time_col}")
    print(f"Time delta: {task.timedelta}")
    print(f"Eval k: {task.eval_k}")
    print(f"Metrics: {[m.__name__ for m in task.metrics]}")
    
    papers_df = db.table_dict["papers"].df
    citations_df = db.table_dict["citations"].df
    
    print(f"\nPapers table: {len(papers_df)} rows")
    print(f"Citations table: {len(citations_df)} rows")
    
    # Load all splits
    print("\nLoading train/val/test tables...")
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
    
    print("Loading dataset... (download=True)")
    dataset = get_dataset("rel-f1", download=True)
    
    print(f"Dataset: {dataset}")
    print(f"Val timestamp: {dataset.val_timestamp}")
    print(f"Test timestamp: {dataset.test_timestamp}")
    
    db = dataset.get_db(upto_test_timestamp=False)
    print(f"\nDatabase tables: {list(db.table_dict.keys())}")
    for table_name, table in db.table_dict.items():
        print(f"  {table_name}: {len(table.df)} rows")
    
    # Store results for interactive exploration
    results = {}
    
    # Inspect driver-top3 task
    print("\n")
    task1, train1, val1, test1, checks1 = inspect_driver_top3(dataset, db)
    results["driver_top3"] = {
        "task": task1,
        "train": train1,
        "val": val1,
        "test": test1,
        "checks": checks1,
    }
    
    # Inspect driver-race-compete task
    print("\n")
    task2, train2, val2, test2, checks2 = inspect_driver_race_compete(dataset, db)
    results["driver_race_compete"] = {
        "task": task2,
        "train": train2,
        "val": val2,
        "test": test2,
        "checks": checks2,
    }
    
    # Final summary
    print_separator("FINAL SUMMARY")
    
    all_checks = checks1 + checks2
    passed = sum(1 for c in all_checks if c.get("passed"))
    total = len(all_checks)
    
    print(f"Total checks across both tasks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print("\n\033[92mAll validation checks passed for both tasks!\033[0m")
    else:
        print("\n\033[91mSome checks failed. Review details above.\033[0m")
    
    return dataset, db, results


def main_arxiv():
    """Main function to load arxiv dataset and inspect co-citation task."""
    print_separator("LOADING REL-ARXIV DATASET")
    
    print("Loading dataset... (download=True)")
    dataset = get_dataset("rel-arxiv", download=True)
    
    print(f"Dataset: {dataset}")
    print(f"Val timestamp: {dataset.val_timestamp}")
    print(f"Test timestamp: {dataset.test_timestamp}")
    
    db = dataset.get_db(upto_test_timestamp=False)
    print(f"\nDatabase tables: {list(db.table_dict.keys())}")
    for table_name, table in db.table_dict.items():
        print(f"  {table_name}: {len(table.df)} rows")
    
    # Store results for interactive exploration
    results = {}
    
    # Inspect co-citation task
    print("\n")
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
    
    print(f"Total checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print("\n\033[92mAll validation checks passed!\033[0m")
    else:
        print("\n\033[91mSome checks failed. Review details above.\033[0m")
    
    return dataset, db, results


if __name__ == "__main__":
    # Run arxiv inspection by default (change to main() for F1 inspection)
    dataset, db, results = main_arxiv()
    
    # The variables are available for interactive exploration
    print("\n" + "=" * 70)
    print(" Variables available for interactive exploration:")
    print("   - dataset: The ArxivDataset object")
    print("   - db: The database with all entity tables")
    print("   - results['co_citation']: task, train, val, test, checks")
    print("=" * 70)
