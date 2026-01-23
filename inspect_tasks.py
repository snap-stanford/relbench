"""
Interactive inspection of rel-f1 tasks: driver-top3 and driver-race-compete.

This script:
1. Initializes both tasks from the relbench package
2. Displays the heads of train/val/test tables for each
3. Performs relationship validation checks to ensure data integrity

Tasks inspected:
- driver-top3: EntityTask (binary classification) - predicts if driver qualifies top-3
- driver-race-compete: RecommendationTask (link prediction) - predicts which races a driver competes in
"""

import pandas as pd
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


def inspect_driver_top3(dataset, db):
    """Inspect the driver-top3 EntityTask."""
    print_separator("DRIVER-TOP3 TASK (EntityTask - Binary Classification)")
    
    print("\nLoading task...")
    task = get_task("rel-f1", "driver-top3", download=True)
    
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
    
    return task, train_table, val_table, test_table, all_checks


def inspect_driver_race_compete(dataset, db):
    """Inspect the driver-race-compete RecommendationTask."""
    print_separator("DRIVER-RACE-COMPETE TASK (RecommendationTask - Link Prediction)")
    
    print("\nLoading task...")
    task = get_task("rel-f1", "driver-race-compete", download=True)
    
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
        print(f"  Avg races per driver-timestamp: {total_races / len(df):.2f}")
    
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


def main():
    """Main function to load dataset and inspect both tasks."""
    print_separator("LOADING REL-F1 DATASET")
    
    print("Loading dataset...")
    dataset = get_dataset("rel-f1", download=True)
    
    print(f"Dataset: {dataset}")
    print(f"Val timestamp: {dataset.val_timestamp}")
    print(f"Test timestamp: {dataset.test_timestamp}")
    
    db = dataset.get_db()
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


if __name__ == "__main__":
    dataset, db, results = main()
    
    # The variables are available for interactive exploration
    print("\n" + "=" * 70)
    print(" Variables available for interactive exploration:")
    print("   - dataset: The F1Dataset object")
    print("   - db: The database with all entity tables")
    print("   - results['driver_top3']: task, train, val, test, checks")
    print("   - results['driver_race_compete']: task, train, val, test, checks")
    print("=" * 70)
