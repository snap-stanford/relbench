"""
Interactive visualization of dataset tables: histogram of timestamps.

This script loads a relbench dataset and creates interactive histograms
showing the distribution of records over time for specified tables.

Usage:
    python inspect_tables.py                      # Default: rel-f1 with default tables
    python inspect_tables.py rel-arxiv papers     # Single table from rel-arxiv
    python inspect_tables.py rel-f1 races results # Multiple tables from rel-f1
"""

import argparse

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    raise ImportError(
        "Plotly is required for interactive visualization. "
        "Install it with: pip install plotly"
    )

import pandas as pd
from relbench.base import Table
from relbench.datasets import get_dataset

# Default tables to visualize per dataset (table_name, display_name)
DEFAULT_TABLES = {
    "rel-f1": [
        ("races", "Races"),
        ("standings", "Driver Standings"),
        ("results", "Results"),
        ("constructor_standings", "Constructor Standings"),
        ("constructor_results", "Constructor Results"),
    ],
    "rel-arxiv": [
        ("papers", "Papers"),
        ("citations", "Citations"),
        ("paperAuthors", "Paper Authors"),
    ],
}


def load_dataset_tables(
    dataset_name: str,
    table_names: list[str] | None = None,
) -> tuple[dict[str, tuple[pd.DataFrame, str]], pd.Timestamp, pd.Timestamp, str]:
    """Load tables from a relbench dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'rel-f1', 'rel-arxiv')
        table_names: List of table names to load. If None, uses defaults for the dataset.
    
    Returns:
        Tuple of (table_data, val_timestamp, test_timestamp, dataset_name)
        where table_data maps table_name to (DataFrame, time_col_name)
    """
    print(f"Loading {dataset_name} dataset...")
    dataset = get_dataset(dataset_name, download=True)
    db = dataset.get_db()
    
    # Determine which tables to load
    if table_names is None:
        if dataset_name in DEFAULT_TABLES:
            tables_to_load = [(name, display) for name, display in DEFAULT_TABLES[dataset_name]]
        else:
            # Load all tables with a time column
            tables_to_load = [
                (name, name.replace("_", " ").title())
                for name, table in db.table_dict.items()
                if table.time_col is not None
            ]
    else:
        tables_to_load = [(name, name.replace("_", " ").title()) for name in table_names]
    
    table_data = {}
    for table_name, display_name in tables_to_load:
        if table_name not in db.table_dict:
            print(f"Warning: Table '{table_name}' not found in {dataset_name}. Skipping.")
            continue
        
        table: Table = db.table_dict[table_name]
        if table.time_col is None:
            print(f"Warning: Table '{table_name}' has no time column. Skipping.")
            continue
        
        df = table.df
        time_col = table.time_col
        table_data[table_name] = (df, time_col, display_name)
        print(f"Loaded {len(df)} records from {table_name} (time_col: {time_col})")
    
    print(f"\nVal timestamp: {dataset.val_timestamp}")
    print(f"Test timestamp: {dataset.test_timestamp}")
    return table_data, dataset.val_timestamp, dataset.test_timestamp, dataset_name


def create_interactive_plot(
    df: pd.DataFrame,
    table_name: str,
    display_name: str,
    time_col: str,
    dataset_name: str,
    val_timestamp: pd.Timestamp,
    test_timestamp: pd.Timestamp,
) -> go.Figure:
    """Create an interactive histogram of record timestamps for a table.
    
    Args:
        df: DataFrame with table data
        table_name: Internal table name
        display_name: Human-readable table name for title
        time_col: Name of the timestamp column in the DataFrame
        dataset_name: Name of the dataset for the title
        val_timestamp: Validation split timestamp cutoff
        test_timestamp: Test split timestamp cutoff
    """
    # Format dataset name for display (e.g., 'rel-f1' -> 'F1')
    dataset_display = dataset_name.replace("rel-", "").upper()
    
    fig = px.histogram(
        df,
        x=time_col,
        nbins=150,
        labels={
            time_col: "Date",
            "count": "Number of Records",
        },
        title=f"{dataset_display} {display_name}: Distribution Over Time",
    )
    
    # Add vertical lines for train/val/test split cutoffs
    # Use add_shape + add_annotation separately to avoid plotly datetime annotation bugs
    fig.add_shape(
        type="line",
        x0=val_timestamp,
        x1=val_timestamp,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dash", color="orange", width=2),
    )
    fig.add_annotation(
        x=val_timestamp,
        y=1,
        yref="paper",
        text="Val cutoff",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="orange"),
    )
    
    fig.add_shape(
        type="line",
        x0=test_timestamp,
        x1=test_timestamp,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dash", color="red", width=2),
    )
    fig.add_annotation(
        x=test_timestamp,
        y=1,
        yref="paper",
        text="Test cutoff",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="red"),
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"Number of {display_name}",
        hovermode="closest",
        template="plotly_white",
        bargap=0.1,
    )
    
    return fig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize timestamp distributions for relbench dataset tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inspect_tables.py                      # Default: rel-f1 with default tables
    python inspect_tables.py rel-arxiv papers     # Single table from rel-arxiv
    python inspect_tables.py rel-f1 races results # Multiple tables from rel-f1
    python inspect_tables.py rel-arxiv            # All default tables from rel-arxiv
        """,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="rel-f1",
        help="Dataset name (e.g., 'rel-f1', 'rel-arxiv'). Default: rel-f1",
    )
    parser.add_argument(
        "tables",
        nargs="*",
        help="Table names to visualize. If not specified, uses dataset defaults.",
    )
    return parser.parse_args()


def main():
    """Main function to load data and display interactive plots."""
    args = parse_args()
    
    dataset_name = args.dataset
    table_names = args.tables if args.tables else None
    
    # Load all table data and timestamps
    table_data, val_timestamp, test_timestamp, dataset_name = load_dataset_tables(
        dataset_name, table_names
    )
    
    if not table_data:
        print("No tables with time columns found to visualize.")
        return
    
    # Create and show plots for each table
    for table_name, (df, time_col, display_name) in table_data.items():
        print(f"\n{display_name} table columns:")
        print(df.columns.tolist())
        print(f"Sample data (first 5 rows, {time_col} column):")
        print(df[[time_col]].head())
        
        fig = create_interactive_plot(
            df, table_name, display_name, time_col, dataset_name,
            val_timestamp, test_timestamp
        )
        fig.show()
        
        print(f"Plot for {display_name} opened in browser.")
    
    print("\nAll interactive plots opened. Use mouse to zoom, pan, and hover for details.")


if __name__ == "__main__":
    main()
