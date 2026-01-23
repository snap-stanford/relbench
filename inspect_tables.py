"""
Interactive visualization of F1 tables: histogram of dates.

This script loads the rel-f1 dataset and creates interactive histograms
showing the distribution of records over time for various tables.
"""

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    raise ImportError(
        "Plotly is required for interactive visualization. "
        "Install it with: pip install plotly"
    )

import pandas as pd
from relbench.datasets import get_dataset

# Tables to visualize with their display names
TABLES_TO_PLOT = [
    ("races", "Races"),
    ("standings", "Driver Standings"),
    ("results", "Results"),
    ("constructor_standings", "Constructor Standings"),
    ("constructor_results", "Constructor Results"),
]


def load_f1_data() -> tuple[dict[str, pd.DataFrame], pd.Timestamp, pd.Timestamp]:
    """Load tables from the rel-f1 dataset.
    
    Returns:
        Tuple of (table_dict, val_timestamp, test_timestamp)
        where table_dict maps table names to DataFrames
    """
    print("Loading rel-f1 dataset...")
    dataset = get_dataset("rel-f1", download=True)
    db = dataset.get_db()
    
    table_dict = {}
    for table_name, display_name in TABLES_TO_PLOT:
        df = db.table_dict[table_name].df
        table_dict[table_name] = df
        print(f"Loaded {len(df)} records from {table_name}")
    
    print(f"\nVal timestamp: {dataset.val_timestamp}")
    print(f"Test timestamp: {dataset.test_timestamp}")
    return table_dict, dataset.val_timestamp, dataset.test_timestamp


def create_interactive_plot(
    df: pd.DataFrame,
    table_name: str,
    display_name: str,
    val_timestamp: pd.Timestamp,
    test_timestamp: pd.Timestamp,
) -> go.Figure:
    """Create an interactive histogram of record dates for a table.
    
    Args:
        df: DataFrame with table data (must have 'date' column)
        table_name: Internal table name
        display_name: Human-readable table name for title
        val_timestamp: Validation split timestamp cutoff
        test_timestamp: Test split timestamp cutoff
    """
    fig = px.histogram(
        df,
        x="date",
        nbins=150,
        labels={
            "date": "Date",
            "count": "Number of Records",
        },
        title=f"F1 {display_name}: Distribution Over Time",
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


def main():
    """Main function to load data and display interactive plots."""
    # Load all table data and timestamps
    table_dict, val_timestamp, test_timestamp = load_f1_data()
    
    # Create and show plots for each table
    for table_name, display_name in TABLES_TO_PLOT:
        df = table_dict[table_name]
        
        print(f"\n{display_name} table columns:")
        print(df.columns.tolist())
        print(f"Sample data (first 5 rows, date column):")
        print(df[["date"]].head())
        
        fig = create_interactive_plot(
            df, table_name, display_name, val_timestamp, test_timestamp
        )
        fig.show()
        
        print(f"Plot for {display_name} opened in browser.")
    
    print("\nAll interactive plots opened. Use mouse to zoom, pan, and hover for details.")


if __name__ == "__main__":
    main()
