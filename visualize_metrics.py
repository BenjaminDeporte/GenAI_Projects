#!/usr/bin/env python3
"""
Visualize Trackio metrics as line plots using matplotlib and plotly.
Extracts data from the Trackio SQLite database and creates interactive visualizations.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def get_trackio_db_path() -> Path:
    """Get the path to the Trackio database."""
    db_path = Path.home() / "GenAI_Projects/logs/trackio/medical-sft-reasoning.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Trackio DB not found at {db_path}")
    return db_path


def load_metrics_from_db(run_name: str | None = None) -> Dict[str, pd.DataFrame]:
    """
    Load metrics from Trackio database.
    
    Args:
        run_name: Optional specific run name. If None, loads all runs.
        
    Returns:
        Dictionary mapping run names to DataFrames with metrics.
    """
    db_path = get_trackio_db_path()
    conn = sqlite3.connect(str(db_path))
    
    # Get all runs if not specified
    if run_name is None:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT run_name FROM metrics ORDER BY run_name")
        runs = [row[0] for row in cursor.fetchall()]
    else:
        runs = [run_name]
    
    results = {}
    
    for run in runs:
        # Query metrics for this run
        cursor = conn.cursor()
        cursor.execute(
            "SELECT step, metrics FROM metrics WHERE run_name = ? ORDER BY step",
            (run,)
        )
        rows = cursor.fetchall()
        
        # Parse JSON metrics into dataframe
        data = {
            'step': [],
            'epoch': []
        }
        metric_names = set()
        
        for step, metrics_json in rows:
            metrics_dict = json.loads(metrics_json)
            data['step'].append(step)
            data['epoch'].append(metrics_dict.pop('epoch', None))
            
            # Collect all metric names
            metric_names.update(metrics_dict.keys())
            
            # Add metric values
            for metric_name in metric_names:
                if metric_name not in data:
                    data[metric_name] = [None] * (len(data['step']) - 1)
                data[metric_name].append(metrics_dict.get(metric_name))
        
        # Fill any missing values for earlier entries
        for metric_name in metric_names:
            if metric_name not in data:
                data[metric_name] = [None] * len(data['step'])
        
        results[run] = pd.DataFrame(data)
    
    conn.close()
    return results


def plot_metrics_matplotlib(metrics_dict: Dict[str, pd.DataFrame], output_dir: Path | None = None):
    """
    Create line plots using matplotlib.
    
    Args:
        metrics_dict: Dictionary mapping run names to DataFrames.
        output_dir: Optional directory to save plots. If None, displays plots.
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all metric names (excluding step and epoch)
    all_metrics = set()
    for df in metrics_dict.values():
        all_metrics.update(col for col in df.columns if col not in ['step', 'epoch'])
    
    # Create subplots for each metric
    n_metrics = len(all_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(sorted(all_metrics)):
        ax = axes[idx]
        
        for run_name, df in metrics_dict.items():
            if metric in df.columns:
                # Use epoch for x-axis if available, otherwise use step
                x = df['epoch'].dropna() if df['epoch'].notna().any() else df['step']
                y = df[metric]
                
                ax.plot(x, y, marker='o', linewidth=2, label=run_name)
        
        ax.set_xlabel('Epoch' if df['epoch'].notna().any() else 'Step')
        ax.set_ylabel(metric)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(len(all_metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / "metrics_plot.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Matplotlib plot saved to: {output_path}")
    else:
        plt.show()


def plot_metrics_plotly(metrics_dict: Dict[str, pd.DataFrame], output_dir: Path | None = None):
    """
    Create interactive line plots using Plotly.
    
    Args:
        metrics_dict: Dictionary mapping run names to DataFrames.
        output_dir: Optional directory to save plots. If None, displays plots.
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all metric names (excluding step and epoch)
    all_metrics = set()
    for df in metrics_dict.values():
        all_metrics.update(col for col in df.columns if col not in ['step', 'epoch'])
    
    all_metrics = sorted(all_metrics)
    n_metrics = len(all_metrics)
    
    # Create subplots
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=all_metrics,
        specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
    )
    
    row = 1
    col = 1
    
    for metric in all_metrics:
        for run_name, df in metrics_dict.items():
            if metric in df.columns:
                # Use epoch for x-axis if available, otherwise use step
                x = df['epoch'] if df['epoch'].notna().any() else df['step']
                y = df[metric]
                
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode='lines+markers', name=run_name, 
                              legendgroup=run_name, showlegend=(col == 1 and row == 1)),
                    row=row, col=col
                )
        
        # Update axes labels
        fig.update_xaxes(title_text="Epoch" if df['epoch'].notna().any() else "Step", row=row, col=col)
        fig.update_yaxes(title_text=metric, row=row, col=col)
        
        col += 1
        if col > n_cols:
            col = 1
            row += 1
    
    fig.update_layout(
        height=300 * n_rows,
        title_text="Training Metrics - Line Plots",
        showlegend=True,
        hovermode='x unified'
    )
    
    if output_dir:
        output_path = output_dir / "metrics_plot_interactive.html"
        fig.write_html(str(output_path))
        print(f"Plotly interactive plot saved to: {output_path}")
        print(f"Open in browser: file://{output_path.resolve()}")
    else:
        fig.show()


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("Loading metrics from Trackio database...")
    metrics_dict = load_metrics_from_db(run_name)
    
    if not metrics_dict:
        print("No metrics found!")
        sys.exit(1)
    
    print(f"Loaded data for runs: {list(metrics_dict.keys())}")
    for run_name, df in metrics_dict.items():
        print(f"  {run_name}: {len(df)} entries")
    
    # Create output directory
    output_dir = Path.home() / "GenAI_Projects/metrics_plots"
    
    print("\nGenerating matplotlib plots...")
    plot_metrics_matplotlib(metrics_dict, output_dir)
    
    print("Generating interactive Plotly plots...")
    plot_metrics_plotly(metrics_dict, output_dir)
    
    print(f"\nPlots saved to: {output_dir}")
