#!/usr/bin/env python3
"""
compare_global_vs_regional_by_region.py

Compares global vs regional model performance by region×year.
Outputs: metrics CSV, delta CSVs (per year), delta bar charts (per year).
"""

import argparse
import glob
import os
import re
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score

# Set deterministic seed
np.random.seed(42)
plt.rcParams['figure.dpi'] = 200

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# 0. REGION AUTO-DETECTION
# ============================================================================

def discover_regions_from_directory(regional_dir: str, years: List[int]) -> List[int]:
    """
    Auto-discover region IDs from prediction files in directory.

    Scans for files matching pattern: predictions_{year}_{region}.csv
    Extracts unique region identifiers across all years.

    Args:
        regional_dir: Directory containing regional prediction files
        years: List of years to scan for

    Returns:
        Sorted list of unique region IDs (as integers)

    Raises:
        ValueError: If no valid region files found or region IDs are not numeric
    """
    print(f"\nAuto-detecting regions from {regional_dir}...")

    # Pattern to match: predictions_*.csv
    pattern = os.path.join(regional_dir, "predictions_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise ValueError(f"No prediction files found matching pattern: {pattern}")

    # Regex to extract year and region from filename
    # Matches: predictions_2022_0.csv, predictions_2023_15.csv, etc.
    filename_pattern = re.compile(r'predictions_(\d{4})_(\d+)\.csv')

    discovered_regions = set()
    valid_files = []

    for filepath in files:
        filename = os.path.basename(filepath)

        # Skip aggregate files like predictions_2022_ALL_REGIONS.csv
        if 'ALL_REGIONS' in filename.upper():
            continue

        match = filename_pattern.match(filename)
        if match:
            year = int(match.group(1))
            region = int(match.group(2))

            # Only include regions from requested years
            if year in years:
                discovered_regions.add(region)
                valid_files.append(filename)

    if not discovered_regions:
        raise ValueError(
            f"No valid region files found for years {years}. "
            f"Expected pattern: predictions_YYYY_R.csv where YYYY in {years} and R is region number"
        )

    # Convert to sorted list
    regions = sorted(list(discovered_regions))

    print(f"  [OK] Discovered {len(regions)} regions: {regions}")
    print(f"  [OK] Found {len(valid_files)} valid prediction files")

    # Validate region range
    if len(regions) < 1:
        raise ValueError("No regions discovered")
    if len(regions) > 20:
        print(f"  [WARNING] Found {len(regions)} regions (>20). This may impact visualization quality.")

    return regions


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_regional_predictions(regional_dir: str, years: List[int],
                              regions: List[int]) -> pd.DataFrame:
    """
    Load all regional prediction files and parse region/year from filename.

    Expected files: predictions_2022_0.csv, ..., predictions_2024_5.csv

    Returns:
        DataFrame with columns: region, year, area_id, date, overall_phase,
        overall_phase_pred, phase3_test, phase3_pred, ...
    """
    dfs = []

    for year in years:
        for region in regions:
            filename = f"predictions_{year}_{region}.csv"
            filepath = os.path.join(regional_dir, filename)

            if not os.path.exists(filepath):
                print(f"  [WARNING] Missing: {filename}")
                continue

            df = pd.read_csv(filepath)
            df['region'] = region
            df['year'] = year
            dfs.append(df)
            print(f"  [OK] Loaded {filename}: {len(df)} rows")

    if not dfs:
        raise ValueError(f"No regional prediction files found in {regional_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total regional predictions: {len(combined):,} rows")
    return combined


def load_global_predictions(global_dir: str, years: List[int]) -> pd.DataFrame:
    """
    Load all global prediction files and extract year from filename.

    Expected files: forecasting_y_pred_test_2022.csv, 2023, 2024

    Returns:
        DataFrame with same schema as regional (no region column yet)
    """
    dfs = []

    for year in years:
        filename = f"forecasting_y_pred_test_{year}.csv"
        filepath = os.path.join(global_dir, filename)

        if not os.path.exists(filepath):
            print(f"  [WARNING] Missing: {filename}")
            continue

        df = pd.read_csv(filepath)
        df['year'] = year
        dfs.append(df)
        print(f"  [OK] Loaded {filename}: {len(df):,} rows")

    if not dfs:
        raise ValueError(f"No global prediction files found in {global_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total global predictions: {len(combined):,} rows")
    return combined


# ============================================================================
# 2. DATA ENRICHMENT
# ============================================================================

def enrich_global_with_region(global_df: pd.DataFrame,
                              mapping_csv: str) -> pd.DataFrame:
    """
    Merge global predictions with area_id->region mapping.

    Uses validate='m:1' to ensure many-to-one relationship.
    Checks for unmapped rows and errors if >5%.

    Returns:
        global_df with 'region' column added
    """
    print("\nEnriching global predictions with region mapping...")

    # Load mapping
    mapping = pd.read_csv(mapping_csv)
    print(f"  Loaded mapping: {len(mapping):,} area_ids → {mapping['region'].nunique()} regions")

    # Validate mapping columns
    if not {'area_id', 'region'}.issubset(mapping.columns):
        raise ValueError(f"Mapping must have 'area_id' and 'region' columns. Found: {mapping.columns.tolist()}")

    # Check for duplicates in mapping
    if mapping['area_id'].duplicated().any():
        raise ValueError("Mapping has duplicate area_ids")

    # Merge with validation
    rows_before = len(global_df)
    global_df = global_df.merge(mapping[['area_id', 'region']],
                                on='area_id',
                                how='left',
                                validate='m:1')

    # Check rows unchanged
    if len(global_df) != rows_before:
        raise ValueError(f"Merge changed row count: {rows_before} → {len(global_df)}")

    # Check unmapped
    unmapped = global_df['region'].isna()
    unmapped_pct = unmapped.sum() / len(global_df) * 100

    if unmapped.sum() > 0:
        print(f"  [WARNING] Unmapped rows: {unmapped.sum():,} ({unmapped_pct:.2f}%)")
        unmapped_area_ids = global_df[unmapped]['area_id'].unique()[:10]
        print(f"  Sample unmapped area_ids: {unmapped_area_ids.tolist()}")

    if unmapped_pct > 5:
        raise ValueError(f"Too many unmapped rows: {unmapped_pct:.1f}% (threshold: 5%)")

    # Drop unmapped rows
    global_df = global_df[~unmapped].copy()

    print(f"  [OK] Added region column. Rows after merge: {len(global_df):,}")
    return global_df


def filter_global_to_regional_coverage(global_df: pd.DataFrame,
                                      regional_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter global predictions to match regional test set coverage for fair comparison.

    Uses inner merge on (area_id, date, year) to get exact same samples.

    Returns:
        Filtered global_df with same coverage as regional_df
    """
    print("\nFiltering global to regional coverage...")

    # Get unique (area_id, date, year) from regional
    regional_keys = regional_df[['area_id', 'date', 'year']].drop_duplicates()
    print(f"  Regional unique (area_id, date, year): {len(regional_keys):,}")

    # Deduplicate global predictions (may have duplicates from CV/bootstrap)
    rows_before_dedup = len(global_df)
    global_df = global_df.drop_duplicates(subset=['area_id', 'date', 'year'])
    rows_after_dedup = len(global_df)
    if rows_before_dedup > rows_after_dedup:
        print(f"  Deduplicated global: {rows_before_dedup:,} → {rows_after_dedup:,} rows")

    # Inner merge to filter global
    rows_before = len(global_df)
    global_df = global_df.merge(regional_keys,
                                on=['area_id', 'date', 'year'],
                                how='inner',
                                validate='1:1')
    rows_after = len(global_df)

    print(f"  Global rows: {rows_before:,} → {rows_after:,} ({rows_after/rows_before*100:.1f}%)")
    print(f"  [OK] Filtered global to regional coverage")

    return global_df


# ============================================================================
# 3. METRICS COMPUTATION
# ============================================================================

def compute_metrics(df: pd.DataFrame) -> Dict:
    """
    Compute metrics exactly as in run_region_models.py using all_metrics().

    Metrics:
    - accuracy: Overall classification accuracy
    - sensitivity: Recall for phase 3+ (crisis threshold)
    - precision: Precision for phase 3+
    - r2_phase3plus: R² for phase 3 population percentage

    Args:
        df: DataFrame with overall_phase, overall_phase_pred, phase3_test, phase3_pred

    Returns:
        Dict with {accuracy, sensitivity, precision, r2_phase3plus, n_samples}
    """
    y_test = df['overall_phase']
    y_pred = df['overall_phase_pred']

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])

    # 1. Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # 2. Sensitivity (Recall for phase 3+)
    # cm[2:, 2:] = phases 3,4,5 predicted as 3,4,5 (true positives)
    # cm[2:, :] = all actual phase 3,4,5 samples
    correct_3_more = np.sum(cm[2:, 2:])
    total_3_more = np.sum(cm[2:, :])
    sensitivity = correct_3_more / total_3_more if total_3_more > 0 else np.nan

    # 3. Precision (for phase 3+)
    # cm[:, 2:] = all predicted as phase 3,4,5
    total_pred_3_more = np.sum(cm[:, 2:])
    precision = correct_3_more / total_pred_3_more if total_pred_3_more > 0 else np.nan

    # 4. R² for phase 3+ population
    if 'phase3_test' in df.columns and 'phase3_pred' in df.columns:
        # Drop NaN values for R² calculation
        valid = df[['phase3_test', 'phase3_pred']].dropna()
        if len(valid) > 0:
            r2_phase3plus = r2_score(valid['phase3_test'], valid['phase3_pred'])
        else:
            r2_phase3plus = np.nan
    else:
        r2_phase3plus = np.nan

    return {
        'n_samples': len(df),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'precision': float(precision),
        'r2_phase3plus': float(r2_phase3plus)
    }


def compute_group_metrics(df: pd.DataFrame,
                         group_by: List[str] = ['model', 'region', 'year']) -> pd.DataFrame:
    """
    Compute metrics for each (model, region, year) group.

    Returns:
        Tidy DataFrame with columns [model, region, year, n_samples, accuracy,
                                     sensitivity, precision, r2_phase3plus]
    """
    print("\nComputing metrics by", ', '.join(group_by), "...")

    results = []

    for group_vals, group_df in df.groupby(group_by):
        if len(group_df) < 5:
            print(f"  [WARNING] Skipping {group_vals}: only {len(group_df)} samples")
            continue

        metrics = compute_metrics(group_df)

        # Combine group values with metrics
        result = dict(zip(group_by, group_vals))
        result.update(metrics)
        results.append(result)

    metrics_df = pd.DataFrame(results)
    print(f"  [OK] Computed metrics for {len(metrics_df)} groups")

    return metrics_df


# ============================================================================
# 4. DELTA COMPUTATION
# ============================================================================

def make_delta_table(metrics_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Compute regional - global deltas for a specific year.

    Args:
        metrics_df: Metrics for all models, regions, years
        year: Specific year to compute deltas for

    Returns:
        DataFrame [region, metric, delta] for the specified year
    """
    print(f"\nComputing deltas (regional - global) for {year}...")

    # Filter to specific year
    year_metrics = metrics_df[metrics_df['year'] == year].copy()

    # Pivot to have regional and global as separate columns
    pivot = year_metrics.pivot_table(
        index='region',
        columns='model',
        values=['accuracy', 'sensitivity', 'precision', 'r2_phase3plus']
    )

    # Compute deltas for each metric
    deltas = []
    for metric in ['accuracy', 'sensitivity', 'precision', 'r2_phase3plus']:
        if (metric, 'regional') in pivot.columns and (metric, 'global') in pivot.columns:
            delta = pivot[(metric, 'regional')] - pivot[(metric, 'global')]
            delta_df = delta.reset_index()
            delta_df.columns = ['region', 'delta']
            delta_df['metric'] = metric
            deltas.append(delta_df)

    if not deltas:
        raise ValueError(f"No deltas computed for year {year}")

    delta_df = pd.concat(deltas, ignore_index=True)

    print(f"  [OK] Computed {len(delta_df)} deltas for {year}")

    return delta_df


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_grouped_bars(delta_df: pd.DataFrame, out_path: str, year: int,
                     n_regions: int, dpi: int = 200):
    """
    Create grouped bar chart: N regions × 4 metrics for a specific year.

    Layout adapts to number of regions:
    - 1-8 regions: Single row
    - 9-16 regions: Two rows (8 regions per row)
    - 17-20 regions: Three rows (7-8 regions per row)

    Args:
        delta_df: Delta table for the year (region, metric, delta)
        out_path: Output PNG file path
        year: Year being plotted
        n_regions: Total number of regions (for layout calculation)
        dpi: Plot resolution
    """
    # Prepare data for plotting
    plot_df = delta_df.pivot(index='region', columns='metric', values='delta')

    # Calculate layout based on number of regions
    if n_regions <= 8:
        # Single row layout
        n_rows = 1
        figsize = (12, 6)
        regions_per_row = n_regions
    elif n_regions <= 16:
        # Two row layout (8 regions per row)
        n_rows = 2
        figsize = (14, 10)
        regions_per_row = 8
    else:
        # Three row layout (7-8 regions per row for up to 20 regions)
        n_rows = 3
        figsize = (14, 14)
        regions_per_row = 7

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize)

    # Handle single vs multiple subplots
    if n_rows == 1:
        axes = [axes]  # Wrap in list for consistent indexing

    # Split regions across rows if needed
    if n_regions <= 8:
        # Single plot with all regions
        plot_df.plot(kind='bar', ax=axes[0], width=0.8)

        # Styling for single plot
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[0].set_xlabel('Region', fontsize=12)
        axes[0].set_ylabel('Delta (Regional - Global)', fontsize=12)
        axes[0].set_title(f'Regional model − Global model performance by region ({year})',
                         fontsize=14, fontweight='bold')
        axes[0].legend(title='Metric', loc='best')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticklabels([f'Region {int(r)}' for r in plot_df.index],
                               rotation=45 if n_regions > 6 else 0,
                               ha='right' if n_regions > 6 else 'center')
    else:
        # Multi-row layout: split regions across subplots
        regions_list = plot_df.index.tolist()

        for row_idx in range(n_rows):
            # Calculate region slice for this row
            start_idx = row_idx * regions_per_row
            end_idx = min(start_idx + regions_per_row, len(regions_list))

            if start_idx >= len(regions_list):
                # No more regions for this row - hide the subplot
                axes[row_idx].set_visible(False)
                continue

            # Get regions for this row
            row_regions = regions_list[start_idx:end_idx]
            row_df = plot_df.loc[row_regions]

            # Plot this row's data
            row_df.plot(kind='bar', ax=axes[row_idx], width=0.8)

            # Styling
            axes[row_idx].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            axes[row_idx].set_xlabel('Region', fontsize=11)
            axes[row_idx].set_ylabel('Delta (Regional - Global)', fontsize=11)
            axes[row_idx].legend(title='Metric', loc='best', fontsize=9)
            axes[row_idx].grid(axis='y', alpha=0.3)
            axes[row_idx].set_xticklabels([f'Region {int(r)}' for r in row_df.index],
                                         rotation=45, ha='right', fontsize=9)

            # Add row label for clarity
            if row_idx == 0:
                axes[row_idx].set_title(
                    f'Regional model − Global model performance by region ({year})',
                    fontsize=13, fontweight='bold', pad=10
                )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Created visualization with {n_rows} row(s) for {n_regions} regions")


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare global vs regional model performance by region×year'
    )
    parser.add_argument('--regional-dir', required=True, help='Regional predictions directory')
    parser.add_argument('--global-dir', required=True, help='Global predictions directory')
    parser.add_argument('--mapping', required=True, help='area_id->region mapping CSV')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024],
                       help='Years to process')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("GLOBAL VS REGIONAL MODEL COMPARISON")
    print("=" * 80)
    print(f"\nConfig:")
    print(f"  Regional dir: {args.regional_dir}")
    print(f"  Global dir: {args.global_dir}")
    print(f"  Mapping: {args.mapping}")
    print(f"  Output dir: {args.out_dir}")
    print(f"  Years: {args.years}")
    print("=" * 80)

    # 1. Load regional predictions
    print("\n[1/7] Loading regional predictions...")
    # Auto-detect regions from directory
    discovered_regions = discover_regions_from_directory(args.regional_dir, args.years)
    regional_df = load_regional_predictions(args.regional_dir, args.years, discovered_regions)

    # 2. Load global predictions
    print("\n[2/7] Loading global predictions...")
    global_df = load_global_predictions(args.global_dir, args.years)

    # 3. Enrich global with region
    print("\n[3/7] Enriching global with region mapping...")
    global_df = enrich_global_with_region(global_df, args.mapping)

    # 4. Filter global to regional coverage (fair comparison)
    print("\n[4/7] Filtering global to regional coverage...")
    global_df = filter_global_to_regional_coverage(global_df, regional_df)

    # 5. Combine and compute metrics
    print("\n[5/7] Computing metrics...")
    regional_df['model'] = 'regional'
    global_df['model'] = 'global'
    combined_df = pd.concat([regional_df, global_df], ignore_index=True)

    metrics_df = compute_group_metrics(combined_df, group_by=['model', 'region', 'year'])

    # Check coverage (dynamically calculated)
    n_regions = len(discovered_regions)
    n_years = len(args.years)
    n_models = 2  # global and regional
    expected_cells = n_regions * n_years * n_models
    actual_cells = len(metrics_df)
    coverage = actual_cells / expected_cells * 100

    print(f"\n  Coverage: {actual_cells}/{expected_cells} cells ({coverage:.1f}%)")
    print(f"  Expected: {n_regions} regions × {n_years} years × {n_models} models")
    if coverage < 90:
        missing_cells = expected_cells - actual_cells
        print(f"  [WARNING] Low coverage: {missing_cells} cells missing")

    # Save Product A
    out_metrics = os.path.join(args.out_dir, 'region_year_metrics_both_models.csv')
    metrics_df.to_csv(out_metrics, index=False)
    print(f"\n  [OK] Saved: {out_metrics}")

    # 6 & 7. Compute deltas and plot for EACH YEAR
    print("\n[6/7] Computing deltas and creating visualizations per year...")

    for year in args.years:
        # Compute delta for this year
        delta_df = make_delta_table(metrics_df, year)

        # Save delta table for this year
        out_delta = os.path.join(args.out_dir, f'delta_by_region_metric_{year}.csv')
        delta_df.to_csv(out_delta, index=False)
        print(f"  [OK] Saved: {out_delta}")

        # Plot for this year
        out_plot = os.path.join(args.out_dir, f'regional_minus_global_delta_by_region_{year}.png')
        plot_grouped_bars(delta_df, out_plot, year, n_regions=len(discovered_regions), dpi=200)
        print(f"  [OK] Saved: {out_plot}")

    # Print summary PER YEAR
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for year in args.years:
        print(f"\n--- Year {year} ---")

        # Load delta table for this year
        delta_path = os.path.join(args.out_dir, f'delta_by_region_metric_{year}.csv')
        delta_df = pd.read_csv(delta_path)

        # Mean delta per metric for this year
        for metric in ['accuracy', 'sensitivity', 'precision', 'r2_phase3plus']:
            metric_deltas = delta_df[delta_df['metric'] == metric]
            if len(metric_deltas) > 0:
                mean_delta = metric_deltas['delta'].mean()
                print(f"  Mean {metric:15s} delta: {mean_delta:+.4f}")

        # Best/worst regions by precision for this year
        precision_deltas = delta_df[delta_df['metric'] == 'precision'].sort_values('delta')
        if len(precision_deltas) > 0:
            best_region = precision_deltas.iloc[-1]['region']
            best_delta = precision_deltas.iloc[-1]['delta']
            worst_region = precision_deltas.iloc[0]['region']
            worst_delta = precision_deltas.iloc[0]['delta']

            print(f"  Best region (precision):  Region {int(best_region)} ({best_delta:+.4f})")
            print(f"  Worst region (precision): Region {int(worst_region)} ({worst_delta:+.4f})")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
