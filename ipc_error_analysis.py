#!/usr/bin/env python3
"""
IPC Crisis Prediction Error Analysis

This script performs geospatial analysis to evaluate crisis prediction errors by:
1. Loading IPC/CH polygon geometries from GeoJSON
2. Loading forecasting predictions from CSV files (2022-2024)
3. Joining prediction points to polygons using haversine nearest-neighbor matching
4. Computing error metrics (TP, TN, FP, FN) for crisis classification
5. Visualizing error rates in a 3-panel map
6. Saving detailed outputs for further analysis

Author: Claude Code
Date: 2025-12-12
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths
GEOJSON_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\gdf_ipc_ch_final.geojson"
PREDICTION_DIR = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\IPCCH_model\forecasting_prediction"

# Output paths
OUTPUT_DIR = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\IPCCH_model\outputs"
POINT_LEVEL_DIR = Path(OUTPUT_DIR) / "point_level"
POLYGON_LEVEL_DIR = Path(OUTPUT_DIR) / "polygon_level"
FIGURES_DIR = Path(OUTPUT_DIR) / "figures"

# Analysis parameters
YEARS = [2022, 2023, 2024]
MAX_DISTANCE_KM = 50.0  # Threshold for flagging poor matches
CRISIS_THRESHOLD = 3  # Phase >= 3 is considered crisis

# Visualization parameters
COLORMAP = 'YlOrRd'
FIGURE_SIZE = (18, 6)
DPI = 300


# ============================================================================
# STEP 1: LOAD AND PREPARE IPC POLYGONS
# ============================================================================

def load_ipc_polygons(geojson_path):
    """
    Load IPC/CH polygons from GeoJSON and compute centroids.

    Parameters:
    -----------
    geojson_path : str
        Path to GeoJSON file

    Returns:
    --------
    gpd.GeoDataFrame
        Polygons with centroid coordinates and formatted strings
    """
    print(f"\n{'='*80}")
    print("STEP 1: LOADING IPC POLYGONS")
    print(f"{'='*80}")

    # Load GeoJSON
    print(f"Loading: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    print(f"  Total features loaded: {len(gdf)}")
    print(f"  Geometry types: {gdf.geometry.type.value_counts().to_dict()}")

    # Filter to Polygon/MultiPolygon only
    print(f"\nFiltering to Polygon/MultiPolygon geometries...")
    polygon_types = ['Polygon', 'MultiPolygon']
    gdf = gdf[gdf.geometry.type.isin(polygon_types)].copy()
    print(f"  Polygons after filtering: {len(gdf)}")

    # Check and ensure EPSG:4326
    print(f"\nChecking CRS...")
    print(f"  Original CRS: {gdf.crs}")

    if gdf.crs is None:
        print(f"  WARNING: CRS not found, assuming EPSG:4326")
        gdf.crs = "EPSG:4326"
    elif gdf.crs.to_string() != "EPSG:4326":
        print(f"  Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
    else:
        print(f"  CRS is already EPSG:4326")

    # Compute centroids
    print(f"\nComputing polygon centroids...")
    gdf['centroid_lat'] = gdf.geometry.centroid.y
    gdf['centroid_lon'] = gdf.geometry.centroid.x

    # Format coordinates to 12 decimals for exact matching
    gdf['centroid_lat_str'] = gdf['centroid_lat'].apply(lambda x: f"{x:.12f}")
    gdf['centroid_lon_str'] = gdf['centroid_lon'].apply(lambda x: f"{x:.12f}")

    # Rename 'id' column to 'polygon_id' for clarity
    if 'id' in gdf.columns:
        gdf = gdf.rename(columns={'id': 'polygon_id'})
    else:
        # Create polygon_id from index if not present
        gdf['polygon_id'] = gdf.index.astype(str)

    # QA checks
    print(f"\nQA Checks:")
    print(f"  Coordinate bounds: {gdf.total_bounds}")
    print(f"  Null geometries: {gdf.geometry.isna().sum()}")
    print(f"  Sample centroids:")
    print(gdf[['polygon_id', 'centroid_lat', 'centroid_lon']].head(3).to_string())

    return gdf


# ============================================================================
# STEP 2: LOAD AND COMBINE PREDICTION DATA
# ============================================================================

def load_predictions(prediction_dir, years):
    """
    Load prediction CSV files for specified years.

    Parameters:
    -----------
    prediction_dir : str
        Directory containing forecasting_y_pred_test_YYYY.csv files
    years : list
        List of years to load

    Returns:
    --------
    gpd.GeoDataFrame
        Combined predictions with point geometries
    """
    print(f"\n{'='*80}")
    print("STEP 2: LOADING PREDICTION DATA")
    print(f"{'='*80}")

    prediction_dir = Path(prediction_dir)
    all_predictions = []

    for year in years:
        filename = f"forecasting_y_pred_test_{year}.csv"
        filepath = prediction_dir / filename

        print(f"\nLoading {filename}...")

        if not filepath.exists():
            print(f"  ERROR: File not found: {filepath}")
            continue

        # Load CSV
        df = pd.read_csv(filepath)
        print(f"  Records loaded: {len(df)}")

        # Add year column
        df['year'] = year

        # Parse date
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        # Validate required columns
        required_cols = ['lat', 'lon', 'overall_phase', 'overall_phase_pred', 'date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {filename}: {missing_cols}")

        # Create formatted coordinate strings (12 decimals)
        df['lat_str'] = df['lat'].apply(lambda x: f"{x:.12f}")
        df['lon_str'] = df['lon'].apply(lambda x: f"{x:.12f}")

        # Create Point geometries
        df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

        # QA checks
        print(f"  QA Checks:")
        print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"    Unique dates: {df['date'].nunique()}")
        print(f"    Lat range: [{df['lat'].min():.6f}, {df['lat'].max():.6f}]")
        print(f"    Lon range: [{df['lon'].min():.6f}, {df['lon'].max():.6f}]")
        print(f"    overall_phase range: [{df['overall_phase'].min()}, {df['overall_phase'].max()}]")
        print(f"    overall_phase_pred range: [{df['overall_phase_pred'].min()}, {df['overall_phase_pred'].max()}]")
        print(f"    Missing values: {df[required_cols].isna().sum().sum()}")

        all_predictions.append(df)

    # Combine all years
    print(f"\n{'='*80}")
    print("COMBINING ALL YEARS")
    print(f"{'='*80}")

    combined_df = pd.concat(all_predictions, ignore_index=True)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(combined_df, geometry='geometry', crs="EPSG:4326")

    print(f"Total records: {len(gdf)}")
    print(f"Years: {sorted(gdf['year'].unique())}")
    print(f"Records per year:")
    for year in sorted(gdf['year'].unique()):
        print(f"  {year}: {len(gdf[gdf['year'] == year])}")

    return gdf


# ============================================================================
# STEP 3: SPATIAL JOIN - POINT TO POLYGON
# ============================================================================

def join_points_to_polygons(points_gdf, polygons_gdf, max_distance_km=50.0):
    """
    Join prediction points to IPC polygons using dual-method approach.

    Method A: Exact coordinate match (12-digit precision)
    Method B: Haversine nearest-neighbor (for unmatched points)

    Parameters:
    -----------
    points_gdf : gpd.GeoDataFrame
        Prediction points
    polygons_gdf : gpd.GeoDataFrame
        IPC polygons with centroids
    max_distance_km : float
        Maximum distance threshold for flagging poor matches

    Returns:
    --------
    gpd.GeoDataFrame
        Points with polygon assignments and match quality
    """
    print(f"\n{'='*80}")
    print("STEP 3: SPATIAL JOIN - POINT TO POLYGON")
    print(f"{'='*80}")
    print(f"Max distance threshold: {max_distance_km} km")

    # Initialize result columns
    points_gdf = points_gdf.copy()
    points_gdf['polygon_id'] = None
    points_gdf['join_method'] = None
    points_gdf['distance_km'] = np.nan
    points_gdf['match_quality'] = 'unmatched'

    # ========================================================================
    # METHOD A: EXACT COORDINATE MATCH
    # ========================================================================
    print(f"\nMETHOD A: Exact coordinate matching (12-digit precision)...")

    # Create merge keys
    points_gdf['merge_key'] = points_gdf['lat_str'] + '_' + points_gdf['lon_str']
    polygons_gdf['merge_key'] = polygons_gdf['centroid_lat_str'] + '_' + polygons_gdf['centroid_lon_str']

    # Perform merge
    merged = points_gdf.merge(
        polygons_gdf[['merge_key', 'polygon_id']],
        on='merge_key',
        how='left',
        suffixes=('', '_poly')
    )

    # Update matched records
    exact_mask = merged['polygon_id_poly'].notna()
    points_gdf.loc[exact_mask, 'polygon_id'] = merged.loc[exact_mask, 'polygon_id_poly']
    points_gdf.loc[exact_mask, 'join_method'] = 'exact'
    points_gdf.loc[exact_mask, 'distance_km'] = 0.0
    points_gdf.loc[exact_mask, 'match_quality'] = 'excellent'

    exact_count = exact_mask.sum()
    print(f"  Exact matches: {exact_count:,} ({exact_count/len(points_gdf)*100:.2f}%)")

    # Clean up temporary column
    points_gdf = points_gdf.drop(columns=['merge_key'])
    polygons_gdf = polygons_gdf.drop(columns=['merge_key'])

    # ========================================================================
    # METHOD B: HAVERSINE NEAREST-NEIGHBOR (for unmatched points)
    # ========================================================================
    unmatched_mask = points_gdf['join_method'].isna()
    unmatched_count = unmatched_mask.sum()

    if unmatched_count > 0:
        print(f"\nMETHOD B: Haversine nearest-neighbor matching...")
        print(f"  Unmatched points: {unmatched_count:,}")

        # Extract unmatched points
        unmatched_points = points_gdf[unmatched_mask].copy()

        # Convert coordinates to radians
        points_coords_rad = np.radians(unmatched_points[['lat', 'lon']].values)
        polygon_coords_rad = np.radians(polygons_gdf[['centroid_lat', 'centroid_lon']].values)

        # Fit NearestNeighbors model
        print(f"  Fitting NearestNeighbors model on {len(polygons_gdf)} polygon centroids...")
        nbrs = NearestNeighbors(n_neighbors=1, metric='haversine', algorithm='ball_tree')
        nbrs.fit(polygon_coords_rad)

        # Find nearest neighbors
        print(f"  Finding nearest polygon for each point...")
        distances, indices = nbrs.kneighbors(points_coords_rad)

        # Convert distances to kilometers
        distances_km = distances.flatten() * 6371  # Earth radius in km
        nearest_polygon_indices = indices.flatten()

        # Update points_gdf for unmatched records
        unmatched_indices = points_gdf[unmatched_mask].index
        points_gdf.loc[unmatched_indices, 'polygon_id'] = polygons_gdf.iloc[nearest_polygon_indices]['polygon_id'].values
        points_gdf.loc[unmatched_indices, 'join_method'] = 'haversine'
        points_gdf.loc[unmatched_indices, 'distance_km'] = distances_km

        # Classify match quality for haversine matches
        haversine_mask = points_gdf['join_method'] == 'haversine'
        good_mask = haversine_mask & (points_gdf['distance_km'] <= 10)
        poor_mask = haversine_mask & (points_gdf['distance_km'] > 10) & (points_gdf['distance_km'] <= max_distance_km)
        bad_mask = haversine_mask & (points_gdf['distance_km'] > max_distance_km)

        points_gdf.loc[good_mask, 'match_quality'] = 'good'
        points_gdf.loc[poor_mask, 'match_quality'] = 'poor'
        points_gdf.loc[bad_mask, 'match_quality'] = 'unmatched'

        print(f"  Haversine matches: {(good_mask | poor_mask).sum():,}")
        print(f"  Distance statistics (km):")
        print(f"    Min:    {distances_km.min():.3f}")
        print(f"    Median: {np.median(distances_km):.3f}")
        print(f"    Max:    {distances_km.max():.3f}")

    # ========================================================================
    # MATCH QUALITY SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("MATCH QUALITY SUMMARY")
    print(f"{'='*80}")

    quality_counts = points_gdf['match_quality'].value_counts()
    for quality in ['excellent', 'good', 'poor', 'unmatched']:
        count = quality_counts.get(quality, 0)
        pct = count / len(points_gdf) * 100
        print(f"  {quality:12s}: {count:8,} ({pct:5.2f}%)")

    method_counts = points_gdf['join_method'].value_counts()
    print(f"\nJoin method distribution:")
    for method, count in method_counts.items():
        pct = count / len(points_gdf) * 100
        print(f"  {method:12s}: {count:8,} ({pct:5.2f}%)")

    # Flag problematic matches
    flagged = points_gdf[points_gdf['match_quality'] == 'unmatched']
    if len(flagged) > 0:
        print(f"\nWARNING: {len(flagged):,} points flagged as 'unmatched' (distance > {max_distance_km} km)")

    return points_gdf


# ============================================================================
# STEP 4: COMPUTE ERROR METRICS
# ============================================================================

def compute_error_metrics(joined_gdf):
    """
    Compute point-level crisis classification error metrics.

    Parameters:
    -----------
    joined_gdf : gpd.GeoDataFrame
        Points joined to polygons

    Returns:
    --------
    pd.DataFrame
        Points with error classifications
    """
    print(f"\n{'='*80}")
    print("STEP 4: COMPUTING ERROR METRICS")
    print(f"{'='*80}")

    df = joined_gdf.copy()

    # Compute crisis flags
    df['actual_crisis'] = (df['overall_phase'] >= CRISIS_THRESHOLD).astype(int)
    df['predicted_crisis'] = (df['overall_phase_pred'] >= CRISIS_THRESHOLD).astype(int)

    # Compute error categories
    conditions = [
        (df['actual_crisis'] == 1) & (df['predicted_crisis'] == 1),  # TP
        (df['actual_crisis'] == 0) & (df['predicted_crisis'] == 0),  # TN
        (df['actual_crisis'] == 0) & (df['predicted_crisis'] == 1),  # FP
        (df['actual_crisis'] == 1) & (df['predicted_crisis'] == 0),  # FN
    ]
    choices = ['TP', 'TN', 'FP', 'FN']
    df['error_category'] = np.select(conditions, choices, default='UNKNOWN')

    # Compute error flag (1 for FP or FN, 0 for TP or TN)
    df['error_flag'] = df['error_category'].isin(['FP', 'FN']).astype(int)

    # Print metrics per year
    print(f"\nError Metrics by Year:")
    print(f"{'='*80}")

    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]

        print(f"\n{year}:")
        print(f"-" * 40)

        # Confusion matrix
        confusion = pd.crosstab(
            year_df['actual_crisis'],
            year_df['predicted_crisis'],
            rownames=['Actual'],
            colnames=['Predicted'],
            margins=True
        )
        print(f"\nConfusion Matrix:")
        print(confusion)

        # Compute metrics
        tp = (year_df['error_category'] == 'TP').sum()
        tn = (year_df['error_category'] == 'TN').sum()
        fp = (year_df['error_category'] == 'FP').sum()
        fn = (year_df['error_category'] == 'FN').sum()

        accuracy = (tp + tn) / len(year_df) if len(year_df) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

    return df


# ============================================================================
# STEP 5: AGGREGATE TO POLYGON LEVEL
# ============================================================================

def aggregate_to_polygons(point_level_df):
    """
    Aggregate point-level errors to polygon level.

    Parameters:
    -----------
    point_level_df : pd.DataFrame
        Points with error metrics

    Returns:
    --------
    pd.DataFrame
        Polygon-level aggregation
    """
    print(f"\n{'='*80}")
    print("STEP 5: AGGREGATING TO POLYGON LEVEL")
    print(f"{'='*80}")

    # Group by polygon_id and year
    agg_df = point_level_df.groupby(['polygon_id', 'year']).agg(
        n=('error_flag', 'count'),
        TP_count=('error_category', lambda x: (x == 'TP').sum()),
        TN_count=('error_category', lambda x: (x == 'TN').sum()),
        FP_count=('error_category', lambda x: (x == 'FP').sum()),
        FN_count=('error_category', lambda x: (x == 'FN').sum()),
        error_rate=('error_flag', 'mean'),
    ).reset_index()

    # Compute accuracy
    agg_df['accuracy'] = (agg_df['TP_count'] + agg_df['TN_count']) / agg_df['n']

    # Print summary per year
    print(f"\nPolygon-level Summary by Year:")
    print(f"{'='*80}")

    for year in sorted(agg_df['year'].unique()):
        year_df = agg_df[agg_df['year'] == year]

        print(f"\n{year}:")
        print(f"  Polygons with predictions: {len(year_df):,}")
        print(f"  Error rate statistics:")
        print(f"    Min:    {year_df['error_rate'].min():.4f}")
        print(f"    Median: {year_df['error_rate'].median():.4f}")
        print(f"    Mean:   {year_df['error_rate'].mean():.4f}")
        print(f"    Max:    {year_df['error_rate'].max():.4f}")

        high_error = year_df[year_df['error_rate'] > 0.5]
        if len(high_error) > 0:
            print(f"  WARNING: {len(high_error):,} polygons with error_rate > 0.5")

    return agg_df


# ============================================================================
# STEP 6: VISUALIZATION - 3-PANEL ERROR MAP
# ============================================================================

def plot_error_maps(polygon_gdf, polygon_summary, years, output_path, cmap='YlOrRd', figsize=(18, 6), dpi=300):
    """
    Create 1x3 subplot figure showing error rates per year.

    Parameters:
    -----------
    polygon_gdf : gpd.GeoDataFrame
        IPC polygons with geometry
    polygon_summary : pd.DataFrame
        Polygon-level error metrics
    years : list
        Years to plot
    output_path : str
        Path to save figure
    cmap : str
        Matplotlib colormap
    figsize : tuple
        Figure size
    dpi : int
        Output resolution
    """
    print(f"\n{'='*80}")
    print("STEP 6: CREATING ERROR MAPS")
    print(f"{'='*80}")

    # Compute global vmin/vmax for consistent color scale
    vmin = 0
    vmax = polygon_summary['error_rate'].max()
    print(f"Color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot each year
    for i, year in enumerate(years):
        ax = axes[i]

        print(f"\nPlotting {year}...")

        # Filter data for current year
        year_data = polygon_summary[polygon_summary['year'] == year]

        # Merge with polygon geometries
        plot_gdf = polygon_gdf.merge(
            year_data,
            on='polygon_id',
            how='left'
        )

        # Reproject to Web Mercator for basemap
        if plot_gdf.crs.to_string() != "EPSG:3857":
            plot_gdf = plot_gdf.to_crs("EPSG:3857")

        # Plot ONLY polygons with data (do not plot polygons without data to avoid confusion)
        has_data = plot_gdf['error_rate'].notna()
        if has_data.sum() > 0:
            plot_gdf[has_data].plot(
                ax=ax,
                column='error_rate',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolor='black',
                linewidth=0.3,
                legend=False
            )
            print(f"  Polygons with data: {has_data.sum():,}")

        # Count but do NOT plot polygons without data
        no_data = ~has_data
        if no_data.sum() > 0:
            print(f"  Polygons without data (not displayed): {no_data.sum():,}")

        # Add basemap
        try:
            ctx.add_basemap(
                ax,
                crs=plot_gdf.crs.to_string(),
                source=ctx.providers.CartoDB.Positron
            )
        except Exception as e:
            print(f"  Warning: Could not add basemap - {e}")

        # Formatting
        ax.set_title(f'{year}', fontsize=14, fontweight='bold')
        ax.set_axis_off()

    # Overall title - position it lower to reduce blank space
    fig.suptitle(
        'Crisis Prediction Error Rate by Polygon (2022-2024)',
        fontsize=16,
        fontweight='bold',
        y=0.95
    )

    # Adjust layout to leave space for title (top) and colorbar (bottom)
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

    # Add shared colorbar as horizontal bar at the bottom (after tight_layout)
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])

    # Create horizontal colorbar at the bottom with proper positioning
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Error Rate (FP + FN) / Total', fontsize=12)

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")

    plt.close()


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline.
    """
    print(f"\n{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "IPC CRISIS PREDICTION ERROR ANALYSIS" + " "*22 + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}")

    # Create output directories
    POINT_LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    POLYGON_LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load IPC polygons
    polygons_gdf = load_ipc_polygons(GEOJSON_PATH)

    # Step 2: Load predictions
    predictions_gdf = load_predictions(PREDICTION_DIR, YEARS)

    # Step 3: Join points to polygons
    joined_gdf = join_points_to_polygons(
        predictions_gdf,
        polygons_gdf,
        max_distance_km=MAX_DISTANCE_KM
    )

    # Step 4: Compute error metrics
    point_level_df = compute_error_metrics(joined_gdf)

    # Step 5: Save point-level data per year
    print(f"\n{'='*80}")
    print("SAVING POINT-LEVEL DATA")
    print(f"{'='*80}")

    for year in YEARS:
        year_df = point_level_df[point_level_df['year'] == year]

        # Save as parquet (efficient)
        parquet_path = POINT_LEVEL_DIR / f"point_level_{year}.parquet"
        year_df.to_parquet(parquet_path, index=False)
        print(f"Saved: {parquet_path} ({len(year_df):,} records)")

        # Save as CSV (for compatibility) - drop geometry column
        csv_path = POINT_LEVEL_DIR / f"point_level_{year}.csv"
        year_df_csv = year_df.drop(columns=['geometry']) if 'geometry' in year_df.columns else year_df
        year_df_csv.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    # Step 6: Aggregate to polygon level
    polygon_summary = aggregate_to_polygons(point_level_df)

    # Step 7: Save polygon-level summaries per year
    print(f"\n{'='*80}")
    print("SAVING POLYGON-LEVEL SUMMARIES")
    print(f"{'='*80}")

    for year in YEARS:
        year_summary = polygon_summary[polygon_summary['year'] == year]

        csv_path = POLYGON_LEVEL_DIR / f"polygon_summary_{year}.csv"
        year_summary.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path} ({len(year_summary):,} polygons)")

    # Save combined summary
    combined_path = POLYGON_LEVEL_DIR / "polygon_summary_all_years.csv"
    polygon_summary.to_csv(combined_path, index=False)
    print(f"Saved: {combined_path}")

    # Step 8: Create visualizations
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")

    # PNG version
    figure_path_png = FIGURES_DIR / "error_rate_maps_2022_2024.png"
    plot_error_maps(
        polygons_gdf,
        polygon_summary,
        years=YEARS,
        output_path=figure_path_png,
        cmap=COLORMAP,
        figsize=FIGURE_SIZE,
        dpi=DPI
    )

    # PDF version
    figure_path_pdf = FIGURES_DIR / "error_rate_maps_2022_2024.pdf"
    plot_error_maps(
        polygons_gdf,
        polygon_summary,
        years=YEARS,
        output_path=figure_path_pdf,
        cmap=COLORMAP,
        figsize=FIGURE_SIZE,
        dpi=DPI
    )

    # Final summary
    print(f"\n{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + " "*30 + "ANALYSIS COMPLETE" + " "*31 + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"\nOutput files:")
    print(f"  Point-level data:    {POINT_LEVEL_DIR}")
    print(f"  Polygon summaries:   {POLYGON_LEVEL_DIR}")
    print(f"  Visualizations:      {FIGURES_DIR}")


if __name__ == "__main__":
    main()
