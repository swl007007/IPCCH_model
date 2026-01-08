#!/usr/bin/env python3
"""
Region-Sliced Forecasting Workflow
Trains independent XGBoost models for each region across multiple years.
Regions are auto-detected from the mapping file (supports 1-20 regions).
Total runs = (number of years) × (number of regions detected).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score

# Import functions from food_crisis_functions.py
from food_crisis_functions import convert_prob_to_phase, all_metrics


# ============================================================================
# 1. DATA LOADING & VALIDATION
# ============================================================================

def load_inputs(
    dataset_path: str,
    region_map_path: str,
    hyperparam_path: str,
    hyperparam_p3_path: str,
    lat_lon_path: str,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict, List[int]]:
    """
    Load all required inputs with validation.

    Returns:
        - df: Main dataset (sorted by area_id, date for determinism)
        - region_map: area_id -> region mapping
        - lat_lon_lookup: area_id -> lat, lon mapping
        - hyperparams: XGBoost params for phases 2,4,5
        - hyperparams_p3: XGBoost params for phase 3
        - unique_regions: Sorted list of unique region IDs from mapping file
    """
    print("=" * 80)
    print("LOADING INPUTS")
    print("=" * 80)

    # Check file existence
    for path, name in [
        (dataset_path, "Dataset"),
        (region_map_path, "Region mapping"),
        (hyperparam_path, "Hyperparameters"),
        (hyperparam_p3_path, "Hyperparameters (Phase 3)")
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"  [OK] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Validate required columns (lat, lon are in separate file)
    required_cols = [
        'area_id', 'year', 'month',
        'phase1_percent', 'phase2_percent', 'phase3_percent',
        'phase4_percent', 'phase5_percent', 'overall_phase'
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Sort for determinism
    df = df.sort_values(by=['area_id', 'year', 'month']).reset_index(drop=True)
    print(f"  [OK] Unique area_ids: {df['area_id'].nunique()}")
    print(f"  [OK] Date range: {df['year'].min()}-{df['month'].min():02d} to {df['year'].max()}-{df['month'].max():02d}")

    # Load region mapping
    print(f"\nLoading region mapping from: {region_map_path}")
    region_map = pd.read_csv(region_map_path)
    print(f"  [OK] Loaded {len(region_map):,} area_id mappings")

    # Validate region mapping columns
    if not {'area_id', 'region'}.issubset(region_map.columns):
        raise ValueError(f"Region mapping must have 'area_id' and 'region' columns. Found: {region_map.columns.tolist()}")

    region_map = region_map.sort_values('area_id').reset_index(drop=True)
    unique_regions = sorted(region_map['region'].unique())

    # Validate region count
    if len(unique_regions) < 1:
        raise ValueError("No regions found in mapping file. Check 'region' column.")

    if len(unique_regions) > 20:
        raise ValueError(f"Found {len(unique_regions)} regions. Maximum supported: 20.")

    print(f"  [OK] Found {len(unique_regions)} unique regions: {unique_regions}")

    # Load lat/lon lookup
    print(f"\nLoading lat/lon lookup from: {lat_lon_path}")
    lat_lon_lookup = pd.read_csv(lat_lon_path)

    # Keep only admin_code, lat, lon columns
    if 'admin_code' in lat_lon_lookup.columns:
        lat_lon_lookup = lat_lon_lookup[['admin_code', 'lat', 'lon']].copy()
        lat_lon_lookup = lat_lon_lookup.rename(columns={'admin_code': 'area_id'})
    elif 'area_id' in lat_lon_lookup.columns:
        lat_lon_lookup = lat_lon_lookup[['area_id', 'lat', 'lon']].copy()
    else:
        raise ValueError(f"Lat/lon file must have 'admin_code' or 'area_id' column. Found: {lat_lon_lookup.columns.tolist()}")

    # Drop duplicates
    lat_lon_lookup = lat_lon_lookup.drop_duplicates(subset='area_id')
    print(f"  [OK] Loaded {len(lat_lon_lookup):,} area_id -> lat/lon mappings")

    # Load hyperparameters
    print(f"\nLoading hyperparameters...")
    with open(hyperparam_path, 'r') as f:
        hyperparams = json.load(f)
    print(f"  [OK] Hyperparameters (phases 2,4,5): {hyperparam_path}")

    with open(hyperparam_p3_path, 'r') as f:
        hyperparams_p3 = json.load(f)
    print(f"  [OK] Hyperparameters (phase 3): {hyperparam_p3_path}")

    # Add random_state to hyperparameters
    hyperparams['random_state'] = seed
    hyperparams_p3['random_state'] = seed

    print(f"\n  [OK] Random seed set to: {seed}")
    print("=" * 80)

    return df, region_map, lat_lon_lookup, hyperparams, hyperparams_p3, unique_regions


def merge_region_keys(df: pd.DataFrame, region_map: pd.DataFrame) -> pd.DataFrame:
    """
    Merge region keys onto dataset using LEFT JOIN to preserve all data.

    Args:
        df: Main dataset
        region_map: Mapping with columns [area_id, region]

    Returns:
        df with 'region' column added
    """
    print("\nMERGING REGION KEYS")
    print("-" * 80)

    # Keep only area_id and region columns
    region_map_clean = region_map[['area_id', 'region']].copy()

    # Left merge to preserve all dataset rows
    df_merged = df.merge(region_map_clean, on='area_id', how='left')

    print(f"  [OK] Merged region keys (LEFT JOIN on area_id)")
    print(f"  [OK] Rows before merge: {len(df):,}")
    print(f"  [OK] Rows after merge: {len(df_merged):,}")

    return df_merged


def validate_merge_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate region mapping coverage and warn/error as needed.

    Checks:
        1. Count unmapped area_ids (region is NaN)
        2. If >5% unmapped -> WARNING
        3. If >10% unmapped -> ERROR
        4. Count rows per region

    Returns:
        DataFrame with unmapped rows dropped
    """
    print("\nVALIDATING REGION COVERAGE")
    print("-" * 80)

    # Count unmapped
    unmapped = df[df['region'].isna()]
    unmapped_pct = len(unmapped) / len(df) * 100

    print(f"  Total rows in dataset: {len(df):,}")
    print(f"  Unmapped rows: {len(unmapped):,} ({unmapped_pct:.2f}%)")

    if len(unmapped) > 0:
        unmapped_area_ids = unmapped['area_id'].unique()
        print(f"  Unmapped area_ids ({len(unmapped_area_ids)}): {unmapped_area_ids[:20].tolist()}" +
              ("..." if len(unmapped_area_ids) > 20 else ""))

    # Check thresholds
    if unmapped_pct > 10:
        raise ValueError(f"ERROR: {unmapped_pct:.1f}% of data unmapped (threshold: 10%)")
    elif unmapped_pct > 5:
        print(f"  [WARNING] {unmapped_pct:.1f}% of data unmapped (threshold: 5%)")

    # Coverage table by region
    print("\nRegion Coverage (BEFORE preprocessing):")
    print(f"{'Region':>8} | {'Area IDs':>10} | {'Rows':>10} | {'Coverage %':>12}")
    print("-" * 50)

    df_mapped = df[df['region'].notna()].copy()

    for region in sorted(df_mapped['region'].unique()):
        df_region = df_mapped[df_mapped['region'] == region]
        n_area_ids = df_region['area_id'].nunique()
        n_rows = len(df_region)
        coverage_pct = (n_rows / len(df)) * 100
        print(f"{region:8.0f} | {n_area_ids:10} | {n_rows:10,} | {coverage_pct:11.2f}%")

        if n_rows < 10:
            print(f"  [WARNING] Region {region} has only {n_rows} rows")

    print("-" * 50)
    print(f"{'TOTAL':>8} | {df_mapped['area_id'].nunique():10} | {len(df_mapped):10,} | {100.0:11.2f}%")

    # Drop unmapped rows
    if len(unmapped) > 0:
        print(f"\n  [OK] Dropping {len(unmapped):,} unmapped rows")
        df_clean = df_mapped
    else:
        print("  [OK] All rows successfully mapped")
        df_clean = df

    print("=" * 80)
    return df_clean


# ============================================================================
# 2. PREPROCESSING (MATCH NOTEBOOK EXACTLY)
# ============================================================================

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply exact preprocessing from notebook.

    Returns:
        - df: Preprocessed DataFrame ready for temporal splitting
        - overall_phase_lookup: Lookup table for merging overall_phase back later
    """
    print("\nPREPROCESSING DATA")
    print("-" * 80)

    # 1. Create date column
    print("  1. Creating date column...")
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))

    # 2. Replace inf/-inf with NaN
    print("  2. Replacing inf/-inf with NaN...")
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. Clip overall_phase to [1, 5]
    print("  3. Clipping overall_phase to [1, 5]...")
    df['overall_phase'] = df['overall_phase'].apply(
        lambda x: 1 if x < 1 else (5 if x > 5 else x)
    )

    # 4. Filter: keep only phase1_percent.notna()
    print(f"  4. Filtering rows where phase1_percent is not NaN...")
    rows_before = len(df)
    df = df[df['phase1_percent'].notna()]
    rows_after = len(df)
    print(f"     Rows: {rows_before:,} → {rows_after:,} (dropped {rows_before - rows_after:,})")

    # 5. Sort by ['area_id', 'date']
    print("  5. Sorting by area_id, date...")
    df = df.sort_values(by=['area_id', 'date']).reset_index(drop=True)

    # 6. Store overall_phase separately before dropping
    print("  6. Storing overall_phase lookup...")
    overall_phase_lookup = df[['area_id', 'date', 'overall_phase']].copy()

    # 7. Create phase_worse targets
    print("  7. Creating phase_worse targets...")
    df['phase2_worse'] = (df['phase2_percent'] + df['phase3_percent'] +
                          df['phase4_percent'] + df['phase5_percent'])
    df['phase3_worse'] = (df['phase3_percent'] + df['phase4_percent'] +
                          df['phase5_percent'])
    df['phase4_worse'] = df['phase4_percent'] + df['phase5_percent']
    df['phase5_worse'] = df['phase5_percent'].copy()

    # 8. Drop original phase columns
    print("  8. Dropping original phase columns...")
    cols_to_drop = ['phase1_percent', 'phase2_percent', 'phase3_percent',
                    'phase4_percent', 'phase5_percent', 'overall_phase']
    df = df.drop(cols_to_drop, axis=1)

    print(f"\n  [OK] Preprocessing complete. Final shape: {df.shape}")
    print("=" * 80)

    return df, overall_phase_lookup


def get_temporal_split_config(year: int) -> Dict[str, str]:
    """
    Return temporal split configuration for each year.

    Returns dict with keys: train_start, test_start, cutoff (or None)
    """
    configs = {
        2024: {
            'train_start': '2021-01-01',
            'test_start': '2024-01-01',
            'cutoff': None
        },
        2023: {
            'train_start': '2020-01-01',
            'test_start': '2023-01-01',
            'cutoff': '2024-01-01'
        },
        2022: {
            'train_start': '2019-01-01',
            'test_start': '2022-01-01',
            'cutoff': '2023-01-01'
        }
    }

    if year not in configs:
        raise ValueError(f"No temporal split config for year {year}. Available: {list(configs.keys())}")

    return configs[year]


def temporal_split(df: pd.DataFrame, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test based on temporal logic.

    Returns:
        train_df, test_df (both still have area_id, date, region columns)
    """
    config = get_temporal_split_config(year)

    train_start = pd.to_datetime(config['train_start'])
    test_start = pd.to_datetime(config['test_start'])
    cutoff = pd.to_datetime(config['cutoff']) if config['cutoff'] else None

    if cutoff is None:
        # Year 2024: no cutoff
        train_df = df[(df['date'] >= train_start) & (df['date'] < test_start)].copy()
        test_df = df[df['date'] >= test_start].copy()
    else:
        # Years 2022, 2023: with cutoff to prevent future leakage
        # Train: [train_start, test_start) - NO OVERLAP with test period
        train_df = df[(df['date'] >= train_start) & (df['date'] < test_start)].copy()
        # Test: [test_start, cutoff) - only test on the target year
        test_df = df[(df['date'] >= test_start) & (df['date'] < cutoff)].copy()

    return train_df, test_df


# ============================================================================
# 3. MODEL TRAINING PIPELINE
# ============================================================================

def train_phase_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    phase: int,
    hyperparams: Dict,
    hyperparams_p3: Dict,
    seed: int
) -> xgb.XGBRegressor:
    """
    Train single XGBoost regressor for one phase.

    Args:
        phase: 2, 3, 4, or 5
        hyperparams: default params (for phases 2,4,5)
        hyperparams_p3: phase-3 specific params
        seed: random seed

    Returns:
        Trained model
    """
    # Select hyperparameters
    params = hyperparams_p3.copy() if phase == 3 else hyperparams.copy()

    # Ensure random_state is set
    params['random_state'] = seed

    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    return model


def run_one_region(
    df: pd.DataFrame,
    overall_phase_lookup: pd.DataFrame,
    lat_lon_lookup: pd.DataFrame,
    year: int,
    region: int,
    hyperparams: Dict,
    hyperparams_p3: Dict,
    seed: int,
    output_dir: str
) -> Dict[str, Any]:
    """
    Complete pipeline for one year-region combination.

    Returns:
        Result dictionary with metrics and file paths
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"YEAR {year}, REGION {region}")
        print(f"{'=' * 80}")

        # 1. Filter to current region
        print(f"  Filtering to region {region}...")
        print(f"    Total rows in global df: {len(df):,}")
        df_region = df[df['region'] == region].copy()
        print(f"    Rows after region filter: {len(df_region):,}")

        if len(df_region) == 0:
            return {
                'year': year,
                'region': region,
                'n_train': 0,
                'n_test': 0,
                'accuracy': None,
                'sensitivity': None,
                'precision': None,
                'r2_phase3plus': None,
                'pred_file': None,
                'metrics_file': None,
                'status': 'error',
                'error_msg': 'No samples in region'
            }

        print(f"  Region subset: {len(df_region):,} rows")

        # 2. Temporal split
        print(f"\n  Applying temporal split for year {year}...")
        config = get_temporal_split_config(year)
        print(f"    Train start: {config['train_start']}")
        print(f"    Test start: {config['test_start']}")
        print(f"    Cutoff: {config.get('cutoff', 'None')}")

        train_df, test_df = temporal_split(df_region, year)

        print(f"  Train: {len(train_df):,} rows")
        print(f"  Test: {len(test_df):,} rows")

        if len(train_df) > 0:
            print(f"    Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
        if len(test_df) > 0:
            print(f"    Test date range: {test_df['date'].min()} to {test_df['date'].max()}")

        if len(train_df) == 0:
            return {
                'year': year,
                'region': region,
                'n_train': 0,
                'n_test': len(test_df),
                'accuracy': None,
                'sensitivity': None,
                'precision': None,
                'r2_phase3plus': None,
                'pred_file': None,
                'metrics_file': None,
                'status': 'error',
                'error_msg': 'No training samples'
            }

        if len(test_df) < 10:
            return {
                'year': year,
                'region': region,
                'n_train': len(train_df),
                'n_test': len(test_df),
                'accuracy': None,
                'sensitivity': None,
                'precision': None,
                'r2_phase3plus': None,
                'pred_file': None,
                'metrics_file': None,
                'status': 'error',
                'error_msg': f'Insufficient test samples: {len(test_df)} < 10'
            }

        # 3. Store metadata for later merge
        test_metadata = test_df[['area_id', 'date']].copy()

        # 4. Drop area_id, date, region from features (lat, lon not in dataset)
        cols_to_drop = ['area_id', 'date', 'region']
        # Only drop columns that exist
        cols_to_drop = [col for col in cols_to_drop if col in train_df.columns]
        train_df = train_df.drop(cols_to_drop, axis=1)
        test_df = test_df.drop(cols_to_drop, axis=1)

        # 5. Train 4 phase models and collect predictions
        print(f"\n  Training phase models...")
        y_pred_test = pd.DataFrame()

        for i in range(2, 6):  # phases 2, 3, 4, 5
            print(f"    Phase {i}...", end=' ')

            # Prepare data for this phase
            train_df_new = train_df.drop(
                [f'phase{j}_worse' for j in range(2, 6) if j != i],
                axis=1
            )
            test_df_new = test_df.drop(
                [f'phase{j}_worse' for j in range(2, 6) if j != i],
                axis=1
            )

            # Drop NaN in target
            train_df_new = train_df_new.dropna(subset=[f'phase{i}_worse'])
            test_df_new = test_df_new.dropna(subset=[f'phase{i}_worse'])

            test_index = test_df_new.index

            # Split features and target
            X_train = train_df_new.drop(f'phase{i}_worse', axis=1)
            y_train = train_df_new[f'phase{i}_worse']
            X_test = test_df_new.drop(f'phase{i}_worse', axis=1)
            y_test = test_df_new[f'phase{i}_worse']

            # Train model
            model = train_phase_model(X_train, y_train, i, hyperparams, hyperparams_p3, seed)

            # Predict
            y_pred = model.predict(X_test)

            # Store predictions (ONLY phase 5 gets test_index, as in notebook)
            if i != 5:
                phase_results = pd.DataFrame({
                    'y_pred': y_pred,
                    'y_test': y_test.values,
                    'phase': [i] * len(y_pred)
                })
            else:  # Phase 5 only
                phase_results = pd.DataFrame({
                    'y_pred': y_pred,
                    'y_test': y_test.values,
                    'phase': [i] * len(y_pred),
                    'test_index': test_index.tolist()  # Convert to list to avoid index issues
                })

            y_pred_test = pd.concat([y_pred_test, phase_results], ignore_index=True)
            print(f"[OK] ({len(X_train)} train, {len(X_test)} test)")

        # 6. Reshape predictions to wide format
        print(f"\n  Converting predictions to phase classifications...")

        # Add phase 1 as placeholder (will be dropped in convert_prob_to_phase)
        # Note: phase1 doesn't need test_index - only phase 5 has it
        phase1_df = pd.DataFrame({
            'y_pred': [0.0] * len(y_pred_test[y_pred_test['phase'] == 2]),
            'y_test': [0.0] * len(y_pred_test[y_pred_test['phase'] == 2]),
            'phase': [1] * len(y_pred_test[y_pred_test['phase'] == 2])
        })
        y_pred_test = pd.concat([phase1_df, y_pred_test], ignore_index=True)

        # Convert to phase classifications (test_index from phase 5 passes through)
        y_pred_test = convert_prob_to_phase(y_pred_test, th=0.2)

        # Fix duplicate test_index columns (pandas concat creates duplicates during horizontal concat)
        # All test_index columns have the same non-NaN values from phase 5, so just keep the first
        if 'test_index' in y_pred_test.columns:
            # Get all test_index columns
            test_index_cols = [col for col in y_pred_test.columns if col == 'test_index']
            if len(test_index_cols) > 1:
                # Keep only the first test_index column, drop the rest
                # Use column positions to avoid the "not unique" error
                cols_to_keep = []
                test_index_kept = False
                for col in y_pred_test.columns:
                    if col == 'test_index':
                        if not test_index_kept:
                            cols_to_keep.append(col)
                            test_index_kept = True
                        # Skip subsequent test_index columns
                    else:
                        cols_to_keep.append(col)

                # Reconstruct DataFrame with unique column names
                y_pred_test = y_pred_test.loc[:, ~y_pred_test.columns.duplicated()]

        # 7. Merge back metadata (area_id, date) using test_metadata stored earlier
        # test_metadata has the correct area_id, date for test_df rows
        # Need to align it with y_pred_test which has same length after convert_prob_to_phase
        y_pred_test = y_pred_test.reset_index(drop=True)
        test_metadata_aligned = test_metadata.reset_index(drop=True)

        # Validation: ensure lengths match (should be equal if all phases had same valid rows)
        if len(y_pred_test) != len(test_metadata_aligned):
            print(f"  [WARNING] Length mismatch: y_pred_test={len(y_pred_test)}, test_metadata={len(test_metadata_aligned)}")
            # Use the shorter length to avoid index errors
            min_len = min(len(y_pred_test), len(test_metadata_aligned))
            y_pred_test = y_pred_test.iloc[:min_len]
            test_metadata_aligned = test_metadata_aligned.iloc[:min_len]

        y_pred_test['area_id'] = test_metadata_aligned['area_id'].values
        y_pred_test['date'] = test_metadata_aligned['date'].values

        # 8. Merge lat, lon from lookup
        y_pred_test = y_pred_test.merge(lat_lon_lookup, on='area_id', how='left')

        # 9. Save predictions (NO metrics calculation - done after concatenation)
        pred_filename = f"predictions_{year}_{region}.csv"
        pred_path = os.path.join(output_dir, 'predictions', pred_filename)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        y_pred_test.to_csv(pred_path, index=False)
        print(f"\n  [OK] Saved predictions: {pred_filename} ({len(y_pred_test)} rows)")

        return {
            'year': year,
            'region': region,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'pred_file': pred_filename,
            'status': 'success',
            'error_msg': None
        }

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"\n  [ERROR] {error_msg}")

        return {
            'year': year,
            'region': region,
            'n_train': 0,
            'n_test': 0,
            'pred_file': None,
            'status': 'error',
            'error_msg': error_msg
        }


# ============================================================================
# 4. AGGREGATION & METRICS
# ============================================================================

def concat_regional_predictions(year: int, output_dir: str, regions: List[int]) -> pd.DataFrame:
    """
    Concatenate all regional predictions for a year.

    Args:
        year: Year to concatenate
        output_dir: Output directory containing predictions/
        regions: List of region IDs to process

    Returns:
        Combined predictions DataFrame
    """
    print(f"\n{'=' * 80}")
    print(f"CONCATENATING REGIONAL PREDICTIONS FOR YEAR {year}")
    print(f"{'=' * 80}")

    dfs = []
    total_rows = 0

    for region in regions:
        pred_path = os.path.join(output_dir, 'predictions', f'predictions_{year}_{region}.csv')

        if os.path.exists(pred_path):
            df_region = pd.read_csv(pred_path)
            dfs.append(df_region)
            total_rows += len(df_region)
            print(f"  Region {region:2d}: {len(df_region):6,} rows")
        else:
            print(f"  Region {region:2d}: MISSING")

    if not dfs:
        raise ValueError(f"No regional predictions found for year {year}")

    # Concatenate
    df_all = pd.concat(dfs, ignore_index=True)

    # Validate
    print(f"\n  Validation:")
    print(f"    Sum of regional rows: {total_rows:,}")
    print(f"    Concatenated rows: {len(df_all):,}")

    if total_rows != len(df_all):
        print(f"    [WARNING] Row count mismatch!")
    else:
        print(f"    [OK] Row counts match")

    # Check for duplicates
    duplicates = df_all.duplicated(subset=['area_id', 'date'])
    if duplicates.sum() > 0:
        print(f"    [WARNING] {duplicates.sum()} duplicate (area_id, date) pairs found")
    else:
        print(f"    [OK] No duplicate (area_id, date) pairs")

    # Sort and save
    df_all = df_all.sort_values(['area_id', 'date']).reset_index(drop=True)

    output_path = os.path.join(output_dir, 'predictions', f'predictions_{year}_ALL_REGIONS.csv')
    df_all.to_csv(output_path, index=False)
    print(f"\n  [OK] Saved: predictions_{year}_ALL_REGIONS.csv ({len(df_all):,} rows)")

    return df_all


def compute_year_metrics(df_all: pd.DataFrame, year: int, output_dir: str) -> Dict[str, any]:
    """
    Compute metrics on the COMPLETE concatenated predictions for a year.
    This avoids unreliable metrics from sparse individual regions.

    Args:
        df_all: Concatenated predictions from all regions
        year: Year being evaluated
        output_dir: Output directory

    Returns:
        {'year': year, 'n_total': ..., 'accuracy': ..., 'sensitivity': ...,
         'precision': ..., 'r2_phase3plus': ...}
    """
    print(f"\nComputing metrics on concatenated data for year {year}...")

    y_test = df_all['overall_phase']
    y_pred = df_all['overall_phase_pred']

    cm = confusion_matrix(y_test, y_pred)
    accuracy, sensitivity, precision, r2_phase3plus = all_metrics(
        y_test, y_pred, cm, df_all
    )

    metrics_dict = {
        'year': year,
        'n_total': len(df_all),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'precision': float(precision),
        'r2_phase3plus': float(r2_phase3plus)
    }

    # Save to file
    metrics_path = os.path.join(output_dir, 'metrics', f'metrics_{year}_OVERALL.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"  Results:")
    print(f"    Total samples: {len(df_all)}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Sensitivity: {sensitivity:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    R2 (Phase 3+): {r2_phase3plus:.4f}")
    print(f"\n  [OK] Saved overall metrics: metrics_{year}_OVERALL.json")

    return metrics_dict


def write_manifest(results: List[Dict], output_dir: str) -> None:
    """
    Write run_manifest.csv with all runs.

    Columns:
        year, region, n_train, n_test, pred_file, status, error_msg
    Note: Metrics are NOT included per-region (only calculated at year level)
    """
    print(f"\n{'=' * 80}")
    print("WRITING MANIFEST")
    print(f"{'=' * 80}")

    df_manifest = pd.DataFrame(results)
    df_manifest = df_manifest.sort_values(['year', 'region']).reset_index(drop=True)

    manifest_path = os.path.join(output_dir, 'run_manifest.csv')
    df_manifest.to_csv(manifest_path, index=False)

    print(f"  [OK] Saved manifest: run_manifest.csv ({len(df_manifest)} runs)")
    print(f"\n  Summary:")
    print(f"    Total runs: {len(df_manifest)}")
    print(f"    Successful: {(df_manifest['status'] == 'success').sum()}")
    print(f"    Failed: {(df_manifest['status'] == 'error').sum()}")

    if (df_manifest['status'] == 'error').sum() > 0:
        print(f"\n  Failed runs:")
        failed = df_manifest[df_manifest['status'] == 'error']
        for _, row in failed.iterrows():
            print(f"    Year {row['year']}, Region {row['region']}: {row['error_msg'][:60]}...")


# ============================================================================
# 5. MAIN ORCHESTRATOR
# ============================================================================

def main():
    """
    Main execution flow.
    """
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description='Region-sliced forecasting workflow with auto-detected regions'
    )
    parser.add_argument('--dataset', required=True,
                       help='Path to main CSV dataset')
    parser.add_argument('--region-map', required=True,
                       help='Path to area_id region mapping CSV')
    parser.add_argument('--out', required=True,
                       help='Output directory')
    parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024],
                       help='Years to process (default: 2022 2023 2024)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--hyperparams', default='forecasting_hyperparameters.json',
                       help='Path to hyperparameters JSON (phases 2,4,5)')
    parser.add_argument('--hyperparams-p3', default='forecasting_hyperparameters_p3.json',
                       help='Path to hyperparameters JSON (phase 3)')
    parser.add_argument('--lat-lon-file',
                       default=r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\IPCCH_2017_2025_final_v12102025_with_zscores.csv',
                       help='Path to file with lat/lon coordinates (default: IPCCH_2017_2025_final_v12102025_with_zscores.csv)')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("REGION-SLICED FORECASTING WORKFLOW")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Region map: {args.region_map}")
    print(f"  Output directory: {args.out}")
    print(f"  Years: {args.years}")
    print(f"  Random seed: {args.seed}")
    print("=" * 80)

    # 1. Load inputs
    df, region_map, lat_lon_lookup, hyperparams, hyperparams_p3, unique_regions = load_inputs(
        args.dataset,
        args.region_map,
        args.hyperparams,
        args.hyperparams_p3,
        args.lat_lon_file,
        args.seed
    )

    # Display region configuration
    print(f"\n{'=' * 80}")
    print("REGION CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"  Detected regions: {len(unique_regions)}")
    print(f"  Region IDs: {unique_regions}")
    print(f"  Total runs: {len(args.years)} years × {len(unique_regions)} regions = {len(args.years) * len(unique_regions)}")
    print(f"{'=' * 80}")

    # 2. Merge region keys and validate
    df = merge_region_keys(df, region_map)
    df = validate_merge_coverage(df)

    # 3. Preprocess data (once, globally)
    df, overall_phase_lookup = preprocess_data(df)

    # 3b. Print region distribution AFTER preprocessing
    print("\n" + "=" * 80)
    print("REGION DISTRIBUTION AFTER PREPROCESSING")
    print("=" * 80)
    print(f"\nTotal rows after preprocessing: {len(df):,}")
    print(f"\nRows per region:")
    region_dist = df.groupby('region').size()
    for region, count in region_dist.items():
        print(f"  Region {int(region):2d}: {count:6,} rows ({count/len(df)*100:5.2f}%)")
    print("\nArea IDs per region:")
    area_dist = df.groupby('region')['area_id'].nunique()
    for region, count in area_dist.items():
        print(f"  Region {int(region):2d}: {count:3d} unique area_ids")
    print("=" * 80)

    # 4. Create output directory structure
    os.makedirs(os.path.join(args.out, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'metrics'), exist_ok=True)

    # 5. Run year-region iterations (dynamic region count)
    results = []

    for year in args.years:
        for region in unique_regions:
            result = run_one_region(
                df=df,
                overall_phase_lookup=overall_phase_lookup,
                lat_lon_lookup=lat_lon_lookup,
                year=year,
                region=region,
                hyperparams=hyperparams,
                hyperparams_p3=hyperparams_p3,
                seed=args.seed,
                output_dir=args.out
            )
            results.append(result)

            print(f"\n  Status: {result['status'].upper()}")
            if result['status'] == 'error':
                print(f"  Error: {result['error_msg'][:100]}")

    # 6. Aggregate predictions by year
    for year in args.years:
        try:
            df_all = concat_regional_predictions(year, args.out, unique_regions)

            # Compute metrics on the concatenated data (not per-region)
            compute_year_metrics(df_all, year, args.out)

        except Exception as e:
            print(f"\n  [ERROR] Error aggregating year {year}: {e}")

    # 7. Write manifest
    write_manifest(results, args.out)

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)

    # Final summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')

    print(f"\nFinal Summary:")
    print(f"  Total runs: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {successful / len(results) * 100:.1f}%")

    if failed == 0:
        print(f"\n  [OK] All runs completed successfully!")
    else:
        print(f"\n  [WARNING] {failed} run(s) failed. Check run_manifest.csv for details.")

    print("\nOutput files:")
    print(f"  {args.out}/predictions/predictions_<year>_<region>.csv  ({len(args.years) * len(unique_regions)} files)")
    print(f"  {args.out}/predictions/predictions_<year>_ALL_REGIONS.csv  (3 files)")
    print(f"  {args.out}/metrics/metrics_<year>_OVERALL.json  (3 files)")
    print(f"  {args.out}/run_manifest.csv")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
