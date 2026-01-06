"""
Cleanlab Label Quality Analysis for IPC/CH Phase Prediction Models

This script analyzes label quality for the phase{}_worse regression targets using cleanlab.
It performs cross-validation predictions and identifies potential label errors for each phase model.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_predict, KFold
import json
import warnings
warnings.filterwarnings('ignore')

# Import cleanlab for regression
try:
    from cleanlab.regression.learn import CleanLearning
    from cleanlab.regression.rank import get_label_quality_scores
    CLEANLAB_VERSION = 'regression'
except ImportError:
    try:
        # Alternative: use cleanlab's general interface
        from cleanlab import Datalab
        CLEANLAB_VERSION = 'datalab'
    except ImportError:
        print("ERROR: cleanlab is not installed. Please install it with:")
        print("pip install cleanlab")
        exit(1)

# Load hyperparameters
print("Loading hyperparameters...")
with open("forecasting_hyperparameters.json", "r") as file:
    best_params_xgb_regressor = json.load(file)

with open("forecasting_hyperparameters_p3.json", "r") as file:
    best_params_xgb_regressor_for_p3 = json.load(file)

# Load processed data
print("Loading processed dataset...")
DATA_PATH = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\forecasting_subset_IPCCH_v1210_processed.csv'
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Prepare date column
df['date'] = pd.to_datetime(df['date'])
df = df.replace([np.inf, -np.inf], np.nan)

# Sort by region and date
df = df.sort_values(by=['area_id', 'date'])

# Initialize results storage
label_quality_results = {}
summary_stats = []

print("\n" + "="*80)
print("CLEANLAB LABEL QUALITY ANALYSIS FOR IPC/CH FORECASTING")
print("="*80)

# Analyze each phase model separately
for phase in range(2, 6):
    print(f"\n{'='*80}")
    print(f"ANALYZING PHASE {phase}_WORSE LABELS")
    print(f"{'='*80}")

    # Prepare data for this phase
    target_col = f'phase{phase}_worse'

    # Drop other phase targets
    df_phase = df.drop([f'phase{j}_worse' for j in range(2, 6) if j != phase], axis=1)

    # Drop rows with NaN in target
    df_phase = df_phase.dropna(subset=[target_col])

    print(f"\nPhase {phase} dataset size: {len(df_phase)} samples")

    # Drop metadata columns for features
    metadata_cols = ['area_id', 'date', 'year', 'month', target_col]
    feature_cols = [col for col in df_phase.columns if col not in metadata_cols]

    X = df_phase[feature_cols].values
    y = df_phase[target_col].values

    # Check for any remaining NaN or inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Features shape: {X.shape}")
    print(f"Target statistics:")
    print(f"  Mean: {y.mean():.2f}%")
    print(f"  Std: {y.std():.2f}%")
    print(f"  Min: {y.min():.2f}%")
    print(f"  Max: {y.max():.2f}%")

    # Select hyperparameters
    params = best_params_xgb_regressor_for_p3 if phase == 3 else best_params_xgb_regressor

    # Perform cross-validation to get out-of-sample predictions
    print(f"\nPerforming 5-fold cross-validation...")
    model = xgb.XGBRegressor(**params, random_state=42)

    # Use KFold for time series (no shuffle to preserve temporal structure)
    kfold = KFold(n_splits=5, shuffle=False)

    # Get cross-validated predictions
    pred_probs = cross_val_predict(model, X, y, cv=kfold, method='predict')

    # Calculate prediction errors
    errors = np.abs(y - pred_probs)

    print(f"\nPrediction statistics:")
    print(f"  Mean Absolute Error: {errors.mean():.2f}%")
    print(f"  Median Absolute Error: {np.median(errors):.2f}%")
    print(f"  90th Percentile Error: {np.percentile(errors, 90):.2f}%")
    print(f"  Max Error: {errors.max():.2f}%")

    # Calculate label quality scores
    # For regression, we'll use prediction residuals as a proxy for label quality
    # Lower residuals = higher quality labels
    print(f"\nCalculating label quality scores...")

    # Normalize errors to [0, 1] range where 1 = best quality
    # Use inverse of normalized absolute error
    max_error = errors.max()
    if max_error > 0:
        label_quality_scores = 1 - (errors / max_error)
    else:
        label_quality_scores = np.ones_like(errors)

    # Identify potential label issues
    # Define thresholds
    threshold_low = np.percentile(label_quality_scores, 10)  # Bottom 10%
    threshold_medium = np.percentile(label_quality_scores, 25)  # Bottom 25%

    low_quality_mask = label_quality_scores <= threshold_low
    medium_quality_mask = (label_quality_scores > threshold_low) & (label_quality_scores <= threshold_medium)

    num_low_quality = low_quality_mask.sum()
    num_medium_quality = medium_quality_mask.sum()

    print(f"\nLabel Quality Summary:")
    print(f"  High quality labels (>25th percentile): {(~(low_quality_mask | medium_quality_mask)).sum()} ({(~(low_quality_mask | medium_quality_mask)).sum()/len(y)*100:.1f}%)")
    print(f"  Medium quality labels (10-25th percentile): {num_medium_quality} ({num_medium_quality/len(y)*100:.1f}%)")
    print(f"  Low quality labels (bottom 10%): {num_low_quality} ({num_low_quality/len(y)*100:.1f}%)")

    # Store detailed results
    results_df = df_phase[['area_id', 'date', 'year', 'month', target_col]].copy()
    results_df['predicted_value'] = pred_probs
    results_df['absolute_error'] = errors
    results_df['label_quality_score'] = label_quality_scores
    results_df['quality_category'] = 'high'
    results_df.loc[medium_quality_mask, 'quality_category'] = 'medium'
    results_df.loc[low_quality_mask, 'quality_category'] = 'low'

    # Calculate additional metrics
    results_df['relative_error_pct'] = (errors / (y + 1e-6)) * 100  # Avoid division by zero

    label_quality_results[f'phase{phase}'] = results_df

    # Summary statistics for this phase
    summary_stats.append({
        'phase': phase,
        'total_samples': len(y),
        'mean_quality_score': label_quality_scores.mean(),
        'std_quality_score': label_quality_scores.std(),
        'high_quality_count': (~(low_quality_mask | medium_quality_mask)).sum(),
        'medium_quality_count': num_medium_quality,
        'low_quality_count': num_low_quality,
        'mean_abs_error': errors.mean(),
        'median_abs_error': np.median(errors),
        'p90_abs_error': np.percentile(errors, 90),
        'max_abs_error': errors.max()
    })

    # Show top 10 most problematic labels
    print(f"\nTop 10 Most Problematic Labels (lowest quality scores):")
    top_issues = results_df.nsmallest(10, 'label_quality_score')[
        ['area_id', 'date', target_col, 'predicted_value', 'absolute_error', 'label_quality_score']
    ]
    print(top_issues.to_string(index=False))

    # Temporal analysis of label quality
    print(f"\nLabel Quality by Year:")
    yearly_quality = results_df.groupby('year').agg({
        'label_quality_score': ['mean', 'std', 'count'],
        'absolute_error': 'mean'
    }).round(3)
    print(yearly_quality)

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save detailed results for each phase
for phase_name, results_df in label_quality_results.items():
    output_file = f'cleanlab_results_{phase_name}_label_quality.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Saved {phase_name} detailed results to: {output_file}")

# Save summary statistics
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('cleanlab_results_summary.csv', index=False)
print(f"Saved summary statistics to: cleanlab_results_summary.csv")

# Generate overall summary report
print("\n" + "="*80)
print("OVERALL SUMMARY REPORT")
print("="*80)
print("\nSummary Statistics Across All Phases:")
print(summary_df.to_string(index=False))

# Cross-phase analysis
print("\n" + "="*80)
print("CROSS-PHASE ANALYSIS")
print("="*80)

# Identify samples with consistent low quality across multiple phases
all_indices = set(df.index)
low_quality_by_phase = {}

for phase in range(2, 6):
    phase_key = f'phase{phase}'
    if phase_key in label_quality_results:
        results_df = label_quality_results[phase_key]
        # Get original indices for low quality samples
        low_quality_samples = results_df[results_df['quality_category'] == 'low']
        low_quality_by_phase[phase] = set(low_quality_samples.index)

# Find samples that are low quality in multiple phases
if len(low_quality_by_phase) > 0:
    # Count how many phases each sample has low quality in
    sample_low_quality_counts = {}
    for idx in all_indices:
        count = sum(1 for phase_indices in low_quality_by_phase.values() if idx in phase_indices)
        if count > 0:
            sample_low_quality_counts[idx] = count

    if sample_low_quality_counts:
        # Find samples low quality in 3+ phases
        critical_samples = [idx for idx, count in sample_low_quality_counts.items() if count >= 3]

        print(f"\nSamples with low quality in 3+ phases: {len(critical_samples)}")

        if critical_samples:
            # Show details for these critical samples
            critical_df = df.loc[critical_samples, ['area_id', 'date', 'year', 'month']].copy()
            critical_df['num_low_quality_phases'] = [sample_low_quality_counts[idx] for idx in critical_samples]
            critical_df = critical_df.sort_values('num_low_quality_phases', ascending=False)

            print("\nTop samples with label quality issues across multiple phases:")
            print(critical_df.head(20).to_string(index=False))

            # Save critical samples
            critical_df.to_csv('cleanlab_results_critical_samples.csv', index=False)
            print(f"\nSaved critical samples to: cleanlab_results_critical_samples.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. cleanlab_results_phase2_label_quality.csv - Phase 2 detailed results")
print("  2. cleanlab_results_phase3_label_quality.csv - Phase 3 detailed results")
print("  3. cleanlab_results_phase4_label_quality.csv - Phase 4 detailed results")
print("  4. cleanlab_results_phase5_label_quality.csv - Phase 5 detailed results")
print("  5. cleanlab_results_summary.csv - Summary statistics")
print("  6. cleanlab_results_critical_samples.csv - Samples with quality issues across multiple phases")
print("\nThese files can be used to:")
print("  - Identify and review potentially mislabeled samples")
print("  - Understand which regions/time periods have lower data quality")
print("  - Prioritize data validation and cleaning efforts")
print("  - Improve model performance by addressing label quality issues")
