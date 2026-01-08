# Region-Sliced Forecasting Workflow

## Overview

`run_region_models.py` implements a reproducible region-sliced forecasting workflow that trains independent XGBoost models for each of 12 regions across 3 years (2022, 2023, 2024), totaling 36 model-training runs.

## Requirements

### Python Dependencies

```bash
pip install numpy pandas xgboost scikit-learn
```

Or if you're using conda:

```bash
conda install numpy pandas xgboost scikit-learn
```

### Required Files

1. **Data files (must exist):**
   - `forecasting_subset_IPCCH_v1210.csv` - Main dataset (29,622 rows, 225 columns)
   - `area_id_country_region_mapping.csv` - Region mapping (area_id → region)
   - `IPCCH_2017_2025_final_v12102025_with_zscores.csv` - Lat/lon coordinates (admin_code → lat, lon)

2. **Hyperparameter files (must be in same directory as script):**
   - `forecasting_hyperparameters.json` - XGBoost params for phases 2, 4, 5
   - `forecasting_hyperparameters_p3.json` - XGBoost params for phase 3

3. **Code dependencies:**
   - `food_crisis_functions.py` - Must be in same directory or on PYTHONPATH
     - Provides: `convert_prob_to_phase()`, `all_metrics()`

## Usage

### Basic Usage

```bash
python run_region_models.py \
    --dataset "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\forecasting_subset_IPCCH_v1210.csv" \
    --region-map "area_id_country_region_mapping.csv" \
    --out "output_region_models" \
    --years 2022 2023 2024 \
    --seed 42
```

**Note:** The `--lat-lon-file` parameter has a default value and doesn't need to be specified unless you're using a different file.

### Command-Line Arguments

- `--dataset` (required): Path to main CSV dataset
- `--region-map` (required): Path to area_id region mapping CSV
- `--out` (required): Output directory (will be created if doesn't exist)
- `--years` (optional): Years to process (default: 2022 2023 2024)
- `--seed` (optional): Random seed for reproducibility (default: 42)
- `--hyperparams` (optional): Path to hyperparameters JSON for phases 2,4,5 (default: `forecasting_hyperparameters.json`)
- `--hyperparams-p3` (optional): Path to hyperparameters JSON for phase 3 (default: `forecasting_hyperparameters_p3.json`)
- `--lat-lon-file` (optional): Path to file with lat/lon coordinates (default: `IPCCH_2017_2025_final_v12102025_with_zscores.csv`)

### Test Run (Single Region)

To test with a single year and verify everything works:

```bash
# Modify the script temporarily to only run year 2024, region 1
python run_region_models.py \
    --dataset "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\forecasting_subset_IPCCH_v1210.csv" \
    --region-map "area_id_country_region_mapping.csv" \
    --out "test_output" \
    --years 2024 \
    --seed 42
```

Then manually check the outputs in `test_output/` directory.

## Expected Outputs

After running the script with default arguments (3 years × 12 regions = 36 runs):

```
{output_dir}/
├── predictions/
│   ├── predictions_2022_1.csv
│   ├── predictions_2022_2.csv
│   ├── ...
│   ├── predictions_2022_12.csv
│   ├── predictions_2022_ALL_REGIONS.csv
│   ├── predictions_2023_1.csv
│   ├── ...
│   ├── predictions_2023_12.csv
│   ├── predictions_2023_ALL_REGIONS.csv
│   ├── predictions_2024_1.csv
│   ├── ...
│   ├── predictions_2024_12.csv
│   └── predictions_2024_ALL_REGIONS.csv
│
├── metrics/
│   ├── metrics_2022_1.json
│   ├── ...
│   ├── metrics_2022_12.json
│   ├── metrics_2022_OVERALL.json
│   ├── metrics_2023_1.json
│   ├── ...
│   ├── metrics_2023_12.json
│   ├── metrics_2023_OVERALL.json
│   ├── metrics_2024_1.json
│   ├── ...
│   ├── metrics_2024_12.json
│   └── metrics_2024_OVERALL.json
│
└── run_manifest.csv
```

### File Schemas

**predictions_{year}_{region}.csv:**
```
Columns: phase2_pred, phase2_test, phase3_pred, phase3_test,
         phase4_pred, phase4_test, phase5_pred, phase5_test,
         test_index, overall_phase, overall_phase_pred,
         area_id, date, lat, lon
```

**metrics_{year}_{region}.json:**
```json
{
  "year": 2022,
  "region": 1,
  "n_train": 1523,
  "n_test": 87,
  "accuracy": 0.7234,
  "sensitivity": 0.8456,
  "precision": 0.7912,
  "r2_phase3plus": 0.8123
}
```

**run_manifest.csv:**
```
Columns: year, region, n_train, n_test, accuracy, sensitivity,
         precision, r2_phase3plus, pred_file, metrics_file,
         status, error_msg
Rows: 36 (one per year-region combination)
```

## Workflow Details

### 1. Data Loading & Validation
- Loads dataset and region mapping
- Validates required columns exist
- Merges region keys via LEFT JOIN on `area_id`
- Validates coverage (warns if >5% unmapped, errors if >10%)

### 2. Preprocessing (Exact Match to Notebook)
- Creates `date` column from year/month
- Replaces inf/-inf with NaN
- Clips `overall_phase` to [1, 5]
- Filters to rows where `phase1_percent` is not NaN
- Sorts by area_id, date
- Creates cumulative targets: `phase2_worse`, `phase3_worse`, `phase4_worse`, `phase5_worse`
- Stores `overall_phase` lookup for later merge

### 3. Model Training (Per Year-Region)
For each of 36 combinations (3 years × 12 regions):

- Filters data to current region
- Applies temporal split (year-specific logic):
  - **2024:** Train on 2021-2023, test on 2024+
  - **2023:** Train on 2020-2023, test on 2023 only
  - **2022:** Train on 2019-2022, test on 2022 only
- Trains 4 separate XGBoost regressors (phases 2, 3, 4, 5)
  - Phase 3 uses specialized hyperparameters
- Converts regression outputs to phase classifications (threshold=0.2)
- Merges back metadata (area_id, date, lat, lon, overall_phase)
- Computes metrics: accuracy, sensitivity, precision, R² for phase 3+
- Saves predictions CSV and metrics JSON

### 4. Aggregation (Per Year)
- Concatenates all 12 regional predictions
- Validates row counts match
- Computes overall metrics (micro-averaged)
- Saves ALL_REGIONS files

### 5. Manifest
- Writes `run_manifest.csv` documenting all 36 runs
- Includes status (success/error) and error messages

## Validation Checkpoints

The script includes extensive validation:

1. **After merge:** Checks unmapped area_ids, coverage by region
2. **After temporal split:** Verifies non-empty train/test sets, no overlap
3. **After concatenation:** Validates row counts, checks for duplicates
4. **At manifest:** Verifies success rate, lists failed runs

## Determinism

The script ensures reproducibility via:

- `random_state=seed` in all XGBoost models
- Sorting by area_id, date after every merge/load
- Stable LEFT JOIN merge strategy
- Index preservation for metadata merging

## Troubleshooting

### Common Issues

1. **Import Error: `ModuleNotFoundError: No module named 'food_crisis_functions'`**
   - Solution: Run script from same directory as `food_crisis_functions.py`
   - Or: Add directory to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:/path/to/IPCCH_model"`

2. **File Not Found: Hyperparameter JSON files**
   - Solution: Ensure `forecasting_hyperparameters.json` and `forecasting_hyperparameters_p3.json` are in current directory
   - Or: Specify full paths with `--hyperparams` and `--hyperparams-p3` arguments

3. **Unmapped Area IDs Warning**
   - If >5% of data is unmapped, check that `area_id_country_region_mapping.csv` covers all area_ids in the dataset
   - Script will drop unmapped rows and continue

4. **Insufficient Test Samples Error**
   - Some regions may have <10 test samples for certain years
   - These runs will be marked as 'error' in manifest but won't stop the workflow

5. **Memory Issues**
   - If running out of memory, try processing years sequentially (run script 3 times with `--years 2022`, `--years 2023`, `--years 2024`)

### Performance

- **Estimated runtime:** 2-5 minutes per region (4 models × 30-60 seconds each)
- **Total runtime:** ~2-3 hours for all 36 regions (sequential execution)
- **Memory usage:** ~2-4 GB peak

## Comparison with Original Notebook

This script replicates the logic from `Table1_Forecasting_main_region_template.ipynb`:

- ✅ Same preprocessing steps (exact cell-by-cell match)
- ✅ Same temporal split logic (year-specific train/test dates)
- ✅ Same model architecture (4 XGBoost regressors, phase-specific hyperparameters)
- ✅ Same metric definitions (custom sensitivity/precision for phase 3+)
- ✅ Same threshold logic for phase classification (0.2 = 20%)
- ➕ Adds region-based filtering and iteration
- ➕ Adds comprehensive validation and error handling
- ➕ Adds manifest tracking for all 36 runs

## Contact

For questions or issues with this script, refer to:
- Original notebook: `Table1_Forecasting_main_region_template.ipynb`
- Shared functions: `food_crisis_functions.py`
- Project documentation: `CLAUDE.md`
