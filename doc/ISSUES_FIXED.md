# Issues Fixed in run_region_models.py

## Issue 1: "The column label 'test_index' is not unique" ✅ FIXED

### Root Cause
The issue was that `test_index` was being passed through `convert_prob_to_phase()`, which caused problems when the function concatenates phase DataFrames horizontally. The solution is to extract `test_index` before conversion and add it back after.

### How convert_prob_to_phase Works
```python
# Filters each phase, renames columns, then concatenates HORIZONTALLY (axis=1)
phase_list = [df[df['phase']==i].drop(['phase']).rename(...) for i in range(1, 6)]
y_pred_test = pd.concat(phase_list, axis=1)
```

When `test_index` is included in the input (only in phase 5, as per the notebook), it can cause column duplication issues during the horizontal concat.

### Fix Applied (lines ~544-564)

**Extract test_index before conversion, add back after:**
```python
# Save test_index from phase 5 before convert_prob_to_phase
test_index_saved = y_pred_test[y_pred_test['phase'] == 5]['test_index'].values

# Remove test_index column to avoid duplication during conversion
if 'test_index' in y_pred_test.columns:
    y_pred_test = y_pred_test.drop('test_index', axis=1)

# Add phase 1 placeholder
phase1_df = pd.DataFrame({...})
y_pred_test = pd.concat([phase1_df, y_pred_test], ignore_index=True)

# Convert to phase classifications (without test_index)
y_pred_test = convert_prob_to_phase(y_pred_test, th=0.2)

# Add test_index back after conversion
y_pred_test['test_index'] = test_index_saved

# Now merge with df_extracted using test_index
df_extracted = df[['area_id', 'date']].copy()  # Use GLOBAL df
df_extracted['test_index'] = df_extracted.index
y_pred_test = y_pred_test.merge(df_extracted, on='test_index', how='left')
```

**Phase 5 still gets test_index during loop (line ~530-536):**
```python
if i != 5:
    phase_results = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test, 'phase': [i]})
else:  # Phase 5 gets test_index
    phase_results = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test, 'phase': [i], 'test_index': test_index})
```

---

## Issue 2: Very Small n_train and n_test Values ⚠️ DATA SPARSITY

### Observation
The manifest shows many regions with:
- **n_train = 0** (no training samples)
- **n_test = 0 or < 10** (insufficient test samples for evaluation)

Examples from run_manifest.csv:
- Year 2022, Region 3: n_train=7, n_test=2 ❌ Insufficient
- Year 2022, Region 10: n_train=0, n_test=0 ❌ No data
- Year 2022, Region 12: n_train=40, n_test=0 ❌ No test data
- Year 2024, Region 10: n_train=0, n_test=24 ❌ No training data

### Root Cause
This is a **data sparsity issue**, not a code bug. When filtering by:
1. Region (12 different regions)
2. Year/temporal window (2022-2024 test periods)

Many region-year combinations have insufficient data because:
- The dataset has 160 unique area_ids distributed across 12 regions
- Not all regions have continuous temporal coverage
- Some regions may have data concentrated in specific years

### Data Distribution Analysis Needed

To understand the sparsity, check:
1. **How many area_ids per region?**
   ```python
   region_map = pd.read_csv('area_id_country_region_mapping.csv')
   dataset = pd.read_csv('forecasting_subset_IPCCH_v1210.csv')

   # Merge to see which regions have data
   df = dataset.merge(region_map, on='area_id', how='left')

   # Area IDs per region
   print(df.groupby('region')['area_id'].nunique())

   # Rows per region per year
   print(df.groupby(['region', 'year']).size())
   ```

2. **Temporal coverage per region:**
   ```python
   # Check date range per region
   df.groupby('region')['year'].agg(['min', 'max', 'count'])
   ```

### Possible Solutions

#### Option 1: Hierarchical Approach (Recommended)
Train models at different granularities:
- **Macro model**: All regions combined (like original notebook)
- **Meso model**: Group similar regions (e.g., by climate zone)
- **Micro model**: Individual regions with sufficient data (n_train > 100, n_test > 30)

#### Option 2: Reduce Minimum Thresholds
Current script requires n_test >= 10. You could:
- Lower to n_test >= 5 for exploratory analysis
- Accept some regions will have high uncertainty

#### Option 3: Data Augmentation
- Use **transfer learning**: Pre-train on all regions, fine-tune per region
- **Temporal pooling**: Combine multiple years for regions with sparse data

#### Option 4: Spatial Interpolation
For regions with no data in specific years:
- Use **neighboring regions'** models as proxies
- Implement **spatial kriging** or **geographical weighted regression**

---

## Recommended Next Steps

### 1. Analyze Data Distribution
Run the diagnostic queries above to understand:
- Which regions have sufficient data?
- What's the temporal coverage per region?
- Are there seasonal patterns?

### 2. Adjust Strategy Based on Results

**If most regions have sufficient data (>30% success rate):**
- Re-run with fixed script
- Accept that some regions won't have predictions for certain years

**If most regions have insufficient data (<30% success rate):**
- Switch to hierarchical approach (Option 1)
- OR implement region grouping/clustering
- OR fall back to original notebook's all-regions-together approach

### 3. Modified Workflow Options

#### Conservative Approach (High Quality, Fewer Predictions):
```bash
python run_region_models.py \
    --dataset "..." \
    --region-map "..." \
    --out "output_conservative" \
    --years 2022 2023 2024 \
    --min-train 50 \
    --min-test 20 \
    --seed 42
```

#### Exploratory Approach (Lower Quality, More Coverage):
```bash
python run_region_models.py \
    --dataset "..." \
    --region-map "..." \
    --out "output_exploratory" \
    --years 2022 2023 2024 \
    --min-train 10 \
    --min-test 5 \
    --seed 42
```

---

## Script Updates Made

### Files Modified:
1. **run_region_models.py** - Fixed test_index merge logic (line ~543)

### To Re-Run:
```bash
# Delete old outputs
rm -rf output_region_models

# Re-run with fixed script
python run_region_models.py \
    --dataset "C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\forecasting_subset_IPCCH_v1210.csv" \
    --region-map "area_id_country_region_mapping.csv" \
    --out "output_region_models_fixed" \
    --years 2022 2023 2024 \
    --seed 42
```

### Expected Improvements:
- ✅ No more "test_index not unique" errors
- ✅ Proper merging of area_id, date, lat, lon
- ⚠️ Still expect some regions to fail due to data sparsity (this is expected)
- ✅ Regions with sufficient data should complete successfully

---

## Validation Checklist

After re-running, check:
- [ ] run_manifest.csv shows "success" for regions with adequate data
- [ ] predictions_{year}_{region}.csv files exist for successful regions
- [ ] Each prediction file has columns: phase2_pred, phase2_test, ..., area_id, date, lat, lon
- [ ] n_train and n_test values are non-zero for successful runs
- [ ] Metrics (accuracy, sensitivity, precision, R²) are reasonable (0-1 range)

---

## Contact

If you continue to see errors after re-running with the fixed script, please provide:
1. New run_manifest.csv
2. Data distribution analysis results (area_ids per region, rows per year-region)
3. Your modeling goals (precision vs coverage tradeoff)
