# Digital Marketing Model for Delivery Vertical Newbies Prediction

## Overview

This model predicts the number of new users (newbies) in the Delivery vertical based on marketing spend by country. It uses CatBoost regressor with temporal split and anti-leak features.

## Project Structure

- `digital_data_set_querry.SQL` - BigQuery SQL for data extraction
- `digital_model.py` - main model training module
- `elast_simulation.py` - budget elasticity simulation module
- `model_run_notebook.ipynb` - usage example notebook
- `__init__.py` - package initialization

## Data

### Data Source (SQL)

Extracts from BigQuery for 2025:
- **Costs by source**: courier, rh_driver, rh_passenger (actual and modelled)
- **Vertical newbies**: new Delivery users (paid/organic) from SQUIRREL system
- **Grouping**: day × country

## digital_model.py Module

### Main Function: `train_country_newbies_model()`

Trains newbies prediction model with data leakage protection.

#### Key Parameters

**Target:**
- `target_column` - explicit target specification (e.g., `'vertical_newbies_delivery_paid'`)
- `target_preferred` - default preferred target
- `drop_other_newbies` - remove other newbies columns before feature engineering

**Temporal Split:**
- `holdout_days=30` - holdout size in days
- `last_days_exclude=8` - exclude last N days (dirty data)
- `seed=42` - random seed

**Features:**
- `half_life_days=60.0` - half-life period for time-decay weights
- `top_n_countries=15` - top N countries by target (selected from train only)
- `lag_days=(1, 7, 14)` - lags for cost features
- `roll_windows=(7, 14)` - rolling window sizes
- `use_monotone_for_costs=True` - monotone constraints for cost features

**CatBoost:**
- `catboost_params` - model parameters (iterations, learning_rate, depth, etc.)

#### Return Value

Dictionary with keys:
- `model` - trained CatBoostRegressor
- `metrics_overall` - holdout metrics (RMSE, MAE, WMAPE)
- `eval_rows` - detailed predictions for holdout rows
- `feature_importance` - feature importance
- `feature_columns` - list of all features
- `categorical_features` - list of categorical features
- `df_model` - prepared dataframe
- `holdout_start`, `last_date`, `train_last_date` - temporal boundaries

#### Implementation Features

1. **Anti-leak**: all lags and rolling windows use `shift(1)` to prevent future leakage
2. **Time-decay weights**: more recent data gets higher weight during training
3. **Top countries**: selected only from train window
4. **Cost features**: works with `cost_modelled_*` columns, raw `cost_*` are removed
5. **Efficiency**: automatically creates CPI and efficiency features with lags

## elast_simulation.py Module

### Main Function: `simulate_budget_elasticity()`

Simulates prediction changes with budget changes (counterfactual analysis).

#### Parameters

- `model` - trained model
- `df_model` - data dataframe
- `feature_columns` - feature list
- `target_col` - target name
- `start_date`, `end_date` - simulation window
- `budg_down=0.8` - multiplier for budget decrease (-20%)
- `budg_up=1.2` - multiplier for budget increase (+20%)
- `efficiency_mode="safe"` - efficiency mode (safe = uses proxy from base data)

#### Return Value

DataFrame by country with columns:
- `actual` - actual mean target value
- `base_pred` - base model prediction
- `avg_cost` - average cost
- `pred_budg_down/up` - predictions with budget changes
- `delta_down/up` - absolute prediction change
- `elasticity_down/up` - elasticity (relative target change / relative budget change)

#### Helper Functions

- `_apply_budget_counterfactual()` - applies multiplier to cost columns in given window
- `_recompute_time_blocks()` - recalculates lags and rolling windows after cost changes
- `_ensure_features()` - ensures all features are present for prediction

## Usage

### Example 1: Train model for paid newbies

```python
from digital_model import train_country_newbies_model
import pandas as pd

# Load data from BigQuery
df = pd.read_gbq("SELECT * FROM ...")

# Train model
result = train_country_newbies_model(
    df,
    target_column='vertical_newbies_delivery_paid',
    seed=42,
    holdout_days=30,
    half_life_days=60,
    top_n_countries=15,
    catboost_params={"iterations": 3000, "learning_rate": 0.05}
)

# View metrics
print(result['metrics_overall'])
print(result['feature_importance'])
```

### Example 2: Elasticity simulation

```python
from elast_simulation import simulate_budget_elasticity

# Simulate ±10% budget change
elast_df = simulate_budget_elasticity(
    model=result["model"],
    df_model=result["df_model"],
    feature_columns=result["feature_columns"],
    categorical_features=result["categorical_features"],
    target_col="vertical_newbies_delivery_paid",
    start_date=result["holdout_start"],
    end_date=result["last_date"],
    budg_down=0.90,
    budg_up=1.10,
    efficiency_mode="safe"
)

# Sort countries by elasticity
print(elast_df.sort_values('elasticity_up', ascending=False))
```

### Example 3: Model for total newbies (paid + organic)

```python
# Create total target
df['newbies_total'] = df['vertical_newbies_delivery_paid'] + df['vertical_newbies_delivery_organic']

# Train model
result_total = train_country_newbies_model(
    df,
    target_column='newbies_total',
    seed=42,
    holdout_days=30,
    top_n_countries=15
)
```

## Quality Metrics

Model is evaluated on holdout (last 30 days):
- **RMSE** - root mean squared error
- **MAE** - mean absolute error
- **WMAPE** - weighted mean absolute percentage error

Typical values for paid newbies:
- RMSE: ~44
- MAE: ~24
- WMAPE: ~5.6%

## Dependencies

```
pandas
numpy
catboost
pandas-gbq
scikit-learn
```

## Notes

1. Model requires minimum 3-7 days of history for rolling windows (parameter `roll_min_periods`)
2. Time-decay weights are calculated relative to last train date per country
3. Monotone constraints for costs ensure budget increase doesn't decrease prediction
4. In safe simulation mode, efficiency is "frozen" via proxy (lag/roll) to prevent circular dependencies

## Next Steps

### 1. Fix Paid Newbies Definition
**Issue**: Currently paid newbies include "no source" newbies, which should be excluded.

**Action**: Update SQL query (`digital_data_set_querry.SQL`) to filter out no-source attributions:
- Add explicit filter: `WHERE media_source IS NOT NULL AND LOWER(media_source) NOT LIKE '%organic%' AND media_source != 'no_source'`
- Validate that paid metrics only include properly attributed marketing channels

### 2. Implement Planned Cost ETL
**Goal**: Automate ingestion of planned marketing costs from digital marketing team.

**Action**:
- Create ETL pipeline to fetch planned costs (by country, source, date)
- Store in BigQuery table (e.g., `automation.tbl_planned_marketing_costs`)
- Join with actual costs in main query
- Schedule daily/weekly updates

### 3. Cost Forecasting via Plan Utilization
**Goal**: Predict future costs based on plan utilization rate to enable 1-month forward forecasting.

**Action**:
- Create `cost_forecasting.py` module with function `forecast_costs_from_plan()`
- Calculate historical plan utilization rate: `actual_cost / planned_cost`
- Features:
  - Rolling utilization rate by country/source
  - Day-of-month effects
  - Seasonality patterns
- Output: forecasted costs for next 30 days
- Enable operational tracking of overspends and forecast accuracy

**API Example**:
```python
def forecast_costs_from_plan(
    df_actual: pd.DataFrame,      # historical actual costs
    df_plan: pd.DataFrame,         # planned costs (future dates)
    country_col: str = "country_name",
    forecast_days: int = 30
) -> pd.DataFrame:
    # Returns: forecasted costs by day × country × source
    pass
```

### 4. Overspending Alerting System
**Goal**: Automated alerts for countries with budget overspend.

**Action**:
- Create `alerting.py` module with function `detect_overspend()`
- Thresholds:
  - Daily overspend: >15% above plan
  - Weekly overspend: >10% above plan
  - Monthly overspend: >5% above plan
- Alert channels: email, Slack, dashboard
- Include: country, source, overspend amount, %, trend

**API Example**:
```python
def detect_overspend(
    df_actual: pd.DataFrame,
    df_plan: pd.DataFrame,
    thresholds: dict = {"daily": 0.15, "weekly": 0.10, "monthly": 0.05}
) -> pd.DataFrame:
    # Returns: countries with overspend, severity, recommended actions
    pass
```

**Priority**: 1 → 2 → 3 → 4
