# models/elasticity.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import catboost as cb

# -------- helpers --------

def _recompute_time_blocks(
    df: pd.DataFrame,
    *,
    keep_cost_prefix: str = "cost_modelled_",
    roll_windows: Tuple[int, ...] = (7, 14),
    lag_days: Tuple[int, ...] = (1, 7, 14),
    roll_min_periods: Dict[int, int] = None,
    country_col: str = "country_name",
) -> pd.DataFrame:
    if roll_min_periods is None:
        roll_min_periods = {7: 3, 14: 7}
    df = df.sort_values([country_col, "day"]).copy()
    grp = df.groupby(country_col, group_keys=False)

    costs = [c for c in df.columns if c.startswith(keep_cost_prefix)] + ["cost_total"]

    lag_frames = []
    for c in costs:
        s = grp[c]
        for L in lag_days:
            lag_frames.append(s.shift(L).rename(f"{c}_lag{L}"))
    lag_block = pd.concat(lag_frames, axis=1) if lag_frames else pd.DataFrame(index=df.index)

    roll_frames = []
    for c in costs:
        s_shifted = grp[c].shift(1)
        for w in roll_windows:
            roll_frames.append(
                s_shifted.rolling(w, min_periods=roll_min_periods.get(w, 1)).mean().rename(f"{c}_roll{w}")
            )
    roll_block = pd.concat(roll_frames, axis=1) if roll_frames else pd.DataFrame(index=df.index)

    df = pd.concat([
        df.drop(
            columns=[col for col in df.columns
                     if col.endswith("_lag1") or col.endswith("_lag7") or col.endswith("_lag14")
                     or col.endswith("_roll7") or col.endswith("_roll14")],
            errors="ignore"
        ),
        lag_block, roll_block
    ], axis=1)

    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="last")].copy()
    return df


def _apply_budget_counterfactual(
    base_df: pd.DataFrame,
    *,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    multiplier: float,
    keep_cost_prefix: str = "cost_modelled_",
    country_col: str = "country_name",
) -> pd.DataFrame:
    df = base_df.copy()
    mask = True
    if start_date is not None:
        mask = mask & (df["day"] >= pd.to_datetime(start_date))
    if end_date is not None:
        mask = mask & (df["day"] <= pd.to_datetime(end_date))

    cost_cols = [c for c in df.columns if c.startswith(keep_cost_prefix)]
    has_components = len(cost_cols) > 0

    if has_components:
        df.loc[mask, cost_cols] = df.loc[mask, cost_cols] * multiplier
        df["cost_total"] = (
            df.get(f"{keep_cost_prefix}courier", 0)
            + df.get(f"{keep_cost_prefix}rh_driver", 0)
            + df.get(f"{keep_cost_prefix}rh_passenger", 0)
        )
    else:
        df.loc[mask, "cost_total"] = df.loc[mask, "cost_total"] * multiplier

    return df


_EFF_COLS = [
    "cpi_raw", "eff_raw",
    "cpi_lag1", "eff_lag1",
    "cpi_roll7", "eff_roll7",
    "log_cpi_roll7", "log_eff_roll7",
]

def _ensure_features(
    df_sim: pd.DataFrame,
    df_base: pd.DataFrame,
    feature_columns: List[str],
    *,
    efficiency_mode: str = "safe"
) -> pd.DataFrame:
    """
    Гарантирует наличие всех feature_columns в df_sim.
    В safe-режиме эффективность заполняется прокси из базового df (lag/roll),
    а 'cpi_raw'/'eff_raw' подменяются на эти прокси (без использования текущего TARGET).
    Остальные отсутствующие фичи копируются из базового df по индексу.
    """
    df = df_sim.copy()

    # 1) сначала создадим заглушки для всех фич, которых нет (из базового df)
    for col in feature_columns:
        if col not in df.columns:
            if col in df_base.columns:
                df[col] = df_base[col]
            else:
                # если совсем нет в базе (редко), создаём NaN — CatBoost переживёт (при обучении он видел nan?)
                df[col] = np.nan

    if efficiency_mode == "safe":
        # подмена эффективности на прокси из базового df
        # если есть lag1 — используем его; иначе roll7; иначе оставляем как есть (уже скопировали выше)
        if "cpi_raw" in feature_columns:
            if "cpi_lag1" in df_base.columns:
                df["cpi_raw"] = df_base["cpi_lag1"]
            elif "cpi_roll7" in df_base.columns:
                df["cpi_raw"] = df_base["cpi_roll7"]
        if "eff_raw" in feature_columns:
            if "eff_lag1" in df_base.columns:
                df["eff_raw"] = df_base["eff_lag1"]
            elif "eff_roll7" in df_base.columns:
                df["eff_raw"] = df_base["eff_roll7"]

        # логарифмы — тоже из базового df (они от прокси во время тренировки)
        if "log_cpi_roll7" in feature_columns and "log_cpi_roll7" in df_base.columns:
            df["log_cpi_roll7"] = df_base["log_cpi_roll7"]
        if "log_eff_roll7" in feature_columns and "log_eff_roll7" in df_base.columns:
            df["log_eff_roll7"] = df_base["log_eff_roll7"]

        # убедимся, что lag/roll эффективности присутствуют (если требуются)
        for col in ["cpi_lag1", "eff_lag1", "cpi_roll7", "eff_roll7"]:
            if (col in feature_columns) and (col in df_base.columns):
                df[col] = df_base[col]

    # порядок колонок не меняем — Pool возьмёт их по имени
    return df

# -------- main API --------

def simulate_budget_elasticity(
    *,
    model: cb.CatBoostRegressor,
    df_model: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    country_col: str = "country_name",
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    budg_down: float = 0.8,
    budg_up: float = 1.2,
    keep_cost_prefix: str = "cost_modelled_",
    roll_windows: Tuple[int, ...] = (7, 14),
    lag_days: Tuple[int, ...] = (1, 7, 14),
    roll_min_periods: Dict[int, int] = None,
    efficiency_mode: str = "safe",
    categorical_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    if roll_min_periods is None:
        roll_min_periods = {7: 3, 14: 7}

    # 0) База: срез окна для отчётной части (и для индексов)
    work_df = df_model.copy()
    if start_date is not None:
        work_df = work_df[work_df["day"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        work_df = work_df[work_df["day"] <= pd.to_datetime(end_date)]

    # 1) Базовый предикт на исходных фичах
    base_X = df_model.loc[work_df.index, feature_columns]
    base_pool = cb.Pool(
        base_X,
        cat_features=None if categorical_features is None
        else [feature_columns.index(c) for c in categorical_features]
    )
    base_pred = model.predict(base_pool).astype("float32")

    base_df = work_df[[country_col, "day", target_col, "cost_total"]].copy()
    base_df["base_pred"] = base_pred

    # подготовим df_base_feats для подстановок (ровно те же строки/индексы)
    df_base_feats = df_model.loc[work_df.index, :]

    # 2) budg_down
    down_df_full = _apply_budget_counterfactual(
        df_model, start_date=start_date, end_date=end_date,
        multiplier=budg_down, keep_cost_prefix=keep_cost_prefix, country_col=country_col
    )
    down_df_full = _recompute_time_blocks(
        down_df_full, keep_cost_prefix=keep_cost_prefix,
        roll_windows=roll_windows, lag_days=lag_days,
        roll_min_periods=roll_min_periods, country_col=country_col
    )
    # гарантируем все фичи + безопасная эффективность
    down_df_full = _ensure_features(
        down_df_full, df_base_feats, feature_columns, efficiency_mode=efficiency_mode
    )
    down_X = down_df_full.loc[work_df.index, feature_columns]
    down_pool = cb.Pool(
        down_X,
        cat_features=None if categorical_features is None
        else [feature_columns.index(c) for c in categorical_features]
    )
    pred_down = model.predict(down_pool).astype("float32")

    # 3) budg_up
    up_df_full = _apply_budget_counterfactual(
        df_model, start_date=start_date, end_date=end_date,
        multiplier=budg_up, keep_cost_prefix=keep_cost_prefix, country_col=country_col
    )
    up_df_full = _recompute_time_blocks(
        up_df_full, keep_cost_prefix=keep_cost_prefix,
        roll_windows=roll_windows, lag_days=lag_days,
        roll_min_periods=roll_min_periods, country_col=country_col
    )
    up_df_full = _ensure_features(
        up_df_full, df_base_feats, feature_columns, efficiency_mode=efficiency_mode
    )
    up_X = up_df_full.loc[work_df.index, feature_columns]
    up_pool = cb.Pool(
        up_X,
        cat_features=None if categorical_features is None
        else [feature_columns.index(c) for c in categorical_features]
    )
    pred_up = model.predict(up_pool).astype("float32")

    # 4) сборка таблицы по странам
    out = base_df[[country_col, "day", target_col, "cost_total", "base_pred"]].copy()
    out.rename(columns={target_col: "actual"}, inplace=True)
    out["pred_budg_down"] = pred_down
    out["pred_budg_up"] = pred_up

    agg = (
        out.groupby(country_col, as_index=False)
           .agg(base_pred=("base_pred", "mean"),
                actual=("actual", "mean"),
                pred_budg_down=("pred_budg_down", "mean"),
                pred_budg_up=("pred_budg_up", "mean"),
                avg_cost=("cost_total", "mean"))
    )

    delta_cost_down = budg_down - 1.0
    delta_cost_up   = budg_up   - 1.0

    agg["delta_down"] = agg["pred_budg_down"] - agg["base_pred"]
    agg["delta_up"]   = agg["pred_budg_up"]   - agg["base_pred"]

    eps = 1e-9
    agg["elasticity_down"] = ((agg["pred_budg_down"] - agg["base_pred"]) / (agg["base_pred"] + eps)) / (delta_cost_down + eps)
    agg["elasticity_up"]   = ((agg["pred_budg_up"]   - agg["base_pred"]) / (agg["base_pred"] + eps)) / (delta_cost_up + eps)

    cols = [country_col, "actual", "base_pred", "avg_cost",
            "pred_budg_down", "delta_down", "elasticity_down",
            "pred_budg_up",   "delta_up",   "elasticity_up"]
    return agg[cols]
