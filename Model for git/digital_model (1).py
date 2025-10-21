# models/newbies.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import catboost as cb

__all__ = ["train_country_newbies_model"]

def train_country_newbies_model(
    df: pd.DataFrame,
    *,
    # ====== таргет ======
    target_column: Optional[str] = None,                 # явный таргет (например, 'vertical_newbies_delivery_organic')
    target_fallback_contains: Tuple[str, str] = ("newbies", "delivery"),
    target_preferred: str = "vertical_newbies_delivery_paid",
    drop_other_newbies: bool = True,                     # дропнуть все прочие *newbies* до построения фич
    # ====== временной сплит ======
    seed: int = 42,
    holdout_days: int = 30,
    last_days_exclude: int = 8,                          # отрезаем самые свежие «грязные» дни
    # ====== фичи ======
    half_life_days: float = 60.0,                        # time-decay
    top_n_countries: int = 15,
    lag_days: Tuple[int, ...] = (1, 7, 14),
    roll_windows: Tuple[int, ...] = (7, 14),
    roll_min_periods: Optional[Dict[int, int]] = None,   # по умолчанию {7:3, 14:7}
    drop_raw_cost_prefix: str = "cost_",
    keep_cost_prefix: str = "cost_modelled_",
    use_monotone_for_costs: bool = True,                 # бинарный флаг монотонности для cost-фич
    # ====== catboost ======
    catboost_params: Optional[Dict] = None,
) -> Dict[str, object]:
    """
    Обучает CatBoostRegressor для предсказания ежедневных новичков по странам.

    Особенности:
      - Явный выбор таргета через `target_column`; при `drop_other_newbies=True` остальные *newbies* удаляются ДО фичей.
      - Anti-leak: роллы по shift(1), лаги только из прошлого.
      - Отбор топ-стран и time-weights считаются ТОЛЬКО по train-окну.
      - Сплит train/holdout без пересечений.
    """
    np.random.seed(seed)
    if roll_min_periods is None:
        roll_min_periods = {7: 3, 14: 7}

    # ===== 0) базовая подготовка =====
    model_data = df.copy()
    model_data["day"] = pd.to_datetime(model_data["day"])
    model_data["month"] = pd.to_datetime(model_data["month"])

    model_data["dow"] = model_data["day"].dt.weekday.astype("int8")
    model_data["month_num"] = model_data["month"].dt.month.astype("int8")
    model_data["is_weekend"] = model_data["dow"].isin([5, 6]).astype("int8")

    if "country_id" in model_data.columns:
        model_data.dropna(subset=["country_id"], inplace=True)
        model_data.drop(columns=["country_id"], inplace=True)

    # убрать сырые cost_* и оставить только cost_modelled_*
    cols_to_drop = [c for c in model_data.columns
                    if c.startswith(drop_raw_cost_prefix) and not c.startswith(keep_cost_prefix)]
    model_data.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # ===== 1) таргет =====
    if target_column is not None:
        if target_column not in model_data.columns:
            raise KeyError(f"target_column='{target_column}' не найден")
        TARGET = target_column
    else:
        if target_preferred in model_data.columns:
            TARGET = target_preferred
        else:
            cand = [c for c in model_data.columns
                    if (target_fallback_contains[0] in c) and (target_fallback_contains[1] in c)]
            def _score(name: str) -> int:
                s = 0
                s += 100 if "paid" in name else 0
                s += 50 if "vertical" in name else 0
                s += 10 if name.startswith("vertical_newbies_delivery") else 0
                return s
            if not cand:
                raise KeyError("Не найден таргет: нет колонок с 'newbies' и 'delivery'")
            TARGET = sorted(cand, key=_score, reverse=True)[0]

    if drop_other_newbies:
        newbies_others = [c for c in model_data.columns if ("newbies" in c) and (c != TARGET)]
        model_data.drop(columns=newbies_others, inplace=True, errors="ignore")

    # ===== 2) cost_total и сортировка =====
    model_data["cost_total"] = (
        model_data.get(f"{keep_cost_prefix}courier", 0)
        + model_data.get(f"{keep_cost_prefix}rh_driver", 0)
        + model_data.get(f"{keep_cost_prefix}rh_passenger", 0)
    )
    model_data.sort_values(["country_name", "day"], inplace=True)
    model_data["is_after_horiz"] = (model_data["month"].dt.month >= 5).astype("int8")

    # ===== 3) фиксируем окна (до отбора стран и весов) =====
    last_date = model_data["day"].max() - pd.Timedelta(days=last_days_exclude)
    holdout_start = last_date - pd.Timedelta(days=holdout_days - 1)
    train_last_date = holdout_start - pd.Timedelta(days=1)   # последняя дата train после установки окон
    train_mask_global = model_data["day"] < holdout_start     # используется для отбора топ-стран

    # ===== 4) лаги/роллы cost (anti-leak) =====
    costs = [c for c in model_data.columns if c.startswith(keep_cost_prefix)] + ["cost_total"]

    lag_frames, roll_frames = [], []
    # строим лаги/роллы, не кэшируя grp — так безопаснее, если далее будет concat
    for c in costs:
        s = model_data.groupby("country_name", group_keys=False)[c]
        for L in lag_days:
            lag_frames.append(s.shift(L).rename(f"{c}_lag{L}"))
        s_shifted = s.shift(1)  # anti-leak для роллов
        for w in roll_windows:
            roll_frames.append(
                s_shifted.rolling(w, min_periods=roll_min_periods.get(w, 1)).mean().rename(f"{c}_roll{w}")
            )

    if lag_frames:
        model_data = pd.concat([model_data, pd.concat(lag_frames, axis=1)], axis=1)
    if roll_frames:
        model_data = pd.concat([model_data, pd.concat(roll_frames, axis=1)], axis=1)

    # убрать возможные дубликаты колонок
    model_data = model_data.loc[:, ~pd.Index(model_data.columns).duplicated(keep="last")].copy()

    # >>> ВАЖНО: после concat пересоздаём groupby <<<
    grp = model_data.groupby("country_name", group_keys=False)

    # ===== 5) эффективность (cpi/eff) + лаги/роллы =====
    eps = 1e-6
    model_data["cpi_raw"] = model_data["cost_total"] / (model_data[TARGET] + eps)
    model_data["eff_raw"] = model_data[TARGET] / (model_data["cost_total"] + eps)
    model_data["cpi_lag1"] = grp["cpi_raw"].shift(1)
    model_data["eff_lag1"] = grp["eff_raw"].shift(1)
    model_data["cpi_roll7"] = grp["cpi_raw"].shift(1).rolling(7, min_periods=3).mean()
    model_data["eff_roll7"] = grp["eff_raw"].shift(1).rolling(7, min_periods=3).mean()
    model_data["log_cpi_roll7"] = np.log1p(model_data["cpi_roll7"])
    model_data["log_eff_roll7"] = np.log1p(model_data["eff_roll7"])

    # даункаст
    for c in model_data.select_dtypes(include=["float64"]).columns:
        model_data[c] = model_data[c].astype("float32")
    for c in model_data.select_dtypes(include=["int64"]).columns:
        model_data[c] = model_data[c].astype("int32")

    # ===== 6) фичи =====
    categorical_features = ["country_name", "month_num", "dow"]
    numerical_features = [c for c in model_data.columns if c not in categorical_features + [TARGET, "day", "month"]]
    feature_columns = categorical_features + numerical_features

    # ===== 7) топ-страны только по train-окну сырого df =====
    best_countries = (
        model_data.loc[train_mask_global]
                  .groupby("country_name", as_index=False)[TARGET].sum()
                  .sort_values(TARGET, ascending=False)
                  .head(top_n_countries)
    )
    train_countries = list(best_countries["country_name"])

    # оставляем выбранные страны и даты <= last_date (чтобы тест не захватывал будущее)
    df_model = model_data[(model_data["country_name"].isin(train_countries)) &
                          (model_data["day"] <= last_date)].copy()

    # ===== 8) time-decay только от max train-даты по стране =====
    train_mask = df_model["day"] < holdout_start
    test_mask  = (df_model["day"] >= holdout_start) & (df_model["day"] <= last_date)

    country_max_train = df_model.loc[train_mask].groupby("country_name")["day"].max()
    df_model["country_max_train"] = df_model["country_name"].map(country_max_train)

    # подстраховка (если у страны ноль строк в train)
    fallback_map = (df_model[df_model["day"] < holdout_start]
                    .groupby("country_name")["day"].max())
    df_model["country_max_train"] = df_model["country_max_train"].fillna(
        df_model["country_name"].map(fallback_map)
    )

    age_days = (df_model["country_max_train"] - df_model["day"]).dt.days.clip(lower=0)
    time_weight = (0.5 ** (age_days / half_life_days)).clip(lower=0.05)
    df_model["time_weight"] = (time_weight / time_weight.mean()).astype("float32")

    # финальная чистка NaN
    df_model = df_model.dropna(subset=feature_columns + [TARGET]).copy()
    train_mask = df_model["day"] < holdout_start
    test_mask  = (df_model["day"] >= holdout_start) & (df_model["day"] <= last_date)

    # ===== 9) сплит и обучение =====
    X_train = df_model.loc[train_mask, feature_columns]
    Y_train = df_model.loc[train_mask, TARGET].astype("float32")
    W_train = df_model.loc[train_mask, "time_weight"]

    X_test  = df_model.loc[test_mask, feature_columns]
    Y_test  = df_model.loc[test_mask, TARGET].astype("float32")

    cat_index = [feature_columns.index(c) for c in categorical_features]

    monotone_constraints = None
    if use_monotone_for_costs:
        def _is_cost_feat(name: str) -> bool:
            return name.startswith(keep_cost_prefix) or name.startswith("cost_total")
        mono_names = [c for c in feature_columns if _is_cost_feat(c)]
        monotone_constraints = {feature_columns.index(c): +1 for c in mono_names if c in feature_columns}

    default_params = dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=3000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=6.0,
        random_seed=seed,
        od_type="Iter",
        od_wait=200,
        verbose=False,
    )
    if monotone_constraints is not None:
        default_params["monotone_constraints"] = monotone_constraints
    if catboost_params:
        default_params.update(catboost_params)

    train_pool = cb.Pool(X_train, Y_train, cat_features=cat_index, weight=W_train)
    eval_pool  = cb.Pool(X_test,  Y_test,  cat_features=cat_index)

    model = cb.CatBoostRegressor(**default_params)
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True, verbose=False)

    # ===== 10) holdout-оценка =====
    y_pred = model.predict(eval_pool).astype("float32")
    eval_rows_df = df_model.loc[X_test.index, ["day", "country_name", TARGET]].copy()
    eval_rows_df.rename(columns={TARGET: "actual"}, inplace=True)
    eval_rows_df["y_pred"] = y_pred
    eval_rows_df["residual"] = eval_rows_df["actual"] - eval_rows_df["y_pred"]
    eval_rows_df["abs_error"] = eval_rows_df["residual"].abs()
    eps_m = 1e-9
    eval_rows_df["mape_row_pct"]  = 100 * eval_rows_df["abs_error"] / (eval_rows_df["actual"].abs() + eps_m)
    eval_rows_df["smape_row_pct"] = 100 * (2 * eval_rows_df["abs_error"] /
                                           (eval_rows_df["actual"].abs() + eval_rows_df["y_pred"].abs() + eps_m))

    metrics_overall_df = pd.DataFrame([{
        "rmse": float(np.sqrt((eval_rows_df["residual"] ** 2).mean())),
        "mae": float(eval_rows_df["abs_error"].mean()),
        "wmape_pct": float(100 * eval_rows_df["abs_error"].sum() / (eval_rows_df["actual"].abs().sum() + eps_m)),
        "rows": int(len(eval_rows_df)),
        "unique_days": int(eval_rows_df["day"].nunique()),
        "unique_countries": int(eval_rows_df["country_name"].nunique()),
        "holdout_start": pd.to_datetime(holdout_start).date().isoformat(),
        "last_date": pd.to_datetime(last_date).date().isoformat(),
        "train_last_date": pd.to_datetime(train_last_date).date().isoformat(),  # корректная train_last_date
    }])

    fi = model.get_feature_importance(train_pool)
    feature_importance_df = (
        pd.DataFrame({"feature": feature_columns, "importance": fi})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": model,
        "metrics_overall": metrics_overall_df,
        "eval_rows": eval_rows_df,
        "feature_importance": feature_importance_df,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "cat_index": cat_index,
        "df_model": df_model,
        "holdout_start": holdout_start,
        "last_date": last_date,
        "train_last_date": train_last_date,
    }
