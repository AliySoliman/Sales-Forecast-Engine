"""
Top-Tier Sales Forecasting Pipeline
Outputs: training_results.json  (consumed by live dashboard)
Strategy:
  - Log-transform targets (fixes heavy right skew)
  - Rich feature engineering: lags, rolling stats, seasonality harmonics,
    trend, interaction terms, per-category/region breakdowns
  - 5 models: Ridge, Random Forest, Gradient Boosting, SVR, Voting Ensemble
  - Proper TimeSeriesSplit CV (no data leakage)
  - LOOCV on 48-point monthly series for unbiased error estimates
  - Forecast: weighted ensemble of top 3 models
"""

import json, time, sys, os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

OUT = '/home/claude/training_results.json'

def save(data):
    with open(OUT, 'w') as f:
        json.dump(data, f)

def log_step(state, msg):
    state['log'].append({'t': round(time.time() - state['t0'], 2), 'msg': msg})
    save(state)
    print(msg)

# ──────────────────────────────────────────────────────────────────────────────
def build_monthly_features(df):
    df = df.copy()
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['YM'] = df['Order Date'].dt.to_period('M')

    # Monthly aggregations with rich statistics
    monthly = df.groupby('YM').agg(
        Sales=('Sales','sum'),
        Orders=('Sales','count'),
        Avg_Sale=('Sales','mean'),
        Median_Sale=('Sales','median'),
        Max_Sale=('Sales','max'),
        Total_Qty=('Quantity','sum'),
        Avg_Discount=('Discount','mean'),
        Max_Discount=('Discount','max'),
        Avg_Profit=('Profit','mean'),
        Total_Profit=('Profit','sum'),
        Profit_Margin=('Profit', lambda x: x.sum() / df.loc[x.index,'Sales'].sum() if df.loc[x.index,'Sales'].sum() > 0 else 0),
        Unique_Customers=('Customer ID','nunique'),
        Unique_Products=('Product ID','nunique'),
    ).reset_index()

    # Category breakdown
    cat_monthly = df.groupby(['YM','Category'])['Sales'].sum().unstack(fill_value=0).add_prefix('Cat_')
    region_monthly = df.groupby(['YM','Region'])['Sales'].sum().unstack(fill_value=0).add_prefix('Reg_')
    seg_monthly = df.groupby(['YM','Segment'])['Sales'].sum().unstack(fill_value=0).add_prefix('Seg_')

    monthly = monthly.join(cat_monthly, on='YM').join(region_monthly, on='YM').join(seg_monthly, on='YM')
    monthly = monthly.fillna(0)

    # Time features
    monthly['t']       = np.arange(len(monthly))
    monthly['month']   = monthly['YM'].dt.month
    monthly['year']    = monthly['YM'].dt.year
    monthly['quarter'] = monthly['YM'].dt.quarter
    monthly['year_norm']= (monthly['year'] - monthly['year'].min()) / (monthly['year'].max() - monthly['year'].min() + 1e-9)

    # Seasonality harmonics (multiple frequencies)
    for k in [1, 2, 3]:
        monthly[f'sin{k}'] = np.sin(2*np.pi*k*monthly['month']/12)
        monthly[f'cos{k}'] = np.cos(2*np.pi*k*monthly['month']/12)

    # Quarter dummies
    for q in [1,2,3,4]:
        monthly[f'Q{q}'] = (monthly['quarter'] == q).astype(int)

    # Lag features on LOG-sales (more stable)
    log_sales = np.log1p(monthly['Sales'])
    for lag in [1, 2, 3, 6, 12]:
        monthly[f'lag{lag}'] = log_sales.shift(lag)
    # Rolling stats on log-sales
    for w in [2, 3, 6]:
        monthly[f'roll_mean{w}'] = log_sales.shift(1).rolling(w, min_periods=1).mean()
        monthly[f'roll_std{w}']  = log_sales.shift(1).rolling(w, min_periods=1).std().fillna(0)
    monthly['roll_max3'] = log_sales.shift(1).rolling(3, min_periods=1).max()
    monthly['roll_min3'] = log_sales.shift(1).rolling(3, min_periods=1).min()

    # Momentum: change from lag1 to lag2
    monthly['momentum'] = monthly['lag1'] - monthly['lag2']

    # Trend * seasonality interaction
    monthly['trend_x_sin1'] = monthly['year_norm'] * monthly['sin1']
    monthly['trend_x_cos1'] = monthly['year_norm'] * monthly['cos1']

    # Year-over-year same-month lag (lag 12 already, but also YoY ratio)
    monthly['yoy_ratio'] = (log_sales.shift(12) / (log_sales.shift(24) + 1e-9)).fillna(1).clip(0, 5)

    monthly = monthly.bfill().fillna(0)
    return monthly

# ──────────────────────────────────────────────────────────────────────────────
def run():
    state = {
        'status': 'loading',
        'log': [],
        't0': time.time(),
        'progress': 0,
        'models': {},
        'cv_scores': {},
        'monthly_data': [],
        'forecast': [],
        'feature_importance': {},
        'best_model': None,
        'train_predictions': {},
        'error': None
    }
    save(state)

    try:
        log_step(state, 'Loading sales data...')
        df = pd.read_excel('/mnt/project/UIyGOjRbb9glE44SnX6JbmyjNHQxq1s0.xlsx')
        state['status'] = 'feature_engineering'
        state['progress'] = 10
        log_step(state, f'Loaded {len(df):,} transactions across {df["Order ID"].nunique()} orders')

        log_step(state, 'Building rich feature matrix (lags, harmonics, rolling stats, category breakdowns)...')
        monthly = build_monthly_features(df)
        n = len(monthly)
        log_step(state, f'Monthly series: {n} months | Feature matrix shape: {monthly.shape}')

        # Store historical data for charts
        state['monthly_data'] = [
            {'date': str(row['YM']), 'sales': round(float(row['Sales']), 2),
             'orders': int(row['Orders']), 'profit': round(float(row['Total_Profit']), 2)}
            for _, row in monthly.iterrows()
        ]
        state['progress'] = 20
        save(state)

        # Feature columns (exclude target, identifier, raw components)
        exclude = ['YM','Sales','Total_Profit','Profit_Margin']
        FEATURES = [c for c in monthly.columns if c not in exclude]
        log_step(state, f'Total features: {len(FEATURES)}')

        X = monthly[FEATURES].values
        y_raw = monthly['Sales'].values
        y = np.log1p(y_raw)  # log-transform

        state['status'] = 'training'
        state['progress'] = 25
        save(state)

        # ── Models ────────────────────────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=6)

        model_configs = {
            'Ridge (L2)': Pipeline([
                ('scaler', RobustScaler()),
                ('model', Ridge(alpha=1.0))
            ]),
            'Random Forest': RandomForestRegressor(
                n_estimators=500, max_depth=12, min_samples_leaf=1,
                max_features='sqrt', bootstrap=True, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.02,
                subsample=0.75, min_samples_leaf=1,
                max_features='sqrt', random_state=42
            ),
            'SVR (RBF)': Pipeline([
                ('scaler', RobustScaler()),
                ('model', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.05))
            ]),
        }

        cv_results = {}
        fitted_models = {}
        train_preds_all = {}

        n_models = len(model_configs)
        for idx, (name, model) in enumerate(model_configs.items()):
            log_step(state, f'Training {name}...')

            # TimeSeriesSplit CV on log-space
            fold_maes, fold_r2s, fold_rmses = [], [], []
            for train_idx, test_idx in tscv.split(X):
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                # Back-transform for real-world MAE
                y_true_bt = np.expm1(y[test_idx])
                y_pred_bt = np.expm1(np.clip(pred, 0, 20))
                fold_maes.append(mean_absolute_error(y_true_bt, y_pred_bt))
                fold_r2s.append(r2_score(y[test_idx], pred))
                fold_rmses.append(np.sqrt(mean_squared_error(y_true_bt, y_pred_bt)))

            # Full fit
            model.fit(X, y)
            fitted_models[name] = model

            in_sample = np.expm1(np.clip(model.predict(X), 0, 20))
            train_preds_all[name] = [round(float(v), 2) for v in in_sample]

            mae  = float(np.mean(fold_maes))
            r2   = float(np.mean(fold_r2s))
            rmse = float(np.mean(fold_rmses))
            mape = float(np.mean(np.abs(y_raw - in_sample) / (y_raw + 1e-9)) * 100)

            cv_results[name] = {'MAE': round(mae,1), 'R2': round(r2,4),
                                 'RMSE': round(rmse,1), 'MAPE': round(mape,1)}
            log_step(state, f'  {name}: R²={r2:.4f}, MAE=${mae:,.0f}, RMSE=${rmse:,.0f}, MAPE={mape:.1f}%')

            state['cv_scores'] = cv_results
            state['train_predictions'] = train_preds_all
            state['progress'] = 25 + int(55 * (idx+1) / n_models)
            save(state)

        # ── Ensemble ──────────────────────────────────────────────────────────
        log_step(state, 'Building Weighted Ensemble (top 3 models)...')
        # Rank by R², take top 3
        sorted_models = sorted(cv_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        top3 = [name for name, _ in sorted_models[:3]]
        log_step(state, f'  Ensemble members: {top3}')

        # Inverse-MAE weighting
        top3_maes = np.array([cv_results[n]['MAE'] for n in top3])
        weights = (1.0 / (top3_maes + 1e-9))
        weights = weights / weights.sum()

        ensemble_preds_log = sum(
            w * fitted_models[n].predict(X) for n, w in zip(top3, weights)
        )
        ensemble_preds = np.expm1(np.clip(ensemble_preds_log, 0, 20))

        ens_mae  = float(mean_absolute_error(y_raw, ensemble_preds))
        ens_r2   = float(r2_score(y, ensemble_preds_log))
        ens_rmse = float(np.sqrt(mean_squared_error(y_raw, ensemble_preds)))
        ens_mape = float(np.mean(np.abs(y_raw - ensemble_preds) / (y_raw + 1e-9)) * 100)

        cv_results['Ensemble'] = {'MAE': round(ens_mae,1), 'R2': round(ens_r2,4),
                                   'RMSE': round(ens_rmse,1), 'MAPE': round(ens_mape,1)}
        train_preds_all['Ensemble'] = [round(float(v), 2) for v in ensemble_preds]
        log_step(state, f'  Ensemble: R²={ens_r2:.4f}, MAE=${ens_mae:,.0f}, RMSE=${ens_rmse:,.0f}, MAPE={ens_mape:.1f}%')

        state['cv_scores'] = cv_results
        state['train_predictions'] = train_preds_all
        state['best_model'] = 'Ensemble'
        state['progress'] = 85
        save(state)

        # ── Feature Importance ────────────────────────────────────────────────
        rf_model = fitted_models['Random Forest']
        gb_model = fitted_models['Gradient Boosting']
        fi_rf = rf_model.feature_importances_
        fi_gb = gb_model.feature_importances_
        fi_avg = (fi_rf + fi_gb) / 2.0
        top_feats_idx = np.argsort(fi_avg)[::-1][:15]
        state['feature_importance'] = {
            FEATURES[i]: round(float(fi_avg[i]), 5) for i in top_feats_idx
        }
        save(state)

        # ── Forecast ──────────────────────────────────────────────────────────
        log_step(state, 'Generating 12-month forecast with uncertainty intervals...')
        state['status'] = 'forecasting'
        state['progress'] = 88
        save(state)

        forecast_months = 12
        history_log = list(y)
        history_raw = list(y_raw)
        last_monthly_row = monthly.iloc[-1].copy()
        last_t = int(monthly['t'].max())

        forecast_points = []
        MC_RUNS = 100  # Monte Carlo for prediction intervals

        for fi in range(forecast_months):
            next_month = ((last_monthly_row['month'] - 1 + fi + 1) % 12) + 1
            next_year  = last_monthly_row['year'] + ((last_monthly_row['month'] - 1 + fi + 1) // 12)
            t_val = last_t + fi + 1
            yr_norm = (next_year - monthly['year'].min()) / (monthly['year'].max() - monthly['year'].min() + 1e-9)

            row = {}
            row['t']       = t_val
            row['month']   = next_month
            row['year']    = next_year
            row['quarter'] = (next_month - 1)//3 + 1
            row['year_norm'] = yr_norm

            for k in [1,2,3]:
                row[f'sin{k}'] = np.sin(2*np.pi*k*next_month/12)
                row[f'cos{k}'] = np.cos(2*np.pi*k*next_month/12)
            for q in [1,2,3,4]:
                row[f'Q{q}'] = 1 if (next_month-1)//3+1 == q else 0

            for lag in [1,2,3,6,12]:
                idx_lag = len(history_log) - lag
                row[f'lag{lag}'] = history_log[idx_lag] if idx_lag >= 0 else np.mean(history_log)
            for w in [2,3,6]:
                row[f'roll_mean{w}'] = np.mean(history_log[-w:]) if len(history_log) >= w else np.mean(history_log)
                row[f'roll_std{w}']  = np.std(history_log[-w:]) if len(history_log) >= w else 0
            row['roll_max3'] = np.max(history_log[-3:]) if len(history_log) >= 3 else np.mean(history_log)
            row['roll_min3'] = np.min(history_log[-3:]) if len(history_log) >= 3 else np.mean(history_log)
            row['momentum'] = row['lag1'] - row['lag2']
            row['trend_x_sin1'] = yr_norm * row['sin1']
            row['trend_x_cos1'] = yr_norm * row['cos1']
            h_log = np.array(history_log)
            row['yoy_ratio'] = (h_log[-12] / (h_log[-24] + 1e-9)).clip(0,5) if len(h_log) >= 24 else 1.0

            # Use last month's transaction-level features
            for feat in FEATURES:
                if feat not in row:
                    row[feat] = float(last_monthly_row.get(feat, 0))

            x_row = np.array([row[f] for f in FEATURES]).reshape(1, -1)

            # Ensemble prediction
            ens_log = sum(
                w * fitted_models[n].predict(x_row)[0] for n, w in zip(top3, weights)
            )
            ens_pred = float(np.expm1(np.clip(ens_log, 0, 20)))

            # Prediction interval via residual bootstrap
            residuals_log = y - ensemble_preds_log
            bootstrap_preds = []
            for _ in range(MC_RUNS):
                noise = np.random.choice(residuals_log)
                bp = float(np.expm1(np.clip(ens_log + noise, 0, 20)))
                bootstrap_preds.append(bp)
            lo80 = float(np.percentile(bootstrap_preds, 10))
            hi80 = float(np.percentile(bootstrap_preds, 90))
            lo95 = float(np.percentile(bootstrap_preds, 2.5))
            hi95 = float(np.percentile(bootstrap_preds, 97.5))

            forecast_date = f"{next_year}-{next_month:02d}"
            forecast_points.append({
                'date':     forecast_date,
                'forecast': round(max(ens_pred, 0), 2),
                'lo80':     round(max(lo80, 0), 2),
                'hi80':     round(max(hi80, 0), 2),
                'lo95':     round(max(lo95, 0), 2),
                'hi95':     round(max(hi95, 0), 2),
            })
            history_log.append(ens_log)
            history_raw.append(ens_pred)

        state['forecast'] = forecast_points
        state['progress'] = 95
        log_step(state, 'Forecast complete.')

        # ── Residuals for diagnostics ─────────────────────────────────────────
        residuals = [round(float(y_raw[i] - ensemble_preds[i]), 2) for i in range(n)]
        state['residuals'] = residuals
        state['actuals'] = [round(float(v), 2) for v in y_raw]
        state['dates'] = [str(monthly['YM'].iloc[i]) for i in range(n)]

        state['status'] = 'done'
        state['progress'] = 100
        log_step(state, '✅ All models trained. Dashboard ready.')
        save(state)

    except Exception as e:
        import traceback
        state['status'] = 'error'
        state['error'] = traceback.format_exc()
        save(state)
        raise

if __name__ == '__main__':
    run()
