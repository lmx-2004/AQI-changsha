实验一
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置显示选项
pd.set_option('display.max_columns', None)

# 文件路径
file_path = r"D:\Anaconda\air_pollution_Changsha.xlsx"

# 读取数据（默认第一个工作表）
df_raw = pd.read_excel(file_path)

print("原始数据形状:", df_raw.shape)
print("\n前5行数据：")
df_raw.head()

# 查看列名
print("列名列表：", df_raw.columns.tolist())

# 查看数据类型和非空统计
df_raw.info()

print("缺失值统计：")
df.isnull().sum()

print("重复行数:", df.duplicated().sum())

# 计算日期差（应为1天）
date_diff = df_raw.index.to_series().diff().dt.days
print("日期间隔异常（非1天）的数量：", (date_diff != 1).sum())
print("\n日期间隔分布：")
date_diff.value_counts().head(10)

plt.figure(figsize=(14, 5))
plt.plot(df_raw.index, df_raw['AQI'], linewidth=0.8, color='steelblue')
plt.title('长沙市日均AQI时间序列（2022.7-2026.6）')
plt.xlabel('日期')
plt.ylabel('AQI')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('aqi_timeseries.png', dpi=300)
plt.show()

def get_season(month):
    if month in [3,4,5]:
        return 'Spring'
    elif month in [6,7,8]:
        return 'Summer'
    elif month in [9,10,11]:
        return 'Autumn'
    else:
        return 'Winter'

df_raw['season'] = df_raw.index.month.map(get_season)

plt.figure(figsize=(6,4))
sns.boxplot(x='season', y='AQI', data=df_raw, order=['Spring','Summer','Autumn','Winter'])
plt.title('不同季节AQI分布')
plt.tight_layout()
plt.savefig('aqi_season_box.png', dpi=300)
plt.show()

# 选择主要数值列（注意列名）
feature_cols = ['AQI', 'PM2.5', 'PM10', 'O3', 'SO2', 'CO', 'NO2', 
                'temp', 'wind_speed', 'humidity', 'pressure', 'rain']
# 检查列名是否存在，若不存在则调整
available_cols = [col for col in feature_cols if col in df.columns]
corr = df[available_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('污染物与气象变量的相关性热力图')
plt.tight_layout()
plt.savefig('corr_heatmap.png', dpi=300)
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df['AQI'], bins=30, kde=True)
plt.title('AQI分布直方图')
plt.xlabel('AQI')
plt.ylabel('频数')
plt.tight_layout()
plt.savefig('aqi_hist.png', dpi=300)
plt.show()

print("按季节统计AQI：")
df.groupby('season')['AQI'].describe()

high_aqi = df[df['AQI'] > 200]
print("AQI > 200 的日期及数值：")
high_aqi[['AQI']].sort_values('AQI', ascending=False)

实验二
import xgboost
import lightgbm

print("xgboost version:", xgboost.__version__)
print("lightgbm version:", lightgbm.__version__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置随机种子
np.random.seed(42)

# 1. 加载数据
file_path = r'D:\Anaconda\Changsha_AQI_features_selected(1).csv'
df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
print("原始数据形状:", df.shape)

# 2. 构造特征（仅使用历史信息，避免泄露）
target = 'AQI'
raw_features = ['PM2.5', 'PM10', 'O3', 'aqi_level_code']  # 这些不能直接使用，只用滞后

# 2.1 原始特征的滞后值（1,2,3,7,14天前）
lags = [1, 2, 3, 7, 14]
for col in raw_features:
    for lag in lags:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# 2.2 AQI自身的滞后特征（强预测因子）
for lag in [1, 2, 3, 7]:
    df[f'AQI_lag{lag}'] = df['AQI'].shift(lag)

# 2.3 滚动统计特征（只基于历史AQI）
for window in [7, 14]:
    df[f'AQI_roll_mean_{window}'] = df['AQI'].shift(1).rolling(window, min_periods=1).mean()
    df[f'AQI_roll_std_{window}'] = df['AQI'].shift(1).rolling(window, min_periods=1).std()

# 2.4 时间特征（不依赖目标）
df['month'] = df.index.month
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# 删除缺失值（前14天会因为滞后而缺失）
df_clean = df.dropna()
print("构造特征后数据量:", df_clean.shape)

# 3. 定义特征集（只包含历史信息和时间特征，不含当天原始污染物）
exclude_cols = ['AQI'] + raw_features   # 移除原始污染物列，只保留滞后和滚动
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
X = df_clean[feature_cols]
y = df_clean['AQI']
print("特征数量:", X.shape[1])
print("特征列表:", X.columns.tolist())

# 4. 时序分割（训练集：2025-07-01之前，测试集：之后）
train_cutoff = pd.Timestamp('2025-07-01')
X_train = X[X.index < train_cutoff]
X_test = X[X.index >= train_cutoff]
y_train = y[y.index < train_cutoff]
y_test = y[y.index >= train_cutoff]
print(f"训练集: {len(X_train)} 条，测试集: {len(X_test)} 条")

# 5. 基线模型（持久性）
baseline_pred = np.full(len(y_test), y_train.iloc[-1])
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_mae = mean_absolute_error(y_test, baseline_pred)
print(f"基线模型 RMSE={baseline_rmse:.2f}, MAE={baseline_mae:.2f}")

# 6. 定义时序交叉验证
tscv = TimeSeriesSplit(n_splits=3)

# 7. 随机森林调优与评估
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
print(f"随机森林: RMSE={rf_rmse:.2f}, MAE={rf_mae:.2f}, R2={rf_r2:.4f}")

# 8. XGBoost 调优与评估
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb = XGBRegressor(random_state=42, n_jobs=-1)
xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_r2 = r2_score(y_test, y_pred_xgb)
print(f"XGBoost: RMSE={xgb_rmse:.2f}, MAE={xgb_mae:.2f}, R2={xgb_r2:.4f}")

# 9. LightGBM 调优与评估
lgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.05],
    'num_leaves': [31, 50],
    'subsample': [0.8, 1.0]
}
lgb = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
lgb_grid = GridSearchCV(lgb, lgb_param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
lgb_grid.fit(X_train, y_train)
lgb_best = lgb_grid.best_estimator_
y_pred_lgb = lgb_best.predict(X_test)
lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
lgb_mae = mean_absolute_error(y_test, y_pred_lgb)
lgb_r2 = r2_score(y_test, y_pred_lgb)
print(f"LightGBM: RMSE={lgb_rmse:.2f}, MAE={lgb_mae:.2f}, R2={lgb_r2:.4f}")

# 10. 性能汇总
results = {
    'Baseline': {'RMSE': baseline_rmse, 'MAE': baseline_mae, 'R2': 0},
    'RandomForest': {'RMSE': rf_rmse, 'MAE': rf_mae, 'R2': rf_r2},
    'XGBoost': {'RMSE': xgb_rmse, 'MAE': xgb_mae, 'R2': xgb_r2},
    'LightGBM': {'RMSE': lgb_rmse, 'MAE': lgb_mae, 'R2': lgb_r2}
}
df_results = pd.DataFrame(results).T.sort_values('RMSE')
print("\n各模型性能对比：")
print(df_results.round(2))

# 11. 最佳模型预测可视化 (以RandomForest为例)
plt.figure(figsize=(12, 5))

# 绘制真实值
plt.plot(y_test.index, y_test.values, label='真实值', alpha=0.7)

# 【修改点】这里将 y_pred_xgb 改为随机森林的预测结果变量
# 假设你随机森林的预测结果保存在 y_pred_rf 中
plt.plot(y_test.index, y_pred_rf, label='随机森林预测值', linestyle='--')

plt.xlabel('日期')
# 【修改点】修改标题，去掉XGBoost字样，强调是无泄露情况
plt.title('测试集AQI真实值与随机森林预测值对比（无泄露）')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 12. 特征重要性
importance = pd.Series(xgb_best.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importance.head(15).plot(kind='bar')
plt.title('XGBoost 特征重要性（Top 15）')
plt.tight_layout()
plt.show()

实验三
模型定阶对比
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# 1. 数据准备# ==========================================
print("🚀 加载并预处理数据...")
df = pd.read_csv('Changsha_AQI_cleaned.csv', header=0)
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

# 特征列
target_col = 'AQI'
exog_cols = ['temp', 'pressure', 'humidity', 'wind_speed', 'PM2.5', 'NO2']

# 处理缺失值并设定频率
df.set_index('date', inplace=True)
df = df.asfreq('D')
df[target_col] = df[target_col].diff()  # 差分处理不平稳
df.fillna(df.mean(), inplace=True)

# 划分训练集和测试集（用最后30天作为测试集，模拟真实预测）
train_size = int(len(df) * 0.9)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print(f"📊 训练集大小: {len(train)}, 测试集大小: {len(test)}")

# ==========================================
# 2. 定义两个待对比的模型
# ==========================================
# Model A: 你跑出来的最优模型 (带季节性差分)
model_a_order = (1, 1, 2)
model_a_seasonal = (0, 1, 1, 7)  # D=1 是关键

# Model B: 原来的模型 (不带季节性差分)
model_b_order = (1, 1, 1)
model_b_seasonal = (1, 0, 1, 7)  # D=0

models = {
    "Model A (1,1,2)(0,1,1,7)": (model_a_order, model_a_seasonal),
    "Model B (1,1,1)(1,0,1,7)": (model_b_order, model_b_seasonal)
}

results_summary = []

# ==========================================
# 3. 循环训练与评估
# ==========================================
for name, (order, seasonal_order) in models.items():
    print(f"\n{'=' * 50}")
    print(f"🏃 正在训练: {name}")
    print(f"{'=' * 50}")

    try:
        # 训练模型
        model = SARIMAX(
            train[target_col],
            exog=train[exog_cols],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        # 预测（需要使用测试集的外生变量）
        forecast = results.get_forecast(steps=len(test), exog=test[exog_cols])
        pred_mean = forecast.predicted_mean

        # 反差分还原数据（因为之前做过 diff()）
        # 注意：预测值需要加上最后一个训练值才能还原真实尺度
        pred_restored = train[target_col].iloc[-1] + pred_mean.cumsum()
        test_restored = train[target_col].iloc[-1] + test[target_col].cumsum()

        # 计算指标
        rmse = np.sqrt(mean_squared_error(test_restored, pred_restored))
        mae = mean_absolute_error(test_restored, pred_restored)

        print(f"✅ 训练成功!")
        print(f"   AIC: {results.aic:.2f}")
        print(f"   BIC: {results.bic:.2f}")
        print(f"   测试集 RMSE: {rmse:.2f}")
        print(f"   测试集 MAE: {mae:.2f}")

        results_summary.append({
            'Model': name,
            'AIC': results.aic,
            'BIC': results.bic,
            'RMSE': rmse,
            'MAE': mae,
            'Predictions': pred_restored,
            'Fitted': results.fittedvalues
        })

    except Exception as e:
        print(f"❌ 训练失败: {e}")

# ==========================================
# 4. 结果汇总与可视化
# ==========================================
results_df = pd.DataFrame(results_summary)
print("\n" + "=" * 50)
print("🏆 模型对比排行榜 (The Lower The Better)")
print("=" * 50)
print(results_df[['Model', 'AIC', 'BIC', 'RMSE', 'MAE']].to_string(index=False))

# 可视化对比
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 图1：拟合效果对比（训练集）
axes[0].plot(train.index, train[target_col], label='Training Data', color='gray', alpha=0.6)
for res in results_summary:
    axes[0].plot(train.index, res['Fitted'], label=f"{res['Model']} Fit")
axes[0].set_title('In-Sample Fit Comparison (训练集拟合对比)')
axes[0].legend()
axes[0].grid(True)

# 图2：预测效果对比（测试集）
axes[1].plot(test_restored.index, test_restored, label='Actual Test Data', color='black', linewidth=2)
for res in results_summary:
    axes[1].plot(test_restored.index, res['Predictions'], label=f"{res['Model']} Forecast (RMSE:{res['RMSE']:.1f})")
axes[1].set_title('Out-of-Sample Forecast Comparison (测试集预测对比)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()




实验四整体代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# 忽略收敛警告
warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================
# 1. 数据加载与预处理
# ==========================
def load_data(filepath):
    """加载并预处理AQI数据"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # 确保AQI列为数值型且无缺失
    df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
    df.dropna(subset=['AQI'], inplace=True)

    print(f"数据范围: {df.index.min()} ~ {df.index.max()}")
    print(f"样本数量: {len(df)}")
    print(f"AQI统计描述:\n{df['AQI'].describe()}")
    return df


# ==========================
# 2. 单变量AQI序列分析
# ==========================
def univariate_analysis(series):
    """绘制时序图、ACF/PACF图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    # 时序图
    axes[0].plot(series, color='steelblue', linewidth=0.8)
    axes[0].set_title('长沙AQI日度时序图', fontsize=14)
    axes[0].set_xlabel('日期')
    axes[0].set_ylabel('AQI')
    axes[0].grid(True, alpha=0.3)

    # ACF
    plot_acf(series, lags=60, ax=axes[1], title='自相关函数 (ACF)')
    axes[1].grid(True, alpha=0.3)

    # PACF
    plot_pacf(series, lags=60, ax=axes[2], title='偏自相关函数 (PACF)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aqi_univariate_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 单变量分析图已保存为 aqi_univariate_analysis.png")


# ==========================
# 3. SARIMA模型 (周期=7)
# ==========================
def fit_sarima(train, test):
    """拟合SARIMA模型并评估"""
    print("\n" + "=" * 50)
    print("📊 SARIMA模型训练 (季节周期=7)")
    print("=" * 50)

    # SARIMA(p,d,q)(P,D,Q,s) - 根据ACF/PACF和经验选择参数
    # 若需自动寻优可使用pmdarima.auto_arima，此处使用合理默认参数
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False, maxiter=200)
    print(results.summary().tables[1])

    # 测试集预测
    forecast = results.forecast(steps=len(test))

    # 评估指标
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"\nSARIMA 测试集 MAE: {mae:.2f}")
    print(f"SARIMA 测试集 RMSE: {rmse:.2f}")

    return results, forecast, mae, rmse


# ==========================
# 4. Holt-Winters指数平滑模型
# ==========================
def fit_holt_winters(train, test):
    """拟合Holt-Winters模型并评估"""
    print("\n" + "=" * 50)
    print("📊 Holt-Winters指数平滑模型训练")
    print("=" * 50)

    model = ExponentialSmoothing(
        train,
        trend='add',
        seasonal='add',
        seasonal_periods=7,
        damped_trend=True
    )

    results = model.fit(optimized=True)

    # 打印关键平滑参数
    params = results.params
    print(f"Alpha (水平): {params.get('smoothing_level', 'N/A'):.4f}")
    print(f"Beta  (趋势): {params.get('smoothing_trend', 'N/A'):.4f}")
    print(f"Gamma (季节): {params.get('smoothing_seasonal', 'N/A'):.4f}")

    # 测试集预测
    forecast = results.forecast(steps=len(test))

    # 评估指标
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"\nHolt-Winters 测试集 MAE: {mae:.2f}")
    print(f"Holt-Winters 测试集 RMSE: {rmse:.2f}")

    return results, forecast, mae, rmse


# ==========================
# 5. 多步预测（未来7天）& 可视化对比
# ==========================
def multi_step_forecast_and_plot(series, sarima_results, hw_results, test_index):
    """基于全量数据重新拟合，预测未来7天并绘图"""
    print("\n" + "=" * 50)
    print("🔮 未来7天AQI多步预测")
    print("=" * 50)

    # 用全部数据重新拟合以获取最新预测
    sarima_full = SARIMAX(
        series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False, maxiter=200)

    hw_full = ExponentialSmoothing(
        series, trend='add', seasonal='add', seasonal_periods=7, damped_trend=True
    ).fit(optimized=True)

    # 未来7天预测
    sarima_future = sarima_full.forecast(steps=7)
    hw_future = hw_full.forecast(steps=7)

    # 生成未来日期索引
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')

    print("\n未来7天AQI预测结果:")
    pred_df = pd.DataFrame({
        '日期': future_dates.strftime('%Y-%m-%d'),
        'SARIMA预测': sarima_future.values.round(1),
        'Holt-Winters预测': hw_future.values.round(1)
    })
    print(pred_df.to_string(index=False))

    # 绘制对比图
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))

    # 上图：测试集拟合效果
    axes[0].plot(series[test_index[0]:], label='真实值', color='black', linewidth=1.2)
    axes[0].plot(test_index, sarima_results.fittedvalues[-len(test_index):] if hasattr(sarima_results,
                                                                                       'fittedvalues') else [np.nan] * len(
        test_index),
                 label='SARIMA(测试集)', color='red', linestyle='--', linewidth=1)
    axes[0].set_title('测试集拟合对比', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 下图：历史数据 + 未来7天预测
    axes[1].plot(series[-90:], label='历史AQI(近90天)', color='steelblue', linewidth=1)
    axes[1].plot(future_dates, sarima_future, 'ro-', label='SARIMA未来7天', markersize=6)
    axes[1].plot(future_dates, hw_future, 'gs--', label='Holt-Winters未来7天', markersize=6)
    axes[1].axvline(x=last_date, color='gray', linestyle=':', label='预测起点')
    axes[1].set_title('未来7天AQI多步预测', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aqi_7day_forecast.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ 预测对比图已保存为 aqi_7day_forecast.png")

    return pred_df


# ==========================
# 主程序入口
# ==========================
if __name__ == "__main__":
    # 1. 加载数据
    FILE_PATH = "Changsha_AQI_cleaned.csv"
    df = load_data(FILE_PATH)
    aqi_series = df['AQI']

    # 2. 单变量分析
    univariate_analysis(aqi_series)

    # 3. 划分训练集/测试集 (最后30天作为测试集)
    TEST_DAYS = 30
    train = aqi_series[:-TEST_DAYS]
    test = aqi_series[-TEST_DAYS:]
    print(f"\n训练集: {len(train)}天 | 测试集: {len(test)}天")

    # 4. SARIMA建模
    sarima_res, sarima_fc, sarima_mae, sarima_rmse = fit_sarima(train, test)

    # 5. Holt-Winters建模
    hw_res, hw_fc, hw_mae, hw_rmse = fit_holt_winters(train, test)

    # 6. 模型对比总结
    print("\n" + "=" * 50)
    print("📋 模型性能对比")
    print("=" * 50)
    print(f"{'模型':<20} {'MAE':>10} {'RMSE':>10}")
    print("-" * 42)
    print(f"{'SARIMA(1,1,1)(1,0,1,7)':<20} {sarima_mae:>10.2f} {sarima_rmse:>10.2f}")
    print(f"{'Holt-Winters':<20} {hw_mae:>10.2f} {hw_rmse:>10.2f}")

    # 7. 未来7天多步预测
    pred_table = multi_step_forecast_and_plot(
        aqi_series, sarima_res, hw_res, test.index
    )

实验四
长沙空气质量指数(AQI)预测 - 深度学习模型完整代码
模型：LSTM / LSTM-ARIMA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs(r"D:\时间序列实训\results", exist_ok=True)
os.makedirs(r"D:\时间序列实训\models", exist_ok=True)

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读数据 ====================
df = pd.read_csv(r"D:\时间序列实训\Changsha_AQI_cleaned.csv", parse_dates=["date"])
df.set_index("date", inplace=True)
df['month'] = df.index.month
df['dayofweek'] = df.index.dayofweek

cols = ['AQI','PM2.5','PM10','NO2','SO2','CO','O3',
        'temp','humidity','wind_speed','month','dayofweek']
df = df[cols].copy()

# ==================== 2. 切分 ====================
train_df = df.iloc[:int(len(df)*0.8)].copy()
test_df  = df.iloc[int(len(df)*0.8):].copy()
train_df = train_df.interpolate('linear').fillna(method='ffill').dropna()
test_df  = test_df.interpolate('linear').fillna(method='ffill').fillna(0)

feature_cols = cols
target_col = 'AQI'

# ==================== 3. 归一化 ====================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
Xtr = scaler_X.fit_transform(train_df[feature_cols].values)
Xte = scaler_X.transform(test_df[feature_cols].values)
Ytr = scaler_y.fit_transform(train_df[[target_col]].values).flatten()
Yte = scaler_y.transform(test_df[[target_col]].values).flatten()

# ==================== 4. 滑动窗口 ====================
TS = 7
def make_win(X, y, ts):
    Xs, Ys = [], []
    for i in range(ts, len(X)):
        Xs.append(X[i-ts:i]); Ys.append(y[i])
    return np.array(Xs), np.array(Ys)

Xtr_l, ytr_l = make_win(Xtr, Ytr, TS)
Xte_l, yte_l = make_win(Xte, Yte, TS)
full_scaled = scaler_X.transform(df[feature_cols].values)
y_true = scaler_y.inverse_transform(yte_l.reshape(-1,1)).flatten()
test_dates = test_df.index[TS:]
fut_dates = pd.date_range(df.index[-1]+pd.Timedelta(days=1), periods=7, freq='D')

def calc(y_t, y_p, name):
    rm = np.sqrt(mean_squared_error(y_t, y_p))
    ma = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    print(f"{name}: RMSE={rm:.2f}  MAE={ma:.2f}  R²={r2:.4f}")
    return rm, ma, r2

def roll_lstm(m, fs, sy, ts=7):
    w, aq = fs[-ts:].copy(), 0
    p = []
    for _ in range(7):
        pr = m.predict(np.expand_dims(w,0), verbose=0)[0,0]
        p.append(pr)
        nr = w[-1].copy(); nr[aq] = pr
        w = np.vstack([w[1:], nr])
    return np.round(sy.inverse_transform(np.array(p).reshape(-1,1)).flatten(), 1)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ===================== 5.1 LSTM =====================
print("\n" + "="*50)
print("LSTM 模型训练")
print("="*50)

m_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(TS, Xtr_l.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
m_lstm.compile(optimizer='adam', loss='mse')
m_lstm.fit(Xtr_l, ytr_l, epochs=100, batch_size=32,
           validation_split=0.1, callbacks=[es], shuffle=False, verbose=1)

yp_l = scaler_y.inverse_transform(m_lstm.predict(Xte_l, verbose=0)).flatten()
lrm, lma, lr2 = calc(y_true, yp_l, "LSTM")
fl = roll_lstm(m_lstm, full_scaled, scaler_y)
m_lstm.save(r"D:\时间序列实训\models\lstm_model.keras")

# ==================== 5.2 LSTM-ARIMA ====================
print("\n" + "="*50)
print("LSTM-ARIMA 混合模型")
print("="*50)

yr_tr  = scaler_y.inverse_transform(m_lstm.predict(Xtr_l, verbose=0)).flatten()
yr_true= scaler_y.inverse_transform(ytr_l.reshape(-1,1)).flatten()
resid  = yr_true - yr_tr

lb = acorr_ljungbox(resid, lags=[7,14,30], return_df=True)
print("Ljung-Box:\n", lb)

ar = ARIMA(resid, order=(1,0,1)).fit()
print(f"ARIMA AIC={ar.aic:.2f}")

test_res = ar.forecast(steps=len(yp_l))
hp = yp_l + np.array(test_res)
fh = fl + ar.forecast(steps=7)

hrm, hma, hr2 = calc(y_true, hp, "LSTM-ARIMA")

# ==================== 汇总输出 ====================
dl = pd.DataFrame({
    '模型': ['LSTM', 'LSTM-ARIMA'],
    'RMSE': [lrm, hrm],
    'MAE':  [lma, hma],
    'R²':   [lr2, hr2]
})
print("\n对比汇总：")
print(dl.to_string(index=False))
dl.to_csv(r"D:\时间序列实训\results\深度学习模型对比.csv", index=False, encoding='utf-8-sig')

# ==================== 输出未来7天预测数值 ====================
print("\n" + "="*60)
print("未来7日具体预测数值")
print("="*60)

future_results = pd.DataFrame({
    '日期': fut_dates,
    'LSTM预测': fl,
    'LSTM-ARIMA预测': fh
})
print(future_results.to_string(index=False))
future_results.to_csv(r"D:\时间序列实训\results\未来7天预测数值表.csv", 
                      index=False, encoding='utf-8-sig')

# ==================== 图1：LSTM测试集效果 ====================
plt.figure(figsize=(14, 5))
plt.plot(test_dates, y_true, 'k-', lw=1.5, label='真实AQI', alpha=0.8)
plt.plot(test_dates, yp_l, '--', c='#1f77b4', lw=1.5, 
         label=f'LSTM预测 (RMSE={lrm:.2f}, R²={lr2:.4f})')
plt.title('LSTM模型测试集预测效果')
plt.xlabel('日期')
plt.ylabel('AQI')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r"D:\时间序列实训\results\LSTM测试集效果图.png", dpi=300)
plt.show()

# ==================== 图2：LSTM-ARIMA测试集效果 ====================
plt.figure(figsize=(14, 5))
plt.plot(test_dates, y_true, 'k-', lw=1.5, label='真实AQI', alpha=0.8)
plt.plot(test_dates, hp, ':', c='#d62728', lw=1.5, 
         label=f'LSTM-ARIMA预测 (RMSE={hrm:.2f}, R²={hr2:.4f})')
plt.title('LSTM-ARIMA混合模型测试集预测效果')
plt.xlabel('日期')
plt.ylabel('AQI')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r"D:\时间序列实训\results\LSTM_ARIMA测试集效果图.png", dpi=300)
plt.show()

# ==================== 图3：LSTM未来7天预测 ====================
plt.figure(figsize=(10, 5))
plt.plot(fut_dates, fl, 'o-', c='#1f77b4', lw=2, markersize=8, label='LSTM预测')
plt.title('LSTM模型未来7日AQI预测')
plt.xlabel('日期')
plt.ylabel('AQI预测值')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(r"D:\时间序列实训\results\LSTM未来7天预测图.png", dpi=300)
plt.show()

# ==================== 图4：LSTM-ARIMA未来7天预测 ====================
plt.figure(figsize=(10, 5))
plt.plot(fut_dates, fh, '^-', c='#d62728', lw=2, markersize=8, label='LSTM-ARIMA预测')
plt.title('LSTM-ARIMA混合模型未来7日AQI预测')
plt.xlabel('日期')
plt.ylabel('AQI预测值')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(r"D:\时间序列实训\results\LSTM_ARIMA未来7天预测图.png", dpi=300)
plt.show()

print("\n✅ 全部完成！结果已保存至 results 文件夹")
print("📊 预测数值表：未来7天预测数值表.csv")
print("🖼️ 图片列表：")
print("  1. LSTM测试集效果图.png")
print("  2. LSTM_ARIMA测试集效果图.png")
print("  3. LSTM未来7天预测图.png")
print("  4. LSTM_ARIMA未来7天预测图.png")