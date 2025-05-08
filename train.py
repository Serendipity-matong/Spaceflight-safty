# from main import N_TARGET_CHANNELS
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

N_VARS = 9
N_LEVELS = 13
N_CHANNELS = N_VARS * N_LEVELS
N_TARGET_CHANNELS = 30

data_dir = "/Users/fangzijie/Documents/processed_data"
file_dir = "batch_1.npy"

file_path = os.path.join(data_dir, file_dir)
data = np.load(file_path)
print(f"\n数据的形状:{data.shape}")

num_feature_columns = 1 + 2 + N_CHANNELS * 3

X = data[:, :num_feature_columns]
Y = data[:, num_feature_columns:]
print(f"\nX的形状:{X.shape}")
print(f"Y的形状:{Y.shape}")

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_results = []
for fold, (train_index, val_index) in enumerate(kf.split(X, Y)):
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]
    # 对训练集进行标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # 对验证集进行标准化
    X_val = scaler.transform(X_val)
    # 训练模型并评估
    print(f"训练xgboost模型，第{fold + 1}折")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, Y_train)
    Y_pred_xgb = xgb_model.predict(X_val)
    print(f"训练lightgbm模型，第{fold + 1}折")
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        num_leaves=31,
        # n_jobs=-1,
    )
    lgb_model.fit(X_train, Y_train)
    Y_pred_lgb = lgb_model.predict(X_val)
    # Y_pred = (Y_pred_xgb + Y_pred_lgb) / 2

    # 备份验证
    Y_val_compare = Y_val

    Y_pred_ensemble = (Y_pred_xgb + Y_pred_lgb) / 2.0

    mse_fold = np.mean((Y_val_compare - Y_pred_ensemble) ** 2)
    rmse_fold = np.sqrt(mse_fold)
    print(f"第{fold + 1}折的RMSE:{rmse_fold}")
    fold_results.append(rmse_fold)

print("\n--- K折交叉验证完成 ---")
if fold_results:  # 确保列表不为空
    print(f"各折的验证集 RMSE: {fold_results}")
    print(f"平均验证集 RMSE: {np.mean(fold_results)}")
    if len(fold_results) > 1:
        print(f"验证集 RMSE 标准差: {np.std(fold_results)}")
else:
    print("没有成功完成任何折的评估。")

