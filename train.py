# from main import N_TARGET_CHANNELS
import numpy as np
import os
import glob  # 导入glob模块用于查找文件
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor  # <--- 添加此导入
import xgboost as xgb
import lightgbm as lgb
import gc  # 导入垃圾回收模块

N_VARS = 9
N_LEVELS = 13
N_CHANNELS = N_VARS * N_LEVELS
N_TARGET_CHANNELS = 30

data_dir = "/share/home/fangzijie/ssfy/datasets/processed_data"

# --- 加载所有批处理数据 ---
all_X_list = []
all_Y_list = []

# 查找所有 batch_*.npy 文件
batch_files = sorted(glob.glob(os.path.join(data_dir, "batch_*.npy")))  # 使用glob查找并排序

if not batch_files:
    print(f"错误：在目录 {data_dir} 中未找到任何 batch_*.npy 文件。脚本将退出。")
    exit()

print(f"找到以下批处理文件: {batch_files}")

batch_files = batch_files[:4]

for file_path in batch_files:
    print(f"正在加载文件: {file_path}")
    try:
        # 1. 使用 mmap_mode='r'
        data_batch = np.load(file_path, mmap_mode='r')
        if data_batch.size == 0:
            print(f"警告: 文件 {file_path} 为空，已跳过。")
            continue

        num_feature_columns = 1 + 2 + N_CHANNELS * 3  # k, lat, lon, first_data, second_data, diff_data

        if data_batch.shape[1] < num_feature_columns + N_TARGET_CHANNELS:
            print(
                f"警告: 文件 {file_path} 的列数不足 ({data_batch.shape[1]})，期望至少 {num_feature_columns + N_TARGET_CHANNELS} 列。已跳过。")
            continue

        X_batch = data_batch[:, :num_feature_columns]
        Y_batch = data_batch[:, num_feature_columns:num_feature_columns + N_TARGET_CHANNELS]

        all_X_list.append(X_batch.astype(np.float32, copy=False))  # 2a. 尽早转换类型，copy=False尝试避免不必要的复制
        all_Y_list.append(Y_batch.astype(np.float32, copy=False))  # 2a. 尽早转换类型

        print(f"    加载成功: X_batch shape {X_batch.shape}, Y_batch shape {Y_batch.shape}")

        # 3. 及时删除大的中间变量
        del data_batch

    except Exception as e:
        print(f"加载或处理文件 {file_path} 时出错: {e}")

if not all_X_list:
    print("错误：未能成功加载任何批处理数据。脚本将退出。")
    exit()

X = np.concatenate(all_X_list, axis=0)
Y = np.concatenate(all_Y_list, axis=0)

# 2b. 确保最终的X, Y是float32类型
if X.dtype != np.float32:
    X = X.astype(np.float32)
if Y.dtype != np.float32:
    Y = Y.astype(np.float32)

print(f"X 数据类型: {X.dtype}, Y 数据类型: {Y.dtype}")

del all_X_list, all_Y_list
gc.collect()  # 在大型列表删除后进行一次垃圾回收

print(f"\n总特征 X 的形状:{X.shape}")
print(f"总目标 Y 的形状:{Y.shape}")

# --- K折交叉验证 ---
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_results_rmse = []  # 重命名为 fold_results_rmse 以保持一致性

print(f"\n开始进行 {n_splits}-折交叉验证 (XGBoost + LightGBM 投票融合)...")

for fold, (train_index, val_index) in enumerate(kf.split(X, Y)):
    print(f"\n--- 第 {fold + 1} 折 ---")

    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]

    print(f"当前折训练集 X_train 的形状: {X_train.shape}")
    print(f"当前折验证集 X_val 的形状: {X_val.shape}")

    # 对训练集进行标准化
    scaler = StandardScaler()  # 保持变量名为 scaler
    X_train_scaled = scaler.fit_transform(X_train)
    # 对验证集进行标准化
    X_val_scaled = scaler.transform(X_val)

    # --- XGBoost 模型 ---
    print(f"正在训练 XGBoost 模型 (第 {fold + 1} 折)...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        device='cuda'
    )
    xgb_model.fit(X_train_scaled, Y_train)  # 使用标准化后的数据
    Y_pred_xgb = xgb_model.predict(X_val_scaled)  # 使用标准化后的数据
    print(f"XGBoost 模型 (第 {fold + 1} 折) 训练和预测完成。")

    # --- 在训练LightGBM之前，尝试释放XGBoost占用的内存 ---
    del xgb_model
    gc.collect()
    print("已尝试释放 XGBoost 模型内存。")

    # --- LightGBM 模型 ---
    print(f"正在训练 LightGBM 模型 (第 {fold + 1} 折)...")
    # 创建一个基础的 LightGBM 回归器
    base_lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,  # 这个 n_jobs 控制单个LGBM模型的线程数 (通常对于GPU影响不大)
        device='gpu',
        # gpu_platform_id=0,
        # gpu_device_id=0,
    )
    # 使用 MultiOutputRegressor 包装基础模型以处理多输出
    # 将 n_jobs 设置为 1，以串行方式在GPU上训练每个目标模型，避免GPU内存溢出
    lgb_model = MultiOutputRegressor(base_lgb_model, n_jobs=6) 

    lgb_model.fit(X_train_scaled, Y_train)  # Y_train 现在可以是2D的
    Y_pred_lgb = lgb_model.predict(X_val_scaled)  # 使用标准化后的数据
    print(f"LightGBM 模型 (第 {fold + 1} 折) 训练和预测完成。")

    # --- 投票融合 (简单平均) ---
    Y_val_compare = Y_val  # 真实值

    # 确保预测结果和真实值在计算RMSE前有相同的形状
    if Y_pred_xgb.ndim == 1 and Y_val_compare.shape[1] > 1: Y_pred_xgb = Y_pred_xgb.reshape(-1,
                                                                                            1)  # 可能不需要，因为Y_train是多输出的
    if Y_pred_lgb.ndim == 1 and Y_val_compare.shape[1] > 1: Y_pred_lgb = Y_pred_lgb.reshape(-1, 1)  # 可能不需要

    Y_pred_ensemble = (Y_pred_xgb + Y_pred_lgb) / 2.0
    print(f"模型预测融合完成 (第 {fold + 1} 折)。")

    # --- 评估 ---
    mse_fold = np.mean((Y_val_compare - Y_pred_ensemble) ** 2)
    rmse_fold = np.sqrt(mse_fold)

    print(f"第 {fold + 1} 折的验证集均方根误差 (RMSE): {rmse_fold}")
    fold_results_rmse.append(rmse_fold)  # 使用 fold_results_rmse

    # 4. 在每折结束后尝试垃圾回收
    del X_train, X_val, Y_train, Y_val, X_train_scaled, X_val_scaled 
    del Y_pred_xgb, Y_pred_lgb, Y_pred_ensemble # xgb_model 已经提前删除了
    if 'lgb_model' in locals() or 'lgb_model' in globals(): # 确保lgb_model存在再删除
        del lgb_model
    gc.collect()

print("\n--- K折交叉验证完成 ---")
if fold_results_rmse:  # 确保列表不为空
    print(f"各折的验证集 RMSE: {fold_results_rmse}")
    print(f"平均验证集 RMSE: {np.mean(fold_results_rmse)}")
    if len(fold_results_rmse) > 1:
        print(f"验证集 RMSE 标准差: {np.std(fold_results_rmse)}")
else:
    print("没有成功完成任何折的评估。")

