from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
X_data = np.load("./processed_data_corrected/X_data.npy")
Y_data = np.load("./processed_data_corrected/Y_data.npy")
NUM_SAMPLES = X_data.shape[0]
INPUT_HISTORY_STEPS = X_data.shape[1]
MODEL_IN_CHANNELS = X_data.shape[2]
IMG_H = X_data.shape[3]
IMG_W = X_data.shape[4]

N_TARGET_CHANNELS = Y_data.shape[1]
OUTPUT_FORECAST_STEPS = Y_data.shape[2]

IMG_SIZE = (IMG_H, IMG_W)

INPUT_FEATURE_DIM = MODEL_IN_CHANNELS * IMG_H * IMG_W
OUTPUT_FEATURE_DIM = N_TARGET_CHANNELS * OUTPUT_FORECAST_STEPS * IMG_H * IMG_W

X_data = X_data.reshape(NUM_SAMPLES, INPUT_HISTORY_STEPS, INPUT_FEATURE_DIM)
Y_data = Y_data.reshape(NUM_SAMPLES, OUTPUT_FORECAST_STEPS, OUTPUT_FEATURE_DIM)

#K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_index, val_index) in enumerate(kf.split(X_data)):
    X_train, X_val = X_data[train_index], X_data[val_index]
    Y_train, Y_val = Y_data[train_index], Y_data[val_index]
    # 数据标准化
    scaler_X_fold = StandardScaler()
    scaler_Y_fold = StandardScaler()

    num_train_samples,train_steps_x,_ = X_train.shape
    X_train = X_train.reshape(num_train_samples * train_steps_x, INPUT_FEATURE_DIM)
    scaler_X_fold.fit(X_train)

    _,train_steps_y,_ = Y_train.shape
    Y_train = Y_train.reshape(num_train_samples * train_steps_y, OUTPUT_FEATURE_DIM)
    scaler_Y_fold.fit(Y_train)

    # Transform 验证数据
    num_val_samples_fold, val_time_steps_x_fold, _ = X_data.shape
    X_val_fold_for_scaling = X_data.reshape(num_val_samples_fold * val_time_steps_x_fold, INPUT_FEATURE_DIM)
    X_val_scaled_flat_fold = scaler_X_fold.transform(X_val_fold_for_scaling)
    X_val_scaled_fold = X_val_scaled_flat_fold.reshape(num_val_samples_fold, val_time_steps_x_fold, INPUT_FEATURE_DIM)

    _, val_time_steps_y_fold, _ = Y_data.shape
    Y_val_fold_for_scaling = Y_data.reshape(num_val_samples_fold * val_time_steps_y_fold, OUTPUT_FEATURE_DIM)
    Y_val_scaled_flat_fold = scaler_Y_fold.transform(Y_val_fold_for_scaling)
    Y_val_scaled_fold = Y_val_scaled_flat_fold.reshape(num_val_samples_fold, val_time_steps_y_fold, OUTPUT_FEATURE_DIM)
    print("  当前折的数据归一化完成。")

    # 3.2.3 转换为 PyTorch Tensors
    X_train_tensor_fold = torch.FloatTensor(X_data)
    Y_train_tensor_fold = torch.FloatTensor(Y_data)
    X_val_tensor_fold = torch.FloatTensor(X_val_scaled_fold)
    Y_val_tensor_fold = torch.FloatTensor(Y_val_scaled_fold)

