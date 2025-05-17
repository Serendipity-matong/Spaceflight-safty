from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import math

X_data_full = np.load("/Users/fangzijie/Documents/processed_data_corrected/X_dataset.npy")
Y_data_full = np.load("/Users/fangzijie/Documents/processed_data_corrected/Y_dataset.npy")

# 数据裁剪：只加载1/3的数据
num_total_samples = X_data_full.shape[0]
num_samples_to_keep = num_total_samples // 3 # 或者 int(num_total_samples / 3)

X_data = X_data_full[:num_samples_to_keep]
Y_data = Y_data_full[:num_samples_to_keep]

print(f"原始样本数: {num_total_samples}, 裁剪后样本数: {X_data.shape[0]}")
NUM_SAMPLES = X_data.shape[0]
INPUT_HISTORY_STEPS = X_data.shape[1]
MODEL_IN_CHANNELS = X_data.shape[2]
IMG_H = X_data.shape[3]
IMG_W = X_data.shape[4]

N_TARGET_CHANNELS = Y_data.shape[1]
OUTPUT_FORECAST_STEPS = Y_data.shape[2]

IMG_SIZE = (IMG_H, IMG_W)

INPUT_FEATURE_DIM = MODEL_IN_CHANNELS * IMG_H * IMG_W
OUTPUT_FEATURE_DIM = N_TARGET_CHANNELS * IMG_H * IMG_W

D_MODEL = 512
N_HEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
LEARNING_RATE = 1e-4
EPOCHS = 50
BATCH_SIZE = 4
N_SPLITS = 5

X_data = X_data.reshape(NUM_SAMPLES, INPUT_HISTORY_STEPS, INPUT_FEATURE_DIM)
Y_data = Y_data.reshape(NUM_SAMPLES, OUTPUT_FORECAST_STEPS, OUTPUT_FEATURE_DIM)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 dropout):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Linear(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        src_embedded = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        src_pos = self.pos_encoder(src_embedded)
        tgt_pos = self.pos_encoder(tgt_embedded)

        tgt_seq_len = tgt_pos.size(0)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        memory = self.transformer.encoder(src_pos)  # 默认不使用 src_mask
        output = self.transformer.decoder(tgt_pos, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output.permute(1, 0, 2)  #

# K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_index, val_index) in enumerate(kf.split(X_data)):
    X_train, X_val = X_data[train_index], X_data[val_index]
    Y_train, Y_val = Y_data[train_index], Y_data[val_index]
    # 数据标准化
    scaler_X_fold = StandardScaler()
    scaler_Y_fold = StandardScaler()

    # --- 处理 X_train ---
    X_train_orig_shape = X_train.shape # 保存原始3D形状 (num_train_samples, train_steps_x, INPUT_FEATURE_DIM)
    # 为了scaler.fit/transform，展平为2D
    X_train_flat = X_train.reshape(-1, INPUT_FEATURE_DIM)
    scaler_X_fold.fit(X_train_flat)
    X_train_scaled_flat = scaler_X_fold.transform(X_train_flat)
    # 转换回原始的3D形状
    X_train_scaled = X_train_scaled_flat.reshape(X_train_orig_shape)

    # --- 处理 Y_train ---
    Y_train_orig_shape = Y_train.shape # 保存原始3D形状 (num_train_samples, train_steps_y, OUTPUT_FEATURE_DIM)
    # 为了scaler.fit/transform，展平为2D
    Y_train_flat = Y_train.reshape(-1, OUTPUT_FEATURE_DIM)
    scaler_Y_fold.fit(Y_train_flat)
    Y_train_scaled_flat = scaler_Y_fold.transform(Y_train_flat)
    # 转换回原始的3D形状
    Y_train_scaled = Y_train_scaled_flat.reshape(Y_train_orig_shape)

    # Transform 验证数据 (X_val) - 这部分逻辑看起来是正确的
    num_val_samples_fold, val_time_steps_x_fold, _ = X_val.shape
    X_val_fold_for_scaling = X_val.reshape(num_val_samples_fold * val_time_steps_x_fold, INPUT_FEATURE_DIM)
    X_val_scaled_flat_fold = scaler_X_fold.transform(X_val_fold_for_scaling)
    X_val_scaled_fold = X_val_scaled_flat_fold.reshape(num_val_samples_fold, val_time_steps_x_fold, INPUT_FEATURE_DIM)

    # Transform 验证数据 (Y_val) - 这部分逻辑看起来是正确的
    # num_val_samples_fold 对于 X_val 和 Y_val 应该是相同的
    val_time_steps_y_fold = Y_val.shape[1]
    Y_val_fold_for_scaling = Y_val.reshape(num_val_samples_fold * val_time_steps_y_fold, OUTPUT_FEATURE_DIM)
    Y_val_scaled_flat_fold = scaler_Y_fold.transform(Y_val_fold_for_scaling)
    Y_val_scaled_fold = Y_val_scaled_flat_fold.reshape(num_val_samples_fold, val_time_steps_y_fold, OUTPUT_FEATURE_DIM)
    print("  当前折的数据归一化完成。")

    # 3.2.3 转换为 PyTorch Tensors
    X_train_tensor_fold = torch.FloatTensor(X_train_scaled) # <--- 使用缩放并恢复3D形状的X_train
    Y_train_tensor_fold = torch.FloatTensor(Y_train_scaled) # <--- 使用缩放并恢复3D形状的Y_train
    X_val_tensor_fold = torch.FloatTensor(X_val_scaled_fold)
    Y_val_tensor_fold = torch.FloatTensor(Y_val_scaled_fold)


    class WeatherDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]


    train_dataset_fold = WeatherDataset(X_train_tensor_fold, Y_train_tensor_fold)
    val_dataset_fold = WeatherDataset(X_val_tensor_fold, Y_val_tensor_fold)
    # --- 修改 DataLoader 参数以匹配 BATCH_SIZE 变量并优化内存 ---
    train_dataloader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_dataloader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    print(f"  当前折 - 训练集样本数: {len(train_dataset_fold)}, 验证集样本数: {len(val_dataset_fold)}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # <--- 注释掉这行
    device = torch.device("cpu") # <--- 强制使用CPU
    print(f"  使用设备: {device}")

    model_fold = TransformerModel(  # 重新初始化模型以保证每折独立训练
        input_dim=INPUT_FEATURE_DIM,
        output_dim=OUTPUT_FEATURE_DIM,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"  使用 {torch.cuda.device_count()} 个 GPUs!")
        model_fold = nn.DataParallel(model_fold)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model_fold.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 7. 当前折的训练循环
    print(f"  开始第 {fold + 1} 折的训练...")
    best_val_loss_fold = float('inf')
    for epoch in range(EPOCHS):
        model_fold.train()
        train_loss_epoch = 0
        for batch_idx, (src_batch, tgt_batch) in enumerate(train_dataloader_fold):
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            optimizer.zero_grad()
            predictions = model_fold(src_batch, tgt_batch)  # 假设模型内部处理了 teacher forcing 和 mask
            loss = criterion(predictions, tgt_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        avg_train_loss = train_loss_epoch / len(train_dataloader_fold)

        # 8. 当前折的验证循环
        model_fold.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for src_batch_val, tgt_batch_val in val_dataloader_fold:
                src_batch_val, tgt_batch_val = src_batch_val.to(device), tgt_batch_val.to(device)
                predictions_val = model_fold(src_batch_val, tgt_batch_val)
                loss_val = criterion(predictions_val, tgt_batch_val)
                val_loss_epoch += loss_val.item()
        avg_val_loss = val_loss_epoch / len(val_dataloader_fold)

        print(
            f"    折 [{fold + 1}/{N_SPLITS}], Epoch [{epoch + 1}/{EPOCHS}] -- Train Loss: {avg_train_loss:.4f} -- Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss_fold:
            best_val_loss_fold = avg_val_loss
            # 可选: 保存当前折的最佳模型
            # torch.save(model_fold.state_dict(), f"transformer_weather_model_fold_{fold+1}_best.pth")

    fold_results.append(best_val_loss_fold)
    print(f"  第 {fold + 1} 折完成。最佳验证损失: {best_val_loss_fold:.4f}")


