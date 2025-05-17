import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import os

INPUT_HISTORY_STEPS = 2  # 从 transformer.py 确认或修改

# --- 维度常量 ---
# 原始输入数据 (.pt 文件) 的维度
RAW_MODEL_IN_CHANNELS = 117  # 与模型输入通道数一致
RAW_IMG_H = 181
RAW_IMG_W = 360

# 模型训练时使用的维度 (根据您的猜测和checkpoint推断)
MODEL_TRAIN_MODEL_IN_CHANNELS = 117
MODEL_TRAIN_IMG_H = 46
MODEL_TRAIN_IMG_W = 71
MODEL_TRAIN_N_TARGET_CHANNELS = 30  # <--- 修改：确保输出30个通道，与竞赛要求一致

# 全局常量，供模型初始化和预/后处理使用，应与模型训练时一致
MODEL_IN_CHANNELS = MODEL_TRAIN_MODEL_IN_CHANNELS
IMG_H = MODEL_TRAIN_IMG_H  # <--- 修改
IMG_W = MODEL_TRAIN_IMG_W  # <--- 修改
N_TARGET_CHANNELS = MODEL_TRAIN_N_TARGET_CHANNELS  # <--- 修改

OUTPUT_FORECAST_STEPS = 12  # 从 transformer.py 确认或修改

# 特征维度将根据上面的模型训练维度计算
INPUT_FEATURE_DIM = MODEL_IN_CHANNELS * IMG_H * IMG_W
OUTPUT_FEATURE_DIM = N_TARGET_CHANNELS * IMG_H * IMG_W

D_MODEL = 512
N_HEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
N_SPLITS = 5  # K-fold splits

DEVICE = torch.device("cpu")  # 竞赛环境通常建议使用CPU，除非明确支持GPU


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
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False  # batch_first=False as in training
        )
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        # Permute src and tgt to (seq_len, batch, feature_dim)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        src_embedded = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        src_pos = self.pos_encoder(src_embedded)
        tgt_pos = self.pos_encoder(tgt_embedded)

        tgt_seq_len = tgt_pos.size(0)
        # Ensure tgt_mask is on the same device as src_pos
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(src_pos.device)

        memory = self.transformer.encoder(src_pos)  # src_mask is not used by default
        output = self.transformer.decoder(tgt_pos, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)

        # Permute output back to (batch, seq_len, feature_dim)
        return output.permute(1, 0, 2)


def load_model_and_scalers(fold_idx, model_dir="."):
    """加载指定折数的模型和scalers"""
    model_path = os.path.join(model_dir, f"transformer_model_fold_{fold_idx + 1}_best.pth")
    scaler_x_path = os.path.join(model_dir, f"scaler_X_fold_{fold_idx + 1}.joblib")
    scaler_y_path = os.path.join(model_dir, f"scaler_Y_fold_{fold_idx + 1}.joblib")

    print(f"  Loading model from: {model_path}")
    print(f"  Loading scaler_X from: {scaler_x_path}")
    print(f"  Loading scaler_Y from: {scaler_y_path}")

    scaler_X = joblib.load(scaler_x_path)
    scaler_Y = joblib.load(scaler_y_path)

    model = TransformerModel(
        input_dim=INPUT_FEATURE_DIM,
        output_dim=OUTPUT_FEATURE_DIM,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, scaler_X, scaler_Y


def preprocess_input_data(input_data_raw_np, scaler_X):
    """
    预处理原始输入数据。
    `input_data_raw_np` 的形状是 (num_samples, INPUT_HISTORY_STEPS, RAW_MODEL_IN_CHANNELS, RAW_IMG_H, RAW_IMG_W)
    模型期望的输入特征维度是基于 (MODEL_IN_CHANNELS, IMG_H, IMG_W)
    """
    num_samples = input_data_raw_np.shape[0]
    S = input_data_raw_np.shape[1]  # INPUT_HISTORY_STEPS
    C_raw = input_data_raw_np.shape[2]  # RAW_MODEL_IN_CHANNELS
    H_raw = input_data_raw_np.shape[3]  # RAW_IMG_H
    W_raw = input_data_raw_np.shape[4]  # RAW_IMG_W

    # 确认原始输入通道数与模型期望的输入通道数一致
    if C_raw != MODEL_IN_CHANNELS:
        raise ValueError(
            f"原始输入数据通道数 ({C_raw}) 与模型期望的输入通道数 ({MODEL_IN_CHANNELS}) 不符。"
        )

    print(f"原始输入数据单一样本特征图形状: ({C_raw}, {H_raw}, {W_raw})")
    print(f"将重采样到模型期望形状: ({MODEL_IN_CHANNELS}, {IMG_H}, {IMG_W})")

    # 将 NumPy 数据转换为 PyTorch 张量以进行插值
    # 重塑为 (N*S, C_raw, H_raw, W_raw) 以便批量插值
    tensor_to_interpolate = torch.from_numpy(
        input_data_raw_np.reshape(num_samples * S, C_raw, H_raw, W_raw)
    ).float().to(DEVICE)

    # 使用双线性插值将 H_raw, W_raw 维度重采样到 IMG_H, IMG_W
    resampled_tensor = torch.nn.functional.interpolate(
        tensor_to_interpolate, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False
    )
    # resampled_tensor 的形状: (num_samples * S, MODEL_IN_CHANNELS, IMG_H, IMG_W)

    # 重塑为模型后续处理期望的3D形状 (num_samples, INPUT_HISTORY_STEPS, INPUT_FEATURE_DIM)
    # INPUT_FEATURE_DIM = MODEL_IN_CHANNELS * IMG_H * IMG_W
    input_data_reshaped_for_scaler = resampled_tensor.reshape(num_samples, S, INPUT_FEATURE_DIM)

    # 为了scaler.transform，展平为2D (num_samples * INPUT_HISTORY_STEPS, INPUT_FEATURE_DIM)
    # scaler 需要 NumPy 数组，并且通常在 CPU 上操作
    input_data_flat = input_data_reshaped_for_scaler.cpu().numpy().reshape(-1, INPUT_FEATURE_DIM)
    scaled_input_flat = scaler_X.transform(input_data_flat)

    # 转换回模型期望的3D形状 (num_samples, INPUT_HISTORY_STEPS, INPUT_FEATURE_DIM) 并移到DEVICE
    scaled_input_reshaped = torch.FloatTensor(
        scaled_input_flat.reshape(num_samples, S, INPUT_FEATURE_DIM)
    ).to(DEVICE)

    return scaled_input_reshaped


def generate_autoregressive_predictions(model, src_tensor):
    """使用模型进行自回归预测"""
    model.eval()
    num_samples = src_tensor.size(0)

    # 初始化解码器输入 (tgt)
    # 对于推理，我们通常以一个起始标记开始，然后逐步填充
    # 这里我们用全零张量作为起始，形状 (batch_size, 1, OUTPUT_FEATURE_DIM)
    # 注意：Transformer的forward期望tgt的seq_len维度与src的seq_len维度不一定相同
    # 在自回归生成时，tgt的seq_len会逐步增加

    # decoder_input_scaled = torch.zeros(num_samples, 1, OUTPUT_FEATURE_DIM, device=DEVICE)
    # all_steps_predictions_scaled = []

    # for _ in range(OUTPUT_FORECAST_STEPS):
    #     with torch.no_grad():
    #         # prediction_for_current_tgt_len 的形状是 (num_samples, current_tgt_seq_len, OUTPUT_FEATURE_DIM)
    #         prediction_for_current_tgt_len = model(src_tensor, decoder_input_scaled)

    #     # 我们需要的是基于当前decoder_input预测出的下一个时间步
    #     # 所以取预测序列的最后一个时间步
    #     next_step_prediction_scaled = prediction_for_current_tgt_len[:, -1:, :] # Shape: (num_samples, 1, OUTPUT_FEATURE_DIM)
    #     all_steps_predictions_scaled.append(next_step_prediction_scaled)

    #     # 将新预测的步添加到decoder_input中，用于下一步的预测
    #     decoder_input_scaled = torch.cat([decoder_input_scaled, next_step_prediction_scaled], dim=1)

    # # 组合所有预测的时间步
    # # decoder_input_scaled 在循环结束后，形状是 (num_samples, 1 (start_token) + OUTPUT_FORECAST_STEPS, OUTPUT_FEATURE_DIM)
    # # 我们需要的是预测的部分，即去掉起始标记
    # final_predictions_scaled = decoder_input_scaled[:, 1:, :]

    # --- 简化版推理：假设模型在训练时tgt输入的是完整的目标序列 ---
    # 对于推理，如果模型训练时tgt是完整序列，我们可以用一个dummy tgt输入
    # 这种方式不是严格的自回归，但有时也用，取决于模型训练方式
    # 如果是严格自回归，需要上面的循环
    # 如果训练时 teacher forcing 使用的是shifted target，那么推理时需要自回归
    # 您的训练代码中 model(src_batch, tgt_batch) 表明 tgt_batch 是已知的目标序列
    # 为了进行真正的预测（不知道未来），我们需要自回归。

    # 实现严格的自回归预测：
    generated_sequence_scaled = torch.zeros(num_samples, OUTPUT_FORECAST_STEPS, OUTPUT_FEATURE_DIM, device=DEVICE)
    current_tgt_input_scaled = torch.zeros(num_samples, 1, OUTPUT_FEATURE_DIM,
                                           device=DEVICE)  # Start with a single zero token

    with torch.no_grad():
        for t in range(OUTPUT_FORECAST_STEPS):
            prediction_step = model(src_tensor,
                                    current_tgt_input_scaled)  # model outputs (batch, seq_len_tgt, features)

            # The prediction for the *next* actual time step is the *last* item in the output sequence
            # given the current_tgt_input_scaled
            predicted_token_for_next_step = prediction_step[:, -1:, :]  # (batch, 1, features)

            generated_sequence_scaled[:, t:t + 1, :] = predicted_token_for_next_step

            # Append the predicted token to the current_tgt_input for the next iteration
            # This forms the new target sequence for the model
            current_tgt_input_scaled = torch.cat([current_tgt_input_scaled, predicted_token_for_next_step], dim=1)

    return generated_sequence_scaled


def postprocess_predictions(predictions_scaled_tensor, scaler_Y):
    """后处理预测结果（反标准化和reshape）"""
    predictions_scaled_np = predictions_scaled_tensor.cpu().numpy()
    num_samples = predictions_scaled_np.shape[0]

    # 为了scaler.inverse_transform，展平为2D
    # (num_samples * OUTPUT_FORECAST_STEPS, OUTPUT_FEATURE_DIM)
    # OUTPUT_FEATURE_DIM 是基于模型训练时的 N_TARGET_CHANNELS, IMG_H, IMG_W
    predictions_flat = predictions_scaled_np.reshape(-1, OUTPUT_FEATURE_DIM)
    unscaled_predictions_flat = scaler_Y.inverse_transform(predictions_flat)

    # Reshape回期望的最终输出格式
    # (num_samples, OUTPUT_FORECAST_STEPS, N_TARGET_CHANNELS, IMG_H, IMG_W)
    # 这里的 N_TARGET_CHANNELS, IMG_H, IMG_W 是模型训练时的维度
    final_predictions_unscaled = unscaled_predictions_flat.reshape(
        num_samples, OUTPUT_FORECAST_STEPS, N_TARGET_CHANNELS, IMG_H, IMG_W
    )
    return final_predictions_unscaled


# -----------------------------------------------------------------------------
# 4. 主执行逻辑
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("开始预测流程...")

    # 定义文件和目录路径
    input_directory = "Input/"
    output_directory = "Output/"
    # num_pt_files 将由实际找到的文件数量确定

    # 模型和scaler文件所在的目录 (假设与 predict.py 在同一级)
    # 假设您的 .pt 文件存放在 "data/input_pt_files/" 目录下
    # 请根据您的实际文件路径修改这个变量
    pt_files_input_directory = "/Users/fangzijie/Downloads/input"  # <--- 修改这里指向您的 .pt 文件目录
    num_pt_files = 60  # 从 00.pt 到 59.pt

    # 预测结果也将保存到 "data/" 目录下
    output_predictions_filename = "predictions.npy"  # 示例文件名
    output_predictions_path = os.path.join("data", output_predictions_filename)

    # 模型和scaler文件所在的目录 (假设与 predict.py 在同一级)
    model_files_dir = "."

    # 确保 data 目录存在，如果输出时需要创建
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)

    # 确保输入 .pt 文件的目录存在
    if not os.path.exists(pt_files_input_directory):
        print(f"错误：存放 .pt 文件的目录未找到: {pt_files_input_directory}")
        print("请确保该目录存在，并且包含 00.pt 到 59.pt 文件。")
        exit()

    # 加载并整合所有 .pt 文件的数据
    all_loaded_pt_data = []
    print(f"从目录 '{pt_files_input_directory}' 加载 .pt 文件...")
    for i in range(num_pt_files):
        file_name = f"{i:03d}.pt"
        file_path = os.path.join(pt_files_input_directory, file_name)

        if os.path.exists(file_path):
            try:
                print(f"  加载文件: {file_path}")
                # 加载 .pt 文件，确保在CPU上加载以避免设备问题
                tensor_data = torch.load(file_path, map_location=torch.device('cpu'))

                if not isinstance(tensor_data, torch.Tensor):
                    print(f"警告：文件 {file_path} 中的内容不是一个PyTorch张量，类型为 {type(tensor_data)}。跳过此文件。")
                    continue

                # 确认张量维度是否符合原始输入数据的预期
                expected_raw_shape = (1, INPUT_HISTORY_STEPS, RAW_MODEL_IN_CHANNELS, RAW_IMG_H, RAW_IMG_W)
                if tensor_data.shape != expected_raw_shape:
                    print(
                        f"警告：文件 {file_path} 的张量形状 {tensor_data.shape} 与预期原始形状 {expected_raw_shape} 不符。跳过此文件。")
                    continue

                all_loaded_pt_data.append(tensor_data.numpy())  # 转换为NumPy数组并添加
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}。跳过此文件。")
        else:
            print(f"文件 {file_path} 未找到。跳过。")

    if not all_loaded_pt_data:
        print(f"错误：未能从 '{pt_files_input_directory}' 加载任何有效的 .pt 文件数据。")
        exit()

    # 将所有加载的 NumPy 数组堆叠成一个大的输入数组
    # 假设每个 .pt 文件代表一个样本，所以我们沿第0轴堆叠
    raw_input_data_np = np.concatenate(all_loaded_pt_data, axis=0)
    print(f"所有 .pt 文件已加载并整合。原始输入数据形状: {raw_input_data_np.shape}")
    # 预期的形状应该是 (60, 2, 117, 181, 360) 或 (实际加载的文件数, 2, 117, 181, 360)

    all_folds_final_outputs_unscaled = []

    for i in range(N_SPLITS):
        print(f"\n--- 处理第 {i + 1} 折 ---")
        try:
            model_fold, scaler_X_fold, scaler_Y_fold = load_model_and_scalers(i, model_files_dir)

            # 预处理输入数据
            src_tensor_fold = preprocess_input_data(raw_input_data_np, scaler_X_fold)
            print(f"  预处理后输入张量形状: {src_tensor_fold.shape}")

            # 生成预测
            predictions_scaled_fold = generate_autoregressive_predictions(model_fold, src_tensor_fold)
            print(f"  预测 (scaled) 张量形状: {predictions_scaled_fold.shape}")

            # 后处理预测结果
            final_output_unscaled_fold = postprocess_predictions(predictions_scaled_fold, scaler_Y_fold)
            print(f"  最终预测 (unscaled) 形状: {final_output_unscaled_fold.shape}")

            all_folds_final_outputs_unscaled.append(final_output_unscaled_fold)

        except FileNotFoundError as e:
            print(f"错误：加载第 {i + 1} 折的模型或scaler文件失败: {e}")
            print("请确保所有必要的 .pth 和 .joblib 文件都存在于指定的目录中。")
            # 根据需要决定是否在此处退出或继续处理其他折
            # exit()
        except Exception as e:
            print(f"处理第 {i + 1} 折时发生错误: {e}")
            # exit()

    if not all_folds_final_outputs_unscaled:
        print("\n错误：未能成功处理任何一折的模型，无法生成集成预测。")
        exit()

    # 集成预测结果 (取平均值)
    print("\n集成所有折的预测结果...")
    # 将列表转换为NumPy数组以便计算平均值
    ensemble_predictions_np = np.array(all_folds_final_outputs_unscaled)
    # 沿着第0轴（折数轴）取平均
    final_ensemble_prediction = np.mean(ensemble_predictions_np, axis=0)
    print(f"集成后的最终预测形状: {final_ensemble_prediction.shape}")

    # 保存最终的集成预测结果
    np.save(output_predictions_path, final_ensemble_prediction)
    print(f"\n最终集成预测结果已保存到: {output_predictions_path}")
    print("预测流程完成。")
    # final_ensemble_prediction_all_samples 的形状: (num_actual_samples, OUTPUT_FORECAST_STEPS, N_TARGET_CHANNELS, IMG_H, IMG_W)
    final_ensemble_prediction_all_samples = np.mean(ensemble_predictions_np, axis=0)
    print(f"所有样本集成后的最终预测形状 (NumPy): {final_ensemble_prediction_all_samples.shape}")

    # 保存每个样本的集成预测结果为单独的 .pt 文件
    print(f"\n将每个样本的预测结果保存到 '{output_directory}' 目录...")
    for j in range(num_actual_samples):  # j 将从 0 到 num_actual_samples - 1
        # 提取单个样本的预测结果
        # sample_prediction_np 形状: (OUTPUT_FORECAST_STEPS, N_TARGET_CHANNELS, IMG_H, IMG_W)
        sample_prediction_np = final_ensemble_prediction_all_samples[j, :, :, :, :]

        # 添加批次维度 (1)，转换为 float16 Tensor
        # 最终形状: (1, OUTPUT_FORECAST_STEPS, N_TARGET_CHANNELS, IMG_H, IMG_W)
        # 即 (1, 12, 30, 46, 71)
        sample_prediction_tensor = torch.from_numpy(sample_prediction_np).unsqueeze(0).to(torch.float16)

        # 构建输出文件名，格式为 000.pt, 001.pt, ...
        output_filename = f"{j:03d}.pt"  # <--- 修改：使用3位补零的数字命名
        individual_output_path = os.path.join(output_directory, output_filename)

        try:
            torch.save(sample_prediction_tensor, individual_output_path)
            print(f"  已保存样本 {j} 的预测到: {individual_output_path}")
        except Exception as e:
            print(f"  保存样本 {j} (文件 {output_filename}) 的预测到 {individual_output_path} 时出错: {e}")

    print("\n预测流程完成。")