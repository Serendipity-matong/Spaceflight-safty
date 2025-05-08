# from locale import normalize
# import os
#
# import pylab
# import xarray as xr
# import torch
# import numpy as np
# from mpmath import power
#
# import matplotlib.pyplot as plt
#
# # 设置 SimHei 字体和禁用 Unicode minus
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 定义变量和气压层的顺序
# VARS = ['z', 'ciwc', 'clwc', 'q', 'crwc', 'cswc', 't', 'u', 'v']
# PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
# # 历史步数
# HISTORY_STEPS = 2
#
# def process_single_file(file_path):
#     """
#     处理单个 nc 文件，将其转换为符合要求的 numpy 数组
#     :param file_path: nc 文件的路径
#     :return: 形状为 (valid_time, 117, lat, lon) 的 numpy 数组
#     """
#     ds = xr.open_dataset(file_path)
#     channels = []
#     for var in VARS:
#         for level in PRESSURE_LEVELS:
#             channel_data = ds[var].sel(pressure_level=level).values
#             channels.append(channel_data)
#     combined_data = np.stack(channels, axis=1)
#     return combined_data
#
# def process_year_data(data_dir, save_path=None, force_reprocess=False):
#     """
#     处理一年的 nc 文件，将其转换为符合比赛要求的 PyTorch 张量
#     :param data_dir: 存储 nc 文件的目录
#     :param save_path: 保存处理后张量的路径
#     :param force_reprocess: 是否强制重新处理
#     :return: 形状为 (num_samples, HISTORY_STEPS, 117, lat, lon) 的 PyTorch 张量
#     """
#     # 检查是否有HDF5格式的保存文件
#     h5_save_path = save_path.replace('.pt', '.h5')
#     if os.path.exists(h5_save_path) and not force_reprocess:
#         print(f"从HDF5文件加载预处理数据: {h5_save_path}")
#         try:
#             import h5py
#             with h5py.File(h5_save_path, 'r') as f:
#                 # 从HDF5加载数据并转换为PyTorch张量
#                 tensor_data = f['tensor'][()]
#                 return torch.from_numpy(tensor_data).float()
#         except Exception as e:
#             print(f"加载HDF5文件出错: {e}")
#
#     # 尝试加载PyTorch格式
#     if save_path and os.path.exists(save_path) and not force_reprocess:
#         print(f"从PyTorch文件加载预处理数据: {save_path}")
#         try:
#             return torch.load(save_path)
#         except Exception as e:
#             print(f"加载张量出错: {e}")
#             print("文件可能已损坏，将重新处理数据。")
#             try:
#                 os.remove(save_path)
#                 print(f"已删除损坏的文件: {save_path}")
#             except:
#                 pass
#
#     all_data = []
#     # 获取目录下所有的 nc 文件，并按文件名排序
#     file_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')])
#     for file_path in file_paths:
#         print(f"Processing file: {file_path}")
#         single_day_data = process_single_file(file_path)
#         all_data.append(single_day_data)
#     # 合并所有天数的数据
#     all_data = np.concatenate(all_data, axis=0)
#
#     samples = []
#     # 生成样本
#     for i in range(len(all_data) - HISTORY_STEPS + 1):
#         sample = all_data[i:i + HISTORY_STEPS]
#         samples.append(sample)
#     samples = np.array(samples)
#     # 转换为 PyTorch 张量
#     tensor = torch.from_numpy(samples).float()
#
#     if save_path:
#         # 确保保存目录存在
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
#         # 首先尝试使用HDF5格式保存
#         try:
#             import h5py
#             h5_save_path = save_path.replace('.pt', '.h5')
#             print(f"正在使用HDF5格式保存预处理数据到: {h5_save_path}")
#             with h5py.File(h5_save_path, 'w') as f:
#                 # 创建数据集，启用压缩
#                 f.create_dataset('tensor', data=tensor.numpy(), compression='gzip', compression_opts=4)
#             print(f"成功保存预处理数据为HDF5格式")
#
#             # 也尝试使用PyTorch格式保存，但不阻止程序继续
#             try:
#                 torch.save(tensor, save_path)
#                 print(f"成功保存预处理数据为PyTorch张量格式")
#             except Exception as e:
#                 print(f"保存PyTorch格式出错: {e}")
#                 print("但HDF5格式已成功保存，可以继续使用")
#         except Exception as e:
#             print(f"保存张量出错: {e}")
#             # 如果HDF5保存失败，尝试分块保存
#             print("尝试分块保存...")
#             save_dir = os.path.dirname(save_path)
#             base_name = os.path.basename(save_path).split('.')[0]
#
#             # 获取张量的形状信息
#             shape_info = {
#                 'shape': tensor.shape,
#                 'dtype': str(tensor.dtype)
#             }
#             # 保存形状信息
#             info_path = os.path.join(save_dir, f"{base_name}_info.pt")
#             try:
#                 torch.save(shape_info, info_path)
#                 print(f"Saved shape info to {info_path}")
#             except:
#                 with open(os.path.join(save_dir, f"{base_name}_info.json"), 'w') as f:
#                     import json
#                     json.dump({'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}, f)
#
#             # 将张量分成多个块保存
#             chunk_size = 100  # 每个块的样本数，可以根据需要调整
#             for i in range(0, tensor.shape[0], chunk_size):
#                 end = min(i + chunk_size, tensor.shape[0])
#                 chunk = tensor[i:end]
#                 chunk_path = os.path.join(save_dir, f"{base_name}_chunk_{i}_{end}.npy")
#                 try:
#                     np.save(chunk_path, chunk.numpy())
#                     print(f"Saved chunk {i} to {end} as numpy array")
#                 except Exception as chunk_error:
#                     print(f"Error saving chunk {i} to {end}: {chunk_error}")
#
#             print(f"Saved tensor in chunks to {save_dir}")
#
#     return tensor
# # 示例调用
# data_directory = "E:/ERA5_China_2018"  # 替换为实际存储 nc 文件的目录
#
# processed_tensor_path = "E:/ERA5_China_2018_tensor/ERA5_China_2018_tensor.pt"
# tensor = process_year_data(data_directory,save_path=processed_tensor_path)
# print("转换后的 PyTorch 张量形状：", tensor.shape)
#
# def analyze_data(tensor):
#     stats={}
#     for var_idx, var in enumerate(VARS):
#         for lvl_idx, lvl in enumerate(PRESSURE_LEVELS):
#             ch_idx = var_idx*len(PRESSURE_LEVELS)+lvl_idx
#             data = tensor[:,:,ch_idx]
#
#             stats[f"{var}{lvl}"] = {
#                 "mean": data.mean().item(),
#                 "std": data.std().item(),
#                 "min": data.min().item(),
#                 "max": data.max().item(),
#                 "q95": torch.quantile(data, 0.95).item(),
#             }
#     return stats
# stats = analyze_data(tensor)
#
# import matplotlib.pyplot as plt
#
#
# def plot_distribution(tensor, var_name):
#     """绘制指定变量的分布直方图"""
#     var_idx = VARS.index(var_name)
#     fig, axes = plt.subplots(3, 5, figsize=(20, 12))
#
#     for i, lvl in enumerate([50, 200, 500, 700, 850, 925, 1000]):
#         ax = axes[i // 5, i % 5]
#         ch_idx = var_idx * len(PRESSURE_LEVELS) + PRESSURE_LEVELS.index(lvl)
#         data = tensor[:, :, ch_idx].flatten().numpy()
#
#         ax.hist(data, bins=50, density=True)
#         ax.set_title(f"{var_name}{lvl} (μ={data.mean():.2f}, σ={data.std():.2f})")
#
#     plt.tight_layout()
#     plt.show()
#
#
# plot_distribution(tensor, "t")  # 温度分布
# plot_distribution(tensor, "ciwc")  # 云冰分布
#
# # 对接近正态分布的数据进行归一化处理，Z-score
# def normalize_gaussian(tensor,var_name):
#     var_idx = VARS.index(var_name)
#     for lvl in PRESSURE_LEVELS:
#         ch_idx = var_idx*len(PRESSURE_LEVELS)+PRESSURE_LEVELS.index(lvl)
#         data = tensor[:,:,ch_idx]
#         mean, std = data.mean(), data.std()
#         tensor[:,:,ch_idx] = (data - mean)/std
#     return tensor
#
# normalized_tensor = normalize_gaussian(tensor, "u")
# normalized_tensor = normalize_gaussian(tensor, "v")
#
# # 对右偏移的数据进行幂变换处理右偏移分布
# def normalize_skewed(tensor,var_name,power=0.3):
#     var_idx = VARS.index(var_name)
#     for lvl in PRESSURE_LEVELS:
#         ch_idx = var_idx*len(PRESSURE_LEVELS)+PRESSURE_LEVELS.index(lvl)
#         data = tensor[:,:,ch_idx]
#         # Yeo-Johnson式变换
#         transformed = torch.sign(data) * torch.abs(data)**power
#         mean,std = transformed.mean(), transformed.std()
#         tensor[:,:,ch_idx] = (transformed - mean)/std
#     return tensor
# normalized_tensor = normalize_skewed(tensor, "t",power=0.4)
#
# def normalize_heavy_tail(tensor,var_name,clip_quantile=0.99):
#     var_idx = VARS.index(var_name)
#     for lvl in PRESSURE_LEVELS:
#         ch_idx =var_idx*len(PRESSURE_LEVELS)+PRESSURE_LEVELS.index(lvl)
#         data = tensor[:,:,ch_idx]
#
#         # 截断极端值
#         threshold = torch.quantile(data, clip_quantile)
#         clipped = torch.clamp(data,max = threshold)
#
#         #确保正数避免对数运算错误
#         epsilon = 1e-10
#         safe_data = torch.clamp(clipped, min=epsilon)*1000
#
#         # 对数处理变换
#         log_transformed = torch.log(safe_data)
#
#         mean,std = log_transformed.mean(), log_transformed.std()
#         tensor[:,:,ch_idx] = (log_transformed - mean)/std
#
#         if std == 0 or torch.isnan(std):
#             print(f"警告: {var_name}{lvl} 的标准差为零或NaN，跳过标准化")
#             tensor[:,:,ch_idx] = log_transformed - mean  # 只进行中心化
#         else:
#             tensor[:,:,ch_idx] = (log_transformed - mean)/std
#     return tensor
#
# def normalize_u_variable(tensor,var_name="u"):
#     var_idx = VARS.index(var_name)
#     for lvl in PRESSURE_LEVELS:
#         ch_idx = var_idx*len(PRESSURE_LEVELS)+PRESSURE_LEVELS.index(lvl)
#         data = tensor[:,:,ch_idx]
#
#         #利用Box-Cox变换
#         #尝试lambda值
#         lambda_value = 0.5
#
#         transformed = torch.sign(data) * torch.abs(data)**lambda_value
#
#         mean,std = transformed.mean(), transformed.std()
#         tensor[:,:,ch_idx] = (transformed - mean)/std
#     return tensor
# for var in  ['ciwc','clwc','cswc','q']:
#     normalized_tensor = normalize_heavy_tail(normalized_tensor,var)
#
# def binary_encoding(tensor,var_name,threshold=1e-6):
#     #将稀疏变量转化成二值特征
#     var_idx = VARS.index(var_name)
#     for lvl in PRESSURE_LEVELS:
#         ch_idx = var_idx*len(PRESSURE_LEVELS)+PRESSURE_LEVELS.index(lvl)
#         data = tensor[:,:,ch_idx]
#         #二值化处理
#         tensor[:,:,ch_idx] = (data > threshold).float()
#         print(f"{var_name}{lvl}的非零比例：{(data > threshold).float().mean().item():.4f}" )
#     return tensor
#
# def two_stage_encoding(tensor,var_name,thresholds=1e-6):
#     #创建两个通道，存在性和数值
#     var_idx = VARS.index(var_name)
#     value_dict = {}
#     for lvl in PRESSURE_LEVELS:
#         ch_idx = var_idx*len(PRESSURE_LEVELS)+PRESSURE_LEVELS.index(lvl)
#         data = tensor[:,:,ch_idx]
#
#         #存在性特征（二值）
#         existence = (data > thresholds).float()
#         #数值特征（标准化）
#         non_zero_mask = data > thresholds
#         value = torch.zeros_like(data)
#         if non_zero_mask.any():
#             non_zero_values = data[non_zero_mask]
#             log_values = torch.log(non_zero_values + thresholds)
#             mean,std = log_values.mean(), log_values.std()
#             if std > 0:
#                 normalized_values = (log_values - mean)/std
#             else:
#                 normalized_values = log_values - mean
#             value[non_zero_mask] = normalized_values
#             value_dict[lvl] = value[non_zero_mask]
#
#         #将两个特征合并
#         tensor[:,:,ch_idx] = existence
#         print(f"{var_name}{lvl}的非零比例：{existence.mean().item():.4f}" )
#
#     return tensor,value_dict
#
# # 可视化验证改进
# import statsmodels.api as sm
# import pylab as pl
# def plot_qq(tensor,var_name,level=500):
#     var_idx = VARS.index(var_name)
#     ch_idx = var_idx*len(PRESSURE_LEVELS)+PRESSURE_LEVELS.index(level)
#     data = tensor[:,:,ch_idx].flatten().numpy()
#
#     # 检查并过滤非有限值
#     valid_mask = np.isfinite(data)
#     valid_data = data[valid_mask]
#
#     if len(valid_data) < len(data):
#         print(f"警告: {var_name}{level} 数据中有 {len(data) - len(valid_data)} 个非有限值被过滤")
#
#     if len(valid_data) == 0:
#         print(f"错误: {var_name}{level} 数据中没有有效值，无法绘制QQ图")
#         return
#
#     from scipy import stats
#     _, (slope, intercept, r) = stats.probplot(valid_data, dist="norm", fit=True)
#     r_squared = r**2
#     sm.qqplot(valid_data, line='45', fit=True)
#     plt.title(f"{var_name}{level}\nR² = {r_squared:.4f}")
#     plt.grid(True)
#     pylab.show()
#
# def plot_all_qq_plots(tensor, save_dir=None):
#     """
#     为所有变量和气压层绘制QQ图
#
#     参数:
#     tensor: 包含所有通道数据的张量
#     save_dir: 如果提供，将图像保存到指定目录
#     """
#     if save_dir and not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)
#
#     r_squared_values = {}
#     for var_idx, var_name in enumerate(VARS): # Loop through variables
#         r_squared_values[var_name] = {}
#
#         # Create a figure for the current variable
#         fig, axes = plt.subplots(4, 4, figsize=(20, 16))
#         axes = axes.flatten()
#
#         # Set the main title for the figure (using English)
#         fig.suptitle(f"{var_name} QQ Plots for Each Pressure Level", fontsize=16)
#
#         for lvl_idx, level in enumerate(PRESSURE_LEVELS): # Loop through pressure levels
#             if lvl_idx < len(axes):
#                 ax = axes[lvl_idx]
#                 ch_idx = var_idx * len(PRESSURE_LEVELS) + PRESSURE_LEVELS.index(level)
#                 data = tensor[:, :, ch_idx].flatten().numpy()
#
#                 valid_mask = np.isfinite(data)
#                 valid_data = data[valid_mask]
#
#                 if len(valid_data) < len(data):
#                     print(f"Warning: {var_name}{level} data contains {len(data) - len(valid_data)} non-finite values, which were filtered out")
#
#                 if len(valid_data) == 0:
#                     print(f"Error: {var_name}{level} data has no valid values, cannot plot QQ plot")
#                     ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
#                     ax.set_title(f"{var_name}{level}")
#                     continue
#
#                 from scipy import stats
#                 _, (slope, intercept, r) = stats.probplot(valid_data, dist="norm", fit=True)
#                 r_squared = r**2
#                 r_squared_values[var_name][level] = r_squared
#
#                 # Plot QQ plot on the specific axis 'ax'
#                 sm.qqplot(valid_data, line='45', ax=ax)
#                 # Set subplot title including R²
#                 ax.set_title(f"{var_name}{level}\nR² = {r_squared:.4f}")
#                 ax.grid(True)
#
#         # Hide unused subplots
#         for i in range(len(PRESSURE_LEVELS), len(axes)):
#             axes[i].axis('off')
#
#         # Adjust layout after plotting all subplots
#         plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#
#         # Save or show the figure
#         if save_dir:
#             save_path = os.path.join(save_dir, f"{var_name}_qq_plots.png")
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             print(f"Saved QQ plot for {var_name} to {save_path}")
#         else:
#             plt.show() # Show the plot if not saving
#
#         # Explicitly close the current figure to free memory
#         plt.close(fig) # <--- Ensure this is called for every 'fig' created in the loop
#
#     return r_squared_values
#
# qq_plots_dir = "E:/qqplot"
# # Ensure the directory exists
# os.makedirs(qq_plots_dir, exist_ok=True)
# r_squared_values = plot_all_qq_plots(normalized_tensor, save_dir=qq_plots_dir)
#
# def analyze_normality(normalized_tensor):
#     print("各变量正态性分析")
#     for var_name,levels in r_squared_values.items():
#         avg_r2 = sum(levels.values()) / len(levels)
#         max_r2 = max(levels.values())
#         max_level = max(levels.items(), key=lambda x: x[1])[0]
#         min_r2 = min(levels.values())
#         min_level = min(levels.items(), key=lambda x: x[1])[0]
#
#         print(f"\n{var_name}:")
#         print(f"  平均R²: {avg_r2:.4f}")
#         print(f"  最佳正态性: {max_level}hPa (R² = {max_r2:.4f})")
#         print(f"  最差正态性: {min_level}hPa (R² = {min_r2:.4f})")
#
# analyze_normality(r_squared_values)
#
#
