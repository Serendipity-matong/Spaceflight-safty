import glob
import os
import sys

from xarray.core.duck_array_ops import var

import numpy as np # 建议使用 np 别名
import xarray as xr
# import cupy as cp

# 定义包含数据的根目录和变量列表
base_path = 'E:/pressure_level'
variables = [
    'geopotential', 
    'specific_cloud_ice_water_content', 
    'specific_cloud_liquid_water_content',
    'specific_humidity', 
    'specific_rain_water_content', 
    'specific_snow_water_content',
    'temperature', 
    'u_component_of_wind', 
    'v_component_of_wind'
]
year = 2018 # 指定要加载的年份

def chunk_time(ds):
    dims = {k:v for k,v in ds.dims.items()}
    #将数据分为一个时间步
    dims['valid_time'] = 1
    ds = ds.chunk(dims)
    return ds

def load_data(base_path, variables, year):
    # 构建文件路径
    datasets_to_merge = []
    print(f"开始加载年份数据")
    
    for var_name in variables:
        data_dir = os.path.join(base_path,var_name,str(year))
        print(f"检查目录：{data_dir}")
        if not os.path.isdir(data_dir):
            print(f"目录不存在：{data_dir}")
            continue
        
        nc_files = sorted(glob.glob(os.path.join(data_dir, '*.nc')))

        try:
             print(f"    正在加载变量: {var_name} 从 {len(nc_files)} 个 .nc 文件...")
             ds_var = xr.open_mfdataset(nc_files, combine='by_coords', parallel=True) 
             datasets_to_merge.append(ds_var)
             print(f"      加载成功: {var_name}") 
        except Exception as e:
            print(f"    加载变量 '{var_name}' 的 .nc 文件时出错: {e}")
            print(f"      目录: {data_dir}")
    if not datasets_to_merge:
        print("错误：未能加载任何数据集。")
        return None

    # 合并所有加载的数据集
    print("\n正在合并所有变量的数据集...")
    try:
        # 注意：如果 open_mfdataset 已经处理了分块，这里的 merge 可能需要更多内存
        merged_ds = xr.merge(datasets_to_merge)
        print("合并完成。")
    except xr.MergeError as e:
        # ... (错误处理保持不变) ...
        return None
    except Exception as e:
        # ... (错误处理保持不变) ...
        return None

    # 对合并后的数据集进行分块 (如果 open_mfdataset 时未分块)
    print("\n正在对合并后的数据集进行时间分块...")
    final_ds = chunk_time(merged_ds)
    print("分块完成。")
    
    return final_ds
# 调用 load_data 函数加载数据
ds_dataset = load_data(base_path, variables, year)

ds = ds_dataset.to_array(dim='variable',name='era5_data')
del ds_dataset
# ds = cp.asarray(ds.compute().values)
print(ds.shape)
# 检查 ds 是否成功加载
if ds is not None:
    print("\n最终数据集信息:")
    print(ds)
    
    # 现在可以查看长度和变量了
#     print("\n时间维度长度:", ds['valid_time'])
#     print("\n数据变量名称:", list(ds.data_vars))
# else:
#     print("\n数据加载失败。")
# ds = ds[:,:,:]

level_coords = ds.pressure_level.values
try:
    l_idx_200 = np.where(level_coords == 200)[0][0]
    l_idx_500 = np.where(level_coords == 500)[0][0]
    l_idx_700 = np.where(level_coords == 700)[0][0]
    l_idx_850 = np.where(level_coords == 850)[0][0]
    l_idx_1000 = np.where(level_coords == 1000)[0][0]
    level_indices = {
        200: l_idx_200,
        500: l_idx_500,
        700: l_idx_700,
        850: l_idx_850,
        1000: l_idx_1000
    }
except IndexError:
    print("\n错误：指定的级别坐标不存在于数据中。")
    sys.exit(1)

variable_map = {name: i for i, name in enumerate(variables)}
var_indices = {
    't': variable_map['temperature'],
    'q': variable_map['specific_humidity'],
    'ciwc': variable_map['specific_cloud_ice_water_content'],
    'clwc': variable_map['specific_cloud_liquid_water_content'],
    'crwc': variable_map['specific_rain_water_content'],
    'cswc': variable_map['specific_snow_water_content']
}

target_channel_indices = []
target_levels = [200, 500, 700, 850, 1000]

for var_key in ['t','q','ciwc','clwc','crwc','cswc']:
    v_idx = var_indices[var_key]
    for level in target_levels:
        l_idx = level_indices[level]
        channel_index = v_idx *13 + l_idx
        target_channel_indices.append((channel_index))

print(f"计算得到的目标通道索引 (共 {len(target_channel_indices)} 个):")
print(target_channel_indices)

# 更新目标通道数量
N_VARS = ds.sizes['variable']
N_TARGET_CHANNELS = len(target_channel_indices)
N_TIME_STEPS_SLICE = 28
N_LEVELS =  ds.sizes['pressure_level']
N_LAT = ds.sizes['latitude']
N_LON = ds.sizes['longitude']
N_CHANNELS = N_VARS * N_LEVELS

lats = ds.latitude.values.tolist()
lons = ds.longitude.values.tolist()

time_dim = 'valid_time'
time_axis = ds.get_axis_num(time_dim)
T = ds.sizes[time_dim]

train_data = []
for i in range(146):
    slicer = [slice(None)] * ds.ndim
    if i==0:
        start_idx_print = -N_TIME_STEPS_SLICE
        end_idx_print = 'None'
        slicer[time_axis] = slice(-N_TIME_STEPS_SLICE, None)
    else:
        idx = i*40
        start_idx_print = -idx-N_TIME_STEPS_SLICE
        end_idx_print = -idx
        slicer[time_axis] = slice(-idx-N_TIME_STEPS_SLICE, -idx)
    
    print(f"切片索引: {start_idx_print} 到 {end_idx_print}" )

    try:
        data_slice_original = ds[tuple(slicer)].values
        data_slice_time_first = np.transpose(data_slice_original, (1,0,2,3,4))
        data_slice_reshaped = data_slice_time_first.reshape(N_TIME_STEPS_SLICE,N_CHANNELS,N_LAT,N_LON)
    except MemoryError:
        print(f"\n错误：内存不足，无法加载样本 {i+1} 的数据！程序终止。")
        sys.exit(1) # 内存不足时直接退出
    except Exception as e:
        print(f"\n处理样本 {i+1} 时发生错误: {e} 程序终止。")
        sys.exit(1) # 其他错误也退出
print(f"\n成功提取 {len(train_data)} 个训练样本。")

# --- 后续可以使用 train_data 列表 ---
# 例如：打印第一个样本的形状
if train_data:
    print(f"第一个样本形状: {train_data[0].shape}")

lation_vals = []
for i in range(181):
    for j in range(360):
        lation_vals.append([lats[i],lons[j]])
lation_vals = np.array(lation_vals)
print(lation_vals)

from tqdm import tqdm
import gc

N_POINTS = N_LAT * N_LON

final_vals_list = []

for flag,dat in tqdm(enumerate(train_data),total=len(train_data),desc = "处理样本"):
    first_feat = dat[0,:,:,:].reshape(N_CHANNELS,N_POINTS).transpose()
    second_feat = dat[1,:,:,:].reshape(N_CHANNELS,N_POINTS).transpose()
    diff_feat = second_feat - first_feat
    final_vals = (dat[1, :, :, :] - dat[0, :, :, :]).reshape(N_CHANNELS, N_POINTS).transpose()

    all_vals = []
    for i in range(12):
        current_idx = i + 2

        time_vals = np.full((N_POINTS,1),i)
        sub_vals_features = np.concatenate((time_vals, lation_vals, first_feat, second_feat, diff_feat), axis=1)

        # 4. 提取当前未来时间步的目标变量
        # 取出 dat 中当前预测时间步的数据，形状: (117, 181, 360)
        current_step_data_full = dat[current_idx, :, :, :]
        # 使用 target_channel_indices 提取目标通道，形状: (30, 181, 360)
        target_data_channels = current_step_data_full[target_channel_indices, :, :]
        # Reshape 和 Transpose -> (65160, 30)
        target_vals = target_data_channels.reshape(N_TARGET_CHANNELS, N_POINTS).transpose()

        # 5. 拼接特征和目标变量
        # sub_vals 形状: (65160, 354 + 30) = (65160, 384)
        sub_vals = np.concatenate((sub_vals_features, target_vals), axis=1)

        # 6. 存储
        all_vals.append(sub_vals)

    # 7. 合并当前样本的所有预测时间步数据
    # all_vals 形状: (12 * 65160, 384) = (781920, 384)
    all_vals = np.concatenate(all_vals, axis=0)
    final_vals_list.append(all_vals)

    # 清理内存
    del first_feat, second_feat, diff_feat, all_vals
    del time_vals, sub_vals_features, current_step_data_full, target_data_channels, target_vals, sub_vals, all_vals
    gc.collect()

# 8. 合并所有样本的数据
print("\n合并所有样本的处理结果...")
# final_vals 形状: (num_samples * 781920, 384) = (357 * 781920, 384) approx (279 million, 384)
final_vals = np.concatenate(final_vals_list, axis=0)

print(f"\n特征矩阵构建完成。最终形状: {final_vals.shape}")


    
   