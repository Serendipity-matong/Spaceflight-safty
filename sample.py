import glob
import os
import sys

from xarray.core.duck_array_ops import var

import numpy as np # 建议使用 np 别名
import xarray as xr
# import cupy as cp
import pickle
# 定义包含数据的根目录和变量列表
base_path = '/Users/fangzijie/Documents/pressure_level'
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

    # 检查是否有索引文件
    index_file = os.path.join(base_path, f"dataset_idx_{year}.pkl")
    try_load_index = False

    if os.path.exists(index_file) and os.path.getsize(index_file) > 0:
        print(f"找到索引文件: {index_file}，尝试使用索引加载数据...")
        try:
            with open(index_file, 'rb') as fp:
                variable_offsets = pickle.load(fp)
            try_load_index = True
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"索引文件损坏或为空: {e}")
            print("将删除损坏的索引文件并重新加载数据...")
            try:
                os.remove(index_file)
            except Exception as e:
                print(f"无法删除损坏的索引文件: {e}")

    if try_load_index:
        # 使用索引加载数据
        for var_name in variables:
            if var_name not in variable_offsets:
                print(f"索引中没有变量 '{var_name}'，跳过...")
                continue

            data_dir = os.path.join(base_path, var_name, str(year))
            if not os.path.isdir(data_dir):
                print(f"目录不存在：{data_dir}")
                continue

            try:
                print(f"    使用索引加载变量: {var_name}...")
                # 使用索引信息加载数据
                nc_files = sorted(glob.glob(os.path.join(data_dir, '*.nc')))
                file_offsets = variable_offsets[var_name]

                # 创建延迟加载的数据集
                ds_var = xr.open_mfdataset(
                    nc_files,
                    combine='by_coords',
                    parallel=False  # <--- 修改这里
                )
                # 加载后再进行分块
                ds_var = ds_var.chunk({'valid_time': 1})
                datasets_to_merge.append(ds_var)
                print(f"      加载成功: {var_name}")
            except Exception as e:
                print(f"    加载变量 '{var_name}' 时出错: {e}")
    else:
        print(f"未找到有效的索引文件，跳过索引创建，直接加载数据...")

        # 初始化变量偏移字典
        variable_offsets = {}

        # 直接加载数据，不创建索引
        for var_name in variables:
            data_dir = os.path.join(base_path, var_name, str(year))
            print(f"检查目录：{data_dir}")
            if not os.path.isdir(data_dir):
                print(f"目录不存在：{data_dir}")
                continue

            nc_files = sorted(glob.glob(os.path.join(data_dir, '*.nc')))
            if not nc_files:
                print(f"目录中没有 .nc 文件: {data_dir}")
                continue

            try:
                print(f"    正在加载变量: {var_name} 从 {len(nc_files)} 个 .nc 文件...")

                # 直接使用延迟加载，不创建索引
                ds_var = xr.open_mfdataset(
                    nc_files,
                    combine='by_coords',
                    parallel=False  # <--- 修改这里
                )
                # 加载后再进行分块
                ds_var = ds_var.chunk({'valid_time': 1})
                datasets_to_merge.append(ds_var)
                print(f"      加载成功: {var_name}")
            except Exception as e:
                print(f"    加载变量 '{var_name}' 的 .nc 文件时出错: {e}")
                print(f"      目录: {data_dir}")

        # 删除重复的索引创建代码块，避免重复加载
        # 这里原来有一个重复的循环，现在已删除

        # 保存索引文件 - 这里我们不再需要创建索引
        # try:
        #     with open(index_file, 'wb') as f:
        #         pickle.dump(variable_offsets, f)
        #     print(f"索引文件已保存到: {index_file}")
        # except Exception as e:
        #     print(f"保存索引文件时出错: {e}")

    if not datasets_to_merge:
        print("错误：未能加载任何数据集。")
        return None

    # 合并所有加载的数据集
    print("\n正在合并所有变量的数据集...")
    try:
        # 使用延迟计算合并数据集
        merged_ds = xr.merge(datasets_to_merge)
        print("合并完成。")
    except xr.MergeError as e:
        print(f"合并数据集时出错 (MergeError): {e}")
        return None
    except Exception as e:
        print(f"合并数据集时出错: {e}")
        return None

    # 对合并后的数据集进行分块
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
N_TIME_STEPS_SLICE = 4  # 修改为一天的时间步数
N_LEVELS =  ds.sizes['pressure_level']
N_LAT = ds.sizes['latitude']
N_LON = ds.sizes['longitude']
N_CHANNELS = N_VARS * N_LEVELS

lats = ds.latitude.values.tolist()
lons = ds.longitude.values.tolist()

BATCH_SIZE = 10
NUM_BATCHES = 37  # 修改为37，这样可以处理约370个样本，覆盖全年

output_dif = "/Users/fangzijie/Documents/process_data"
os.makedirs(output_dif, exist_ok=True)

time_dim = 'valid_time'
time_axis = ds.get_axis_num(time_dim)
T = ds.sizes[time_dim]

lation_vals = []
for i in range(N_LAT):
    for j in range(N_LON):
        lation_vals.append([lats[i],lons[j]])
lation_vals = np.array(lation_vals)

from tqdm import tqdm
import gc

for batch in range(NUM_BATCHES):
    print(f"\n开始处理第 {batch + 1} 批次...")
    # 1. 确定批次的时间范围
    start_idx = batch * BATCH_SIZE
    end_idx = min((batch + 1) * BATCH_SIZE, 365)  # 修改为365
    train_data = []

    for i in range(start_idx, end_idx):
        slicer = [slice(None)] * ds.ndim
        print(f"\n处理第 {i + 1} 个时间步...")
        # 2. 提取当前时间步的数据
        if i==0:
            start_index = -N_TIME_STEPS_SLICE
            end_idx_print = 'None'
            slicer[time_axis] = slice(start_index, None)
        else:
            start_index = -i - N_TIME_STEPS_SLICE
            end_idx_print = -i
            slicer[time_axis] = slice(start_index, end_idx_print)

        try:
            data_slice_original = ds[tuple(slicer)].values
            data_slice_time = np.transpose(data_slice_original, (1,0,2,3,4))
            # 重塑为(时间步, 变量, 气压层, 纬度, 经度)的结构
            data_slice_reshaped = data_slice_time.reshape(N_TIME_STEPS_SLICE, N_VARS, N_LEVELS, N_LAT, N_LON)
            train_data.append(data_slice_reshaped)
            print(f"成功加载样本 {i+1}")
        except MemoryError:
            print(f"\n错误：内存不足，无法加载样本 {i+1}，跳过此样本。")
            continue  # 跳过此样本而不是终止程序
        except Exception as e:
            print(f"\n处理样本 {i+1} 时发生错误: {e}，跳过此样本。")

    print(f"当前批次成功提取{len(train_data)}个样本。")

    if not train_data:
        print("\n当前批次没有可处理的样本，跳过此批次。")
        continue


    # ... (您之前的代码) ...

    # 假设 train_data 是这样定义的列表:
    # train_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] 或其他包含数值的列表

    # 将 train_data 列表转换为 NumPy 数组并指定数据类型为 float32
    train_data = np.array(train_data, dtype=np.float32)

    # ... (您之后的代码) ...
    train_data.astype(np.float32)
    batch_output_file = os.path.join(output_dif,f"{batch+1}.npy")
    np.save(batch_output_file, train_data)
    gc.collect()
# 8. 合并所有样本的数据
print("\n合并所有样本的处理结果...")
# final_vals 形状: (num_samples * 781920, 384) = (357 * 781920, 384) approx (279 million, 384)
# final_vals = np.concatenate(final_vals_list, axis=0)

# print(f"\n特征矩阵构建完成。最终形状: {final_vals.shape}")


#
#
# from main import N_TARGET_CHANNELS
# import numpy as np
# import os
# # os
# # N_VARS = 9
# # N_LEVELS = 13
# # N_CHANNELS = N_VARS * N_LEVELS
# # N_TARGET_CHANNELS = 30
#
# data_dir = "/Users/fangzijie/Documents/processed_data"
# file_dir = "batch_1.npy"
#
# file_path = os.path.join(data_dir, file_dir)
# data = np.load(file_path)
# print(f"\n数据的形状:{data.shape}")
# 定义中国区域的经纬度范围
