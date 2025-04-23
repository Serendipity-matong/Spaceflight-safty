import glob
import os
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
lats = ds.latitude.values.tolist()
lons = ds.longitude.values.tolist()

time_dim = 'valid_time'
time_axis = ds.get_axis_num(time_dim)
T = ds.sizes[time_dim]

train_data = []
for i in range(146):
    slicer = [slice(None)] * ds.ndim
    if i==0:
        start_idx_print = -28
        end_idx_print = 'None'
        slicer[time_axis] = slice(-28, None)
    else:
        idx = i*40
        start_idx_print = -idx-28
        end_idx_print = -idx
        slicer[time_axis] = slice(-idx-28, -idx)
    
    print(f"切片索引: {start_idx_print} 到 {end_idx_print}" )

    try:
        data_slice = ds[tuple(slicer)].values
        train_data.append(data_slice)
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