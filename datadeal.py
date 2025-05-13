# Removed unused imports: _EndPositionT, T, end_fill
import numpy as np
import os
import glob  # 新增导入 glob 模块

# file = "/Users/fangzijie/Documents/process_data/1.npy" # 这行将被移除或注释，因为我们将从目录加载多个文件
# data = np.load(file) # 这行也将被移除或注释

MODEL_IMG_SIZE = (48, 72)
MODEL_IN_CHANS = 117
MODEL_OUT_CHANS = 30
MODEL_NUM_FORECAST_STEPS = 12  # Renamed from PREDICTION_HORIZON_STEPS for consistency if used in process_segment
INPUT_HISTORY_STEPS = 2

variable_map = [
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
pressure_levels = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
]
target_output_channel_space = [
    ('t', 200), ('t', 500), ('t', 700), ('t', 850), ('t', 1000),
    ('q', 200), ('q', 500), ('q', 700), ('q', 850), ('q', 1000),
    ('ciwc', 200), ('ciwc', 500), ('ciwc', 700), ('ciwc', 850), ('ciwc', 1000),
    ('clwc', 200), ('clwc', 500), ('clwc', 700), ('clwc', 850), ('clwc', 1000),
    ('crwc', 200), ('crwc', 500), ('crwc', 700), ('crwc', 850), ('crwc', 1000),
    ('cswc', 200), ('cswc', 500), ('cswc', 700), ('cswc', 850), ('cswc', 1000)
]
var_short_map = {
    't': 'temperature',
    'q': 'specific_humidity',
    'ciwc': 'specific_cloud_ice_water_content',
    'clwc': 'specific_cloud_liquid_water_content',
    'crwc': 'specific_rain_water_content',
    'cswc': 'specific_snow_water_content',
    'z': 'geopotential',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind'
}
# target_output_channel is defined but seems unused later.
# target_output_channel_space is used in get_target_channel_indices call.
target_output_channel = [
    (var_short_map[spec[0]], spec[1]) for spec in target_output_channel_space
]
target_lat_range = (10, 55)  # (South_Lat, North_Lat) -> 10N to 55N
target_lon_range = (70, 140)  # (West_Lon, East_Lon) -> 70E to 140E


# 假设已定义 LATITUDE_COORDS_IN_DATA 和 LONGITUDE_COORDS_IN_DATA

def get_spatial_slices(lat_coords, lon_coords, lat_range_deg, lon_range_deg):
    # 纬度处理 (假设 lat_coords 是从北到南递减, e.g., 90, 89, ..., -90)
    # lat_range_deg = (南边界, 北边界)
    lat_south_deg, lat_north_deg = min(lat_range_deg), max(lat_range_deg)

    lat_idx_north = np.argmin(np.abs(lat_coords - lat_north_deg))
    lat_idx_south = np.argmin(np.abs(lat_coords - lat_south_deg))
    lat_slice = slice(min(lat_idx_north, lat_idx_south), max(lat_idx_north, lat_idx_south) + 1)

    lon_west_deg, lon_east_deg = min(lon_range_deg), max(lon_range_deg)
    lon_idx_west = np.argmin(np.abs(lon_coords - lon_west_deg))
    lon_idx_east = np.argmin(np.abs(lon_coords - lon_east_deg))
    lon_slice = slice(min(lon_idx_west, lon_idx_east), max(lon_idx_west, lon_idx_east) + 1)

    return lat_slice, lon_slice


# Updated function definition to match the call and logic
def get_target_channel_indices(target_specs_short_names_list,
                               vars_short_to_full_map,
                               all_vars_full_names_list,
                               all_pressure_levels_list):
    var_full_name_to_idx = {name: i for i, name in enumerate(all_vars_full_names_list)}
    plev_to_idx = {level: i for i, level in enumerate(all_pressure_levels_list)}

    calculated_target_indices = []
    num_levels_per_var = len(all_pressure_levels_list)

    for var_short_key, plev_val in target_specs_short_names_list:
        full_var_name = vars_short_to_full_map.get(var_short_key)
        if full_var_name is None:
            raise ValueError(f"短变量名 '{var_short_key}' 未在 var_short_map 中找到。")

        var_idx_in_all_vars = var_full_name_to_idx.get(full_var_name)
        if var_idx_in_all_vars is None:
            raise ValueError(
                f"完整目标变量名 '{full_var_name}' (来自 '{var_short_key}') 未在 all_vars_full_names_list (variable_map) 中找到。")

        if plev_val not in plev_to_idx:
            raise ValueError(f"目标气压层 '{plev_val}hPa' 未在 all_pressure_levels_list (pressure_levels) 中找到。")

        plev_idx_in_all_plevs = plev_to_idx[plev_val]
        flat_channel_idx = var_idx_in_all_vars * num_levels_per_var + plev_idx_in_all_plevs
        calculated_target_indices.append(flat_channel_idx)

    # Moved print outside the loop and ensure unique sorted indices
    unique_sorted_indices = sorted(list(set(calculated_target_indices)))
    print(f"计算得到的目标通道索引 (共 {len(unique_sorted_indices)} 个):")
    print(unique_sorted_indices)
    return unique_sorted_indices


# 主处理函数
def process_segment(segment_data, lat_coords, lon_coords, target_lat_range, target_lon_range, y_channel_indices,
                    mode_img_sz, mode_in_ch, mode_out_ch):
    # 确保传入的 segment_data 是 6D: (days, intraday_steps, num_vars, num_levels, H, W)
    if segment_data.ndim != 6:
        raise ValueError(
            f"process_segment 预期 segment_data 为6维 (days, intraday_steps, vars, levels, H, W)，但得到 {segment_data.ndim} 维。")

    days, intreday_steps, num_vars, num_levels, H, W = segment_data.shape

    if mode_in_ch != num_vars * num_levels:
        raise ValueError(f"传入的 mode_in_ch ({mode_in_ch}) 与数据本身的通道数 ({num_vars * num_levels}) 不符。")

    lat_slicer, lon_slicer = get_spatial_slices(lat_coords, lon_coords, target_lat_range, target_lon_range)
    snapshots_all_fields = segment_data.reshape(days * intreday_steps, num_vars, num_levels, H, W)
    total_snapshots = snapshots_all_fields.shape[0]
    # Reshape using mode_in_ch for consistency
    snapshots_channels = snapshots_all_fields.reshape(total_snapshots, mode_in_ch, H, W)

    X_simple_list = []
    Y_simple_list = []
    # Corrected typo: num_generatable_samples
    num_generatable_samples = total_snapshots - INPUT_HISTORY_STEPS - MODEL_NUM_FORECAST_STEPS + 1
    if num_generatable_samples < 1:
        print("警告: 数据不足以生成任何样本，请检查输入数据长度和定义的步长。")
        return np.array(X_simple_list), np.array(Y_simple_list)  # Return empty arrays
    for i in range(num_generatable_samples):
        X_sample_full_channels = snapshots_channels[i:i + INPUT_HISTORY_STEPS, :, lat_slicer, lon_slicer]
        # Corrected list name
        X_simple_list.append(X_sample_full_channels)
        Y_sample_full_channels = snapshots_channels[
                                 i + INPUT_HISTORY_STEPS:i + INPUT_HISTORY_STEPS + MODEL_NUM_FORECAST_STEPS,
                                 y_channel_indices, lat_slicer, lon_slicer]
        # Corrected list name
        Y_simple_list.append(Y_sample_full_channels)

    if not X_simple_list:  # Handle case where loop might not run but num_generatable_samples was >=1
        return np.array([]), np.array([])

    # Convert to numpy arrays before returning
    return np.array(X_simple_list, dtype=np.float32), np.array(Y_simple_list, dtype=np.float32)


if __name__ == "__main__":
    # 定义包含数据段 .npy 文件的目录
    # 请将 "/path/to/your/segment_files/" 替换为实际的目录路径
    segment_data_dir = "/Users/fangzijie/Documents/process_data/"  # 示例路径，请修改

    # 查找目录中所有的 .npy 文件
    segment_files = sorted(glob.glob(os.path.join(segment_data_dir, "*.npy")))

    if not segment_files:
        raise FileNotFoundError(f"在目录 '{segment_data_dir}' 中没有找到 .npy 文件。")

    num_segments = len(segment_files)
    print(f"找到了 {num_segments} 个数据段文件。")

    # 从第一个数据段文件获取维度信息 (假设所有文件维度一致)
    # 我们需要先加载第一个文件来确定 ORIGINAL_LAT_COUNT 和 ORIGINAL_LON_COUNT
    try:
        first_segment_data = np.load(segment_files[0])
        if first_segment_data.ndim != 6:
            raise ValueError(f"第一个数据文件 '{segment_files[0]}' 不是预期的6维。实际维度: {first_segment_data.ndim}")
        # 预期维度: (days, intraday_steps, vars, levels, H_orig, W_orig)
        ORIGINAL_LAT_COUNT = first_segment_data.shape[4]
        ORIGINAL_LON_COUNT = first_segment_data.shape[5]
        del first_segment_data  # 释放内存
    except Exception as e:
        print(f"加载第一个数据文件 '{segment_files[0]}' 以获取维度信息时出错: {e}")
        exit()

    LATITUDE_COORDS_IN_DATA = np.linspace(90, -90, ORIGINAL_LAT_COUNT, endpoint=True)
    LONGITUDE_COORDS_IN_DATA = np.linspace(0, 360 - (360 / ORIGINAL_LON_COUNT), ORIGINAL_LON_COUNT, endpoint=True)

    print(f"每个数据段的原始纬度点数: {ORIGINAL_LAT_COUNT}, 经度点数: {ORIGINAL_LON_COUNT}")

    print("计算目标Y通道的索引...")
    y_indices = get_target_channel_indices(
        target_specs_short_names_list=target_output_channel_space,
        vars_short_to_full_map=var_short_map,
        all_vars_full_names_list=variable_map,
        all_pressure_levels_list=pressure_levels
    )
    print(f"已计算 {len(y_indices)} 个目标Y通道索引 (相对于{MODEL_IN_CHANS}个原始通道): {y_indices}")

    all_X_samples = []
    all_Y_samples = []

    print(f"开始处理 {num_segments} 个数据段文件...")
    for seg_idx, file_path in enumerate(segment_files):
        print(f"  处理数据段文件 {seg_idx + 1}/{num_segments}: {file_path}...")
        try:
            current_segment_data = np.load(file_path)
            # 验证每个加载的数据段是否为6维
            if current_segment_data.ndim != 6:
                print(f"    警告: 文件 {file_path} 的数据不是6维 (实际为 {current_segment_data.ndim}维)，已跳过。")
                continue

            # 验证空间维度是否与第一个文件一致
            if current_segment_data.shape[4] != ORIGINAL_LAT_COUNT or current_segment_data.shape[
                5] != ORIGINAL_LON_COUNT:
                print(
                    f"    警告: 文件 {file_path} 的空间维度 ({current_segment_data.shape[4]}x{current_segment_data.shape[5]}) 与预期 ({ORIGINAL_LAT_COUNT}x{ORIGINAL_LON_COUNT}) 不符，已跳过。")
                continue

        except Exception as e:
            print(f"    加载文件 {file_path} 时出错: {e}，已跳过。")
            continue

        X_from_segment, Y_from_segment = process_segment(
            segment_data=current_segment_data,
            lat_coords=LATITUDE_COORDS_IN_DATA,
            lon_coords=LONGITUDE_COORDS_IN_DATA,
            target_lat_range=target_lat_range,
            target_lon_range=target_lon_range,
            y_channel_indices=y_indices,
            mode_img_sz=MODEL_IMG_SIZE,
            mode_in_ch=MODEL_IN_CHANS,
            mode_out_ch=MODEL_OUT_CHANS
        )
        if X_from_segment.size > 0:
            all_X_samples.append(X_from_segment)
            all_Y_samples.append(Y_from_segment)
            print(f"    从数据段 {seg_idx + 1} 生成了 {X_from_segment.shape[0]} 个 (X,Y) 样本对。")
        else:
            print(f"    数据段 {seg_idx + 1} 未能生成样本（可能太短或处理逻辑问题）。")

    if not all_X_samples:
        print("未能从任何数据段生成任何处理后的数据。")
    else:
        final_X_dataset = np.concatenate(all_X_samples, axis=0)
        final_Y_dataset = np.concatenate(all_Y_samples, axis=0)

        print(f"\n所有数据段处理完成!")
        print(f"  最终生成的 X 数据集形状: {final_X_dataset.shape}")
        print(f"  最终生成的 Y 数据集形状: {final_Y_dataset.shape}")
        # 可选: 保存处理后的数据
        output_dir = "./processed_data_corrected"
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "X_dataset.npy"), final_X_dataset)
        np.save(os.path.join(output_dir, "Y_dataset.npy"), final_Y_dataset)
        print(f"已将处理后的数据集保存到目录: {output_dir}")