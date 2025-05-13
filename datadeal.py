import numpy as np
import os

file = "/Users/fangzijie/Documents/process_data/1.npy"
data = np.load(file)  # 直接使用加载的数据，不要 reshape 成 (10,)
# data = data.reshape(10,) # <--- 这行应该删除或注释掉

MODEL_IMG_SIZE = (48, 72)
MODEL_IN_CHANS = 117
MODEL_OUT_CHANS = 30
MODEL_NUM_FORECAST_STEPS = 12
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
target_output_channel = [
    (var_short_map[spec[0]], spec[1]) for spec in target_output_channel_space  # <--- 修正 "foe" 为 "for"
]
target_lat_range = (10, 55)  # (South_Lat, North_Lat) -> 9N to 56N
target_lon_range = (70, 140)  # (West_Lon, East_Lon) -> 69E to 140E


# 假设已定义 LATITUDE_COORDS_IN_DATA 和 LONGITUDE_COORDS_IN_DATA

def get_spatial_slices(lat_coords, lon_coords, lat_range_deg, lon_range_deg):
    # 纬度处理 (假设 lat_coords 是从北到南递减, e.g., 90, 89, ..., -90)
    # lat_range_deg = (南边界, 北边界)
    lat_south_deg, lat_north_deg = min(lat_range_deg), max(lat_range_deg)

    lat_idx_north = np.argmin(np.abs(lat_coords - lat_north_deg))
    lat_idx_south = np.argmin(np.abs(lat_coords - lat_south_deg))
    # slice 会自动处理索引顺序，但确保 min_idx <= max_idx
    lat_slice = slice(min(lat_idx_north, lat_idx_south), max(lat_idx_north, lat_idx_south) + 1)

    # 经度处理 (假设 lon_coords 是从西到东递增, e.g., 0, 1, ..., 359)
    # lon_range_deg = (西边界, 东边界)
    lon_west_deg, lon_east_deg = min(lon_range_deg), max(lon_range_deg)
    lon_idx_west = np.argmin(np.abs(lon_coords - lon_west_deg))
    lon_idx_east = np.argmin(np.abs(lon_coords - lon_east_deg))
    lon_slice = slice(min(lon_idx_west, lon_idx_east), max(lon_idx_west, lon_idx_east) + 1)

    return lat_slice, lon_slice


def get_target_channel_indices(target_output_channel_space, variable_map, pressure_levels):
    var_idx = {name: i for i, name in enumerate(variable_map)}
    level_idx = {level: i for i, level in enumerate(pressure_levels)}
    target_channel_indices = []
    for var_key, level in target_output_channel_space:
        v_idx = var_idx[var_key]
        l_idx = level_idx[level]
        channel_index = v_idx * len(pressure_levels) + l_idx
        target_channel_indices.append(channel_index)
        print(f"计算得到的目标通道索引 (共 {len(target_channel_indices)} 个):")
        print(target_channel_indices)
    return target_channel_indices


# 主处理函数
def process_segment(segment_data, lat_coords, lon_coords, target_lat_range, target_lon_range, y_channel_indices,
                    mode_img_sz, mode_in_ch, mode_out_ch):
    days, intreday_steps, num_vars, num_levels, H, W = segment_data.shape
    lat_slicer, lon_slicer = get_spatial_slices(lat_coords, lon_coords, target_lat_range, target_lon_range)
    snapshots_all_fields = segment_data.reshape(days * intreday_steps, num_vars, num_levels, H, W)
    total_snapshots = snapshots_all_fields.shape[0]
    snapshots_channels = snapshots_all_fields.reshape(total_snapshots, num_vars * num_levels, H, W)

    X_simple_list = []
    Y_simple_list = []
    num_generatable_samples = total_snapshots - INPUT_HISTORY_STEPS - MODEL_NUM_FORECAST_STEPS + 1
    if num_generatable_samples < 1:
        print("警告: 数据不足以生成任何样本，请检查输入数据长度和定义的步长。")
        return np.array(X_simple_list), np.array(Y_simple_list)
    for i in range(num_generatable_samples):
        X_sample_full_channels = snapshots_channels[i:i + INPUT_HISTORY_STEPS, :, lat_slicer, lon_slicer]
        X_simple_list.append(X_sample_full_channels)
        Y_sample_full_channels = snapshots_channels[
                                 i + INPUT_HISTORY_STEPS:i + INPUT_HISTORY_STEPS + MODEL_NUM_FORECAST_STEPS,
                                 y_channel_indices, lat_slicer, lon_slicer]
        Y_simple_list.append(Y_sample_full_channels)

    return np.array(X_simple_list), np.array(Y_simple_list)

