# import os
# import xarray as xr
# import numpy as np
# from datetime import datetime, timedelta
#
#
# class ERA5DataProcessor:
#     """完全兼容版ERA5处理器（解决所有警告和错误）"""
#
#     # 变量映射（目录名 -> 标准变量名）
#     VAR_MAP = {
#         'geopotential': 'z',
#         'temperature': 't',
#         'u_component_of_wind': 'u',
#         'v_component_of_wind': 'v',
#         'specific_humidity': 'q',
#         'specific_cloud_ice_water_content': 'ciwc',
#         'specific_cloud_liquid_water_content': 'clwc',
#         'specific_rain_water_content': 'crwc',
#         'specific_snow_water_content': 'cswc'
#     }
#
#     # 标准气压层 (hPa) - 与数据完全一致
#     PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500,
#                        600, 700, 850, 925, 1000]
#
#     # 中国区域范围（使用原始维度名称）
#     CHINA_BBOX = {'latitude': slice(55, 10), 'longitude': slice(70, 140)}
#
#     def __init__(self, data_root):
#         self.data_root = data_root
#         # 自动检测存在的变量目录
#         self.var_dirs = [d for d in os.listdir(data_root)
#                          if os.path.isdir(os.path.join(data_root, d))]
#
#     def _get_nc_path(self, var_dir, year, date_str):
#         """构建NC文件路径"""
#         return os.path.join(self.data_root, var_dir, str(year), f"{date_str}.nc")
#
#     def load_single_day(self, year, month, day):
#         """
#         加载单日数据（完全兼容xarray未来版本）
#         :return: 合并后的xarray Dataset
#         """
#         date_str = f"{year}{month:02d}{day:02d}"
#         datasets = []
#
#         for var_dir in self.var_dirs:
#             nc_path = self._get_nc_path(var_dir, year, date_str)
#             if not os.path.exists(nc_path):
#                 print(f"警告: 文件不存在: {nc_path}")
#                 continue
#
#             try:
#                 # 保持原始维度名称加载
#                 ds = xr.open_dataset(nc_path)
#                 var_name = self.VAR_MAP.get(var_dir, var_dir)
#
#                 # 重命名变量（保持原始维度名）
#                 if len(ds.data_vars) == 1:
#                     ds = ds.rename({list(ds.data_vars)[0]: var_name})
#
#                 datasets.append(ds)
#             except Exception as e:
#                 print(f"跳过变量 {var_dir}: {str(e)}")
#                 continue
#
#         if not datasets:
#             raise ValueError(f"{date_str} 没有有效变量数据")
#
#         # 合并时保持原始维度名称
#         merged = xr.merge(datasets)
#
#         # 裁剪中国区域（使用原始维度名）
#         return merged.sel(
#             latitude=slice(55, 10),
#             longitude=slice(70, 140)
#         )
#
#     def create_training_samples(self, start_date, end_date, history_steps=2):
#         """
#         生成训练样本（完全兼容xarray未来版本）
#         :return: (samples, valid_dates) 元组
#         """
#         start = datetime.strptime(start_date, "%Y%m%d")
#         end = datetime.strptime(end_date, "%Y%m%d")
#         dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
#
#         # 预分配内存 (days, steps, vars, levels, lat, lon)
#         sample_shape = (len(dates), history_steps, len(self.VAR_MAP),
#                         len(self.PRESSURE_LEVELS), 46, 71)  # 中国区域裁剪后大小
#         inputs = np.full(sample_shape, np.nan, dtype=np.float32)
#         valid_dates = []
#
#         for i, date in enumerate(dates):
#             try:
#                 ds = self.load_single_day(date.year, date.month, date.day)
#
#                 # 使用sizes替代dims检查维度大小
#                 if ds.sizes['pressure_level'] != len(self.PRESSURE_LEVELS):
#                     raise ValueError(
#                         f"气压层数量不匹配: 预期{len(self.PRESSURE_LEVELS)}，实际{ds.sizes['pressure_level']}")
#
#                 for v, var in enumerate(self.VAR_MAP.values()):
#                     if var not in ds.data_vars:
#                         print(f"警告: 变量 {var} 不存在于 {date.date()}")
#                         continue
#
#                     var_data = ds[var]
#                     if 'pressure_level' in var_data.dims:
#                         # 使用气压层值直接选择
#                         for l, level in enumerate(self.PRESSURE_LEVELS):
#                             try:
#                                 level_data = var_data.sel(pressure_level=level)
#                                 inputs[i, :, v, l] = level_data.values[:history_steps]
#                             except Exception as e:
#                                 print(f"跳过{date.date()} {var} {level}hPa: {str(e)}")
#                     else:
#                         # 处理无气压层变量
#                         inputs[i, :, v, 0] = var_data.values[:history_steps]
#
#                 valid_dates.append(date)
#             except Exception as e:
#                 print(f"跳过{date.date()}: {str(e)}")
#                 continue
#
#         # 移除全NaN的样本
#         valid_mask = ~np.isnan(inputs).all(axis=(1, 2, 3, 4, 5))
#         return inputs[valid_mask], [d for i, d in enumerate(dates) if valid_mask[i]]
#
#     def save_daily_data(self, start_date, end_date, output_dir):
#         """
#         保存指定日期范围内的每日数据为NetCDF文件
#         :param start_date: 起始日期 (格式: "YYYYMMDD")
#         :param end_date: 结束日期 (格式: "YYYYMMDD")
#         :param output_dir: 输出目录路径
#         """
#         os.makedirs(output_dir, exist_ok=True)
#
#         start = datetime.strptime(start_date, "%Y%m%d")
#         end = datetime.strptime(end_date, "%Y%m%d")
#         dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
#
#         for date in dates:
#             date_str = date.strftime("%Y%m%d")
#             try:
#                 ds = self.load_single_day(date.year, date.month, date.day)
#                 output_path = os.path.join(output_dir, f"era5_china_{date_str}.nc")
#                 ds.to_netcdf(output_path)
#                 print(f"成功保存: {output_path}")
#             except Exception as e:
#                 print(f"保存{date_str}失败: {str(e)}")
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 初始化处理器
#     processor = ERA5DataProcessor("E:/PRESSURE_LEVEL")
#
#     # 保存2018年全年数据
#     output_directory = "E:/ERA5_China_2018"  # 设置你的输出目录
#     processor.save_daily_data("20180101", "20181231", output_directory)