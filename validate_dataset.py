def check_integrity(year):
    """验证全年数据完整性"""
    missing_days = []
    for month in range(1, 13):
        pt_file = Path(f"processed/{year}_{month:02d}.pt")
        if not pt_file.exists():
            print(f"缺失文件: {pt_file}")
            continue

        data = torch.load(pt_file)
        expected_shape = (days_in_month * 4, 9, 13, 181, 360)
        if data.shape != expected_shape:
            print(f"形状异常: {pt_file} 应为{expected_shape}, 实际{data.shape}")

        # 检查NaN值
        if torch.isnan(data).any():
            print(f"包含NaN值: {pt_file}")


check_integrity(2018)