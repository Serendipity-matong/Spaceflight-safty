import torch
import os


def load_and_inspect_pt_files(data_directory, num_files=60):
    """
    加载并查看指定目录下的 .pt 文件。

    参数:
    data_directory (str): 存放 .pt 文件的目录路径。
    num_files (int): 要加载的文件数量 (例如，60 表示 00.pt 到 59.pt)。
    """
    print(f"开始加载和查看来自 '{data_directory}' 的 .pt 文件...")

    for i in range(num_files):
        # 构建文件名，例如 "00.pt", "01.pt", ..., "59.pt"
        file_name = f"{i:03d}.pt"
        file_path = os.path.join(data_directory, file_name)

        if os.path.exists(file_path):
            try:
                # 加载 .pt 文件中的张量
                # 使用 map_location=torch.device('cpu') 确保在任何设备上都能加载
                tensor_data = torch.load(file_path, map_location=torch.device('cpu'))

                print(f"\n--- 文件: {file_name} ---")
                print(f"  路径: {file_path}")

                if isinstance(tensor_data, torch.Tensor):
                    print(f"  张量形状: {tensor_data.shape}")
                    print(f"  数据类型: {tensor_data.dtype}")
                    print(f"  设备: {tensor_data.device}")

                    # 您可以取消注释以下行来查看张量的前几个元素
                    # print(f"  前几个元素 (如果适用):")
                    # if tensor_data.ndim > 0 and tensor_data.numel() > 0:
                    #     # 根据维度和大小决定如何展示，这里简单处理
                    #     if tensor_data.numel() > 5:
                    #         print(tensor_data.flatten()[:5])
                    #     else:
                    #         print(tensor_data.flatten())
                    # else:
                    #     print("张量为空或无维度。")

                else:
                    print(f"  加载的内容不是一个张量，而是类型: {type(tensor_data)}")
                    # 如果 .pt 文件保存的是字典或其他对象，您可能需要不同的处理方式
                    # print(f"  内容: {tensor_data}")

            except Exception as e:
                print(f"\n--- 文件: {file_name} ---")
                print(f"  加载文件 {file_path} 时出错: {e}")
        else:
            print(f"\n--- 文件: {file_name} ---")
            print(f"  文件 {file_path} 未找到。")


if __name__ == "__main__":
    # 假设您的 .pt 文件在 'data/input_pt_files/' 目录下
    # 请根据您的实际文件路径修改这个变量
    pt_files_directory = "/Users/fangzijie/Downloads/input"

    # 确保目标目录存在，如果不存在则创建一个示例目录（在实际使用中您应该有这个目录）
    if not os.path.exists(pt_files_directory):
        # os.makedirs(pt_files_directory, exist_ok=True)
        print(f"警告：目录 '{pt_files_directory}' 不存在。请确保您的 .pt 文件位于正确的路径。")
        # 为了演示，您可以手动创建该目录并放入一些示例 .pt 文件
        # 例如:
        # if not os.path.exists(os.path.join(pt_files_directory, "00.pt")):
        #     torch.save(torch.randn(2, 3, 4, 5, 6), os.path.join(pt_files_directory, "00.pt")) # 示例5D张量
        # if not os.path.exists(os.path.join(pt_files_directory, "01.pt")):
        #     torch.save(torch.randn(1, 3, 10, 20, 20), os.path.join(pt_files_directory, "01.pt")) # 另一个示例

    load_and_inspect_pt_files(pt_files_directory)