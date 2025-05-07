# from main import N_TARGET_CHANNELS
import numpy as np
import os

N_VARS = 9
N_LEVELS = 13
N_CHANNELS = N_VARS * N_LEVELS
N_TARGET_CHANNELS = 30

data_dir = "/Users/fangzijie/Documents/processed_data"
file_dir = "batch_1.npy"

file_path = os.path.join(data_dir, file_dir)
data = np.load(file_path)
print(f"\n数据的形状:{data.shape}")

num_feature_columns = 1+2+N_CHANNELS*3

X = data[:,:num_feature_columns]
Y = data[:,num_feature_columns:]
print(f"\nX的形状:{X.shape}")
print(f"Y的形状:{Y.shape}")

