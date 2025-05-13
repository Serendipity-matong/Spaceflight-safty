import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

# --- Configuration ---
BASE_DATA_PATH = "/Users/fangzijie/Documents/pressure_level"  # Path to the 'pressure_level' directory
YEAR = "2018"
# List of variable directories to process
VARIABLES = [
    "specific_humidity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "geopotential",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind"
]

NUM_FORECAST_STEPS = 12  # Must match NUM_FORECAST_STEPS in transformer.py
PATCH_SIZE = 4  # Must match PATCH_SIZE in transformer.py
OUT_CHANS_COUNT = 30  # Number of channels to predict, must match OUT_CHANS in transformer.py
INTRADAY_TIMESTEPS_EXPECTED = 4  # Number of intraday timesteps expected in the data


# --- Helper Functions ---

def get_sorted_nc_files(base_path, variable, year):
    """Gets a sorted list of .nc files for a given variable and year."""
    pattern = os.path.join(base_path, variable, year, f"{year}*.nc")
    files = sorted(glob.glob(pattern))
    return files


def load_and_process_nc_file(file_path):
    """
    Loads a single .nc file, extracts data, handles intraday timesteps, and reshapes.
    Returns a numpy array of shape (intraday_steps, levels, height, width) or None if error.
    Also returns original height and width.
    """
    try:
        with xr.open_dataset(file_path) as ds:
            data_vars = [var for var in ds.data_vars if
                         len(ds[var].shape) >= 2]
            if not data_vars:
                print(f"Warning: No suitable data variable found in {file_path}")
                return None, None, None

            data_var_name = data_vars[0]
            data_array_initial = ds[data_var_name]

            print(
                f"  Initial Dims for {file_path}, var {data_var_name}: {data_array_initial.dims}, Shape: {data_array_initial.shape}")

            possible_level_dims = ['level', 'plev', 'pressure', 'lev', 'pressure_level'] # <--- 修改处：添加了 'pressure_level'
            possible_lat_dims = ['latitude', 'lat', 'y']
            possible_lon_dims = ['longitude', 'lon', 'x']

            level_dim_name = next((d for d in possible_level_dims if d in data_array_initial.dims), None)
            lat_dim_name = next((d for d in possible_lat_dims if d in data_array_initial.dims), None)
            lon_dim_name = next((d for d in possible_lon_dims if d in data_array_initial.dims), None)

            if not lat_dim_name or not lon_dim_name:
                print(
                    f"Warning: Could not determine lat/lon dimensions in {file_path}. Found dims: {data_array_initial.dims}")
                return None, None, None

            original_height = data_array_initial.sizes[lat_dim_name]
            original_width = data_array_initial.sizes[lon_dim_name]

            selector = {}
            intraday_step_dim_name = None
            # Core dimensions that should remain after initial processing by isel (if needed)
            # Intraday will be handled by transpose later.
            core_dims_for_selection = {lat_dim_name, lon_dim_name}
            if level_dim_name:
                core_dims_for_selection.add(level_dim_name)

            data_array_processed = data_array_initial

            # Identify intraday dimension and other dimensions to select/reduce
            temp_processed_dims = set(core_dims_for_selection)  # Dims we definitely want to keep or handle specifically

            for dim_name_iter in data_array_initial.dims:
                if dim_name_iter in temp_processed_dims:  # Already accounted for (lat, lon, level)
                    continue

                dim_size = data_array_initial.sizes[dim_name_iter]

                if dim_size == INTRADAY_TIMESTEPS_EXPECTED and intraday_step_dim_name is None:
                    intraday_step_dim_name = dim_name_iter
                    temp_processed_dims.add(dim_name_iter)
                    print(f"    Identified '{dim_name_iter}' as intraday step dimension (size {dim_size})")
                # Handle other dimensions (e.g., a 'time' dimension of size 1 for daily files)
                # by selecting the first element.
                elif dim_name_iter not in core_dims_for_selection and intraday_step_dim_name != dim_name_iter:
                    selector[dim_name_iter] = 0
                    temp_processed_dims.add(dim_name_iter)  # Mark as processed by selector
                    print(f"    Selecting index 0 for dimension '{dim_name_iter}' (size {dim_size})")

            if intraday_step_dim_name is None:
                # Attempt to find a dimension that could be intraday if not explicitly matched by size
                # This is a fallback, might need adjustment based on actual data
                potential_intraday_dims = [d for d in data_array_initial.dims if
                                           d not in core_dims_for_selection and d not in selector]
                if len(potential_intraday_dims) == 1:
                    intraday_step_dim_name = potential_intraday_dims[0]
                    print(
                        f"    Warning: Assuming '{intraday_step_dim_name}' is intraday step dimension (size {data_array_initial.sizes[intraday_step_dim_name]}). Expected size {INTRADAY_TIMESTEPS_EXPECTED}.")
                    if data_array_initial.sizes[intraday_step_dim_name] != INTRADAY_TIMESTEPS_EXPECTED:
                        print(
                            f"    Critical Warning: Assumed intraday dim '{intraday_step_dim_name}' size {data_array_initial.sizes[intraday_step_dim_name]} != expected {INTRADAY_TIMESTEPS_EXPECTED}. This may lead to errors.")
                else:
                    print(
                        f"Warning: Could not definitively identify intraday step dimension in {file_path}. Dims: {data_array_initial.dims}. Selector: {selector}")
                    return None, None, None

            if selector:
                data_array_selected = data_array_initial.isel(**selector)
            else:
                data_array_selected = data_array_initial

            print(
                f"    Dims after isel for {file_path}: {data_array_selected.dims}, Shape: {data_array_selected.shape}")

            # Define the final order of dimensions: (intraday, level, lat, lon)
            # or (intraday, lat, lon) if no level.
            final_dims_order = [intraday_step_dim_name]
            if level_dim_name and level_dim_name in data_array_selected.dims:
                final_dims_order.append(level_dim_name)
            final_dims_order.extend([lat_dim_name, lon_dim_name])

            missing_dims = [d_name for d_name in final_dims_order if d_name not in data_array_selected.dims]
            if missing_dims:
                print(
                    f"Warning: Not all expected dimensions for transpose found in {file_path}. Missing: {missing_dims}. Available dims: {data_array_selected.dims}")
                return None, None, None

            data_array_ordered = data_array_selected.transpose(*final_dims_order)
            data_np = data_array_ordered.values

            # If data was originally (intraday, lat, lon) due to no level_dim,
            # expand to (intraday, 1, lat, lon) to represent single level.
            if data_np.ndim == 3 and (level_dim_name is None or level_dim_name not in data_array_selected.dims):
                data_np = np.expand_dims(data_np, axis=1)

                # Final check: data_np should be 4D (intraday_steps, levels, height, width)
            if data_np.ndim == 4:
                if data_np.shape[0] != data_array_initial.sizes[
                    intraday_step_dim_name]:  # Check against original intraday dim size
                    print(
                        f"Warning: Intraday timesteps dimension mismatch in {file_path}. Expected {data_array_initial.sizes[intraday_step_dim_name]}, got {data_np.shape[0]}")
                    return None, None, None
                if data_np.shape[2] != original_height or data_np.shape[3] != original_width:
                    print(
                        f"Warning: Mismatch in H/W for {file_path}. Original: ({original_height},{original_width}), Processed np shape: {data_np.shape}")
                    return None, None, None
                return data_np.astype(np.float32), original_height, original_width
            else:
                print(
                    f"Warning: Unexpected final data shape {data_np.shape} (ndim={data_np.ndim}) from {file_path} after processing. Expected 4D.")
                return None, None, None

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        traceback.print_exc()
        return None, None, None


# --- Main Script ---
if __name__ == "__main__":
    print("Starting data processing...")

    first_var_files = get_sorted_nc_files(BASE_DATA_PATH, VARIABLES[0], YEAR)
    if not first_var_files:
        print(f"Error: No files found for the first variable {VARIABLES[0]}. Exiting.")
        exit()

    dates = sorted(list(set(os.path.basename(f).split('.')[0] for f in first_var_files)))
    print(f"Found {len(dates)} unique dates in {YEAR}.")

    all_daily_data_stacked = []
    processed_dates = []
    original_img_h, original_img_w = None, None

    for date_str in dates:
        print(f"Processing date: {date_str}...")
        current_day_all_var_data_reshaped = []
        valid_date = True
        for var_idx, variable_name in enumerate(VARIABLES):
            file_name = f"{date_str}.nc"
            file_path = os.path.join(BASE_DATA_PATH, variable_name, YEAR, file_name)

            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path} for date {date_str}. Skipping this date.")
                valid_date = False
                break

            var_data_4d, h, w = load_and_process_nc_file(file_path)
            if var_data_4d is None:
                print(f"Warning: Failed to load data for {variable_name} on {date_str}. Skipping this date.")
                valid_date = False
                break

            num_intraday_steps = var_data_4d.shape[0]
            num_levels_var = var_data_4d.shape[1]
            current_h, current_w = var_data_4d.shape[2], var_data_4d.shape[3]

            if original_img_h is None:
                original_img_h, original_img_w = current_h, current_w
            elif current_h != original_img_h or current_w != original_img_w:
                print(
                    f"Error: Inconsistent image dimensions for {variable_name} on {date_str} before reshape. Expected ({original_img_h},{original_img_w}), got ({current_h},{current_w}). Exiting.")
                exit()

            var_data_reshaped = var_data_4d.reshape(num_intraday_steps * num_levels_var, original_img_h, original_img_w)
            current_day_all_var_data_reshaped.append(var_data_reshaped)

        if valid_date and current_day_all_var_data_reshaped:
            try:
                daily_stacked_data = np.concatenate(current_day_all_var_data_reshaped, axis=0)
                all_daily_data_stacked.append(daily_stacked_data)
                processed_dates.append(date_str)
            except ValueError as e:
                print(f"Error concatenating data for date {date_str}: {e}. Skipping this date.")
                print(
                    "This might be due to inconsistent spatial dimensions (H,W) across variables for this day after reshaping.")
                for i, arr_s in enumerate(current_day_all_var_data_reshaped):  # Renamed arr to arr_s to avoid conflict
                    print(f"  Shape of reshaped var {VARIABLES[i]} data: {arr_s.shape}")  # Renamed arr to arr_s
        else:
            if not valid_date:  # Added check for valid_date before printing skip message
                print(f"Skipping date {date_str} due to previous errors.")

    if not all_daily_data_stacked:
        print("No data successfully processed. Exiting.")
        exit()

    all_data_chronological = np.stack(all_daily_data_stacked, axis=0)
    print(f"Raw stacked data shape: {all_data_chronological.shape}")

    IN_CHANS_discovered = all_data_chronological.shape[1]
    print(f"Discovered IN_CHANS: {IN_CHANS_discovered}")
    print(f"Original IMG_SIZE (H, W): ({original_img_h}, {original_img_w})")

    target_h_padded = ((original_img_h + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    target_w_padded = ((original_img_w + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    print(f"Padded IMG_SIZE (H, W) for PATCH_SIZE={PATCH_SIZE}: ({target_h_padded}, {target_w_padded})")

    pad_h_total = target_h_padded - original_img_h
    pad_w_total = target_w_padded - original_img_w

    padding_config = (
        (0, 0),
        (0, 0),
        (0, pad_h_total),
        (0, pad_w_total)
    )
    all_data_padded = np.pad(all_data_chronological, padding_config, mode='constant', constant_values=0)
    print(f"Padded data shape: {all_data_padded.shape}")

    initial_states_list = []
    target_futures_list = []

    num_samples_possible = len(all_data_padded) - NUM_FORECAST_STEPS
    if num_samples_possible < 1:
        print(
            f"Error: Not enough data ({len(all_data_padded)} days) to create even one sample with NUM_FORECAST_STEPS={NUM_FORECAST_STEPS}. Exiting.")
        exit()

    for i in range(num_samples_possible):
        X_sample = all_data_padded[i]

        Y_sample_full_channels = all_data_padded[i + 1: i + 1 + NUM_FORECAST_STEPS]

        if OUT_CHANS_COUNT > IN_CHANS_discovered:
            print(
                f"Error: OUT_CHANS_COUNT ({OUT_CHANS_COUNT}) is greater than discovered IN_CHANS ({IN_CHANS_discovered}). Adjust configuration. Exiting.")
            exit()
        Y_sample = Y_sample_full_channels[:, :OUT_CHANS_COUNT, :, :]

        initial_states_list.append(X_sample)
        target_futures_list.append(Y_sample)

    all_initial_states_raw = np.array(initial_states_list, dtype=np.float32)
    all_target_futures_raw = np.array(target_futures_list, dtype=np.float32)

    print(f"Shape of all_initial_states_raw: {all_initial_states_raw.shape}")
    print(f"Shape of all_target_futures_raw: {all_target_futures_raw.shape}")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    initial_states_path = os.path.join(output_dir, f"initial_states_{YEAR}.npy")  # Added year to filename
    target_futures_path = os.path.join(output_dir, f"target_futures_{YEAR}.npy")  # Added year to filename

    np.save(initial_states_path, all_initial_states_raw)
    print(f"Saved initial states to {initial_states_path}")
    np.save(target_futures_path, all_target_futures_raw)
    print(f"Saved target futures to {target_futures_path}")

    print("Data processing finished.")