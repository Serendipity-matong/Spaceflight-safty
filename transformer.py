import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset
import time # 用于计时

class PatchEmbed(nn.Module):
    """将2D空间数据分割成Patch并进行线性嵌入"""
    def __init__(self, img_size=(46, 71), patch_size=4, in_chans=104, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch size.")
        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [Batch, Channels, Height, Width]
        x = self.proj(x)  # [Batch, EmbedDim, PatchesH, PatchesW]
        x = x.flatten(2)  # [Batch, EmbedDim, NumPatches]
        x = x.transpose(1, 2)  # [Batch, NumPatches, EmbedDim]
        return x

class TransformerForecaster(nn.Module):
    def __init__(self, img_size=(46, 71), patch_size=4, 
                 in_chans=104, out_chans=30, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.prediction_head = nn.Linear(embed_dim, out_chans * patch_size * patch_size) 
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_features(self, x_current_step):
        B = x_current_step.shape[0]
        x = self.patch_embed(x_current_step) 
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        return x

    def forward_prediction_head(self, x_features):
        B = x_features.shape[0]
        pred_flat_patches = self.prediction_head(x_features) 
        # pred_flat_patches shape: [B, NumPatches, C_out * PS * PS]
        
        # Reconstruct to spatial dimensions: [B, C_out, H, W]
        num_patches_h = self.img_size[0] // self.patch_size
        num_patches_w = self.img_size[1] // self.patch_size
        
        # Reshape to [B, num_patches_h, num_patches_w, self.out_chans, self.patch_size, self.patch_size]
        pred_structured = pred_flat_patches.reshape(
            B, 
            num_patches_h, 
            num_patches_w, 
            self.out_chans, 
            self.patch_size, 
            self.patch_size
        )
        
        # Permute to [B, self.out_chans, num_patches_h, self.patch_size, num_patches_w, self.patch_size]
        pred_permuted = pred_structured.permute(0, 3, 1, 4, 2, 5)
        
        # Reshape to [B, self.out_chans, H, W]
        pred_reconstructed = pred_permuted.reshape(
            B, 
            self.out_chans, 
            self.img_size[0], 
            self.img_size[1]
        )
        return pred_reconstructed

    def forward(self, x_initial_state, num_forecast_steps=12):
        predictions_over_time = []
        current_input_state = x_initial_state 

        for _ in range(num_forecast_steps):
            features = self.forward_features(current_input_state)
            predicted_next_30_channels = self.forward_prediction_head(features)
            predictions_over_time.append(predicted_next_30_channels)

            # Autoregressive update of the input state for the next step
            if self.in_chans == self.out_chans:
                current_input_state = predicted_next_30_channels
            else:
                # IMPORTANT ASSUMPTION:
                # The `out_chans` (30) predicted channels are used to update
                # the FIRST `out_chans` channels of the `in_chans` (104) input state.
                # The remaining `in_chans - out_chans` channels are carried over from the previous step.
                # If your channel mapping is different, you MUST modify this logic.
                temp_input = current_input_state.clone()
                temp_input[:, :self.out_chans, :, :] = predicted_next_30_channels
                # For the remaining channels (from out_chans to in_chans), they are kept from the previous state.
                # If these other channels also need to be predicted or updated based on physics/other models,
                # this part needs to be significantly more complex.
                current_input_state = temp_input
        
        output_tensor = torch.stack(predictions_over_time, dim=1)
        return output_tensor

# --- Data Normalization ---
class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        # data shape: (n_samples, channels, height, width) or (n_samples, steps, channels, height, width)
        self.mean = np.mean(data, axis=(0, 2, 3) if len(data.shape) == 4 else (0, 1, 3, 4), keepdims=True)
        self.std = np.std(data, axis=(0, 2, 3) if len(data.shape) == 4 else (0, 1, 3, 4), keepdims=True)
        self.std[self.std == 0] = 1.0 # Avoid division by zero

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

# --- Dummy Competition Metric ---
def calculate_competition_metric(predictions, targets):
    # Placeholder: Mean Squared Error
    # Replace with your actual competition metric (e.g., weighted RMSE)
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    return np.mean((predictions - targets)**2)

# --- Main Training and Evaluation Script ---
if __name__ == '__main__':
    # --- Configuration ---
    IMG_SIZE = (48, 72) # Example: divisible by patch_size
    PATCH_SIZE = 4
    IN_CHANS = 104
    OUT_CHANS = 30
    EMBED_DIM = 256 # Reduced for faster demo
    DEPTH = 4       # Reduced
    NUM_HEADS = 4   # Reduced
    MLP_RATIO = 2.0 # Reduced
    DROPOUT = 0.1

    NUM_EPOCHS = 10 # Reduced for demo
    BATCH_SIZE = 2  # Reduced for demo
    LEARNING_RATE = 1e-4
    NUM_FORECAST_STEPS = 12 # 3 days * (24/6) steps
    NUM_SAMPLES_DEMO = 50 # Number of samples for dummy data
    N_SPLITS_CV = 3     # Number of splits for TimeSeriesSplit, reduced for demo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Generate Dummy Data (Replace with your actual data loading) ---
    print("Generating dummy data...")
    # Initial states: [NumSamples, InChans, H, W]
    all_initial_states_raw = np.random.rand(NUM_SAMPLES_DEMO, IN_CHANS, IMG_SIZE[0], IMG_SIZE[1]).astype(np.float32)
    # Target future states: [NumSamples, NumForecastSteps, OutChans, H, W]
    all_target_futures_raw = np.random.rand(NUM_SAMPLES_DEMO, NUM_FORECAST_STEPS, OUT_CHANS, IMG_SIZE[0], IMG_SIZE[1]).astype(np.float32)
    print("Dummy data generated.")

    # --- TimeSeriesSplit Cross-Validation ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    fold_metrics = []

    for fold, (train_indices, val_indices) in enumerate(tscv.split(all_initial_states_raw)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS_CV} ---")
        
        X_train_raw, X_val_raw = all_initial_states_raw[train_indices], all_initial_states_raw[val_indices]
        Y_train_raw, Y_val_raw = all_target_futures_raw[train_indices], all_target_futures_raw[val_indices]

        # Normalize data (Fit on training data of the current fold)
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()

        scaler_X.fit(X_train_raw)
        X_train_norm = scaler_X.transform(X_train_raw)
        X_val_norm = scaler_X.transform(X_val_raw)
        
        # Note: Y (targets) should also be scaled if they are not already in a comparable range or if the loss function is sensitive to scale
        # For simplicity here, we assume Y is also scaled, or the model learns to output in Y's original scale.
        # In a real scenario, scale Y_train_raw and use that scaler for Y_val_raw and inverse transform of predictions.
        scaler_Y.fit(Y_train_raw) # Fit on Y_train_raw
        Y_train_norm = scaler_Y.transform(Y_train_raw)
        Y_val_norm = scaler_Y.transform(Y_val_raw)


        train_dataset = TensorDataset(torch.tensor(X_train_norm), torch.tensor(Y_train_norm))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
        
        val_dataset = TensorDataset(torch.tensor(X_val_norm), torch.tensor(Y_val_norm))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)

        # --- Model, Loss, Optimizer ---
        model = TransformerForecaster(
            img_size=IMG_SIZE, patch_size=PATCH_SIZE, 
            in_chans=IN_CHANS, out_chans=OUT_CHANS, 
            embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO, dropout=DROPOUT
        ).to(device)
        
        criterion = nn.MSELoss() # Or your custom weighted loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        best_val_metric = float('inf')
        best_model_path = f"transformer_model_fold_{fold+1}_best.pth"

        print(f"Starting training for fold {fold+1}...")
        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            model.train()
            epoch_train_loss = 0
            for batch_x_initial, batch_y_future in train_loader:
                batch_x_initial = batch_x_initial.to(device)
                batch_y_future = batch_y_future.to(device)
                
                optimizer.zero_grad()
                predicted_sequence = model(batch_x_initial, num_forecast_steps=NUM_FORECAST_STEPS)
                loss = criterion(predicted_sequence, batch_y_future)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)

            # Validation
            model.eval()
            epoch_val_loss = 0
            all_val_preds_epoch = []
            all_val_targets_epoch = []
            with torch.no_grad():
                for batch_x_val_initial, batch_y_val_future in val_loader:
                    batch_x_val_initial = batch_x_val_initial.to(device)
                    batch_y_val_future = batch_y_val_future.to(device) # Ground truth for validation
                    
                    val_pred_sequence_norm = model(batch_x_val_initial, num_forecast_steps=NUM_FORECAST_STEPS)
                    val_loss = criterion(val_pred_sequence_norm, batch_y_val_future) # Loss on normalized scale
                    epoch_val_loss += val_loss.item()
                    
                    # Inverse transform for metric calculation
                    val_pred_sequence_unnorm = scaler_Y.inverse_transform(val_pred_sequence_norm.cpu().numpy())
                    batch_y_val_future_unnorm = scaler_Y.inverse_transform(batch_y_val_future.cpu().numpy())

                    all_val_preds_epoch.append(val_pred_sequence_unnorm)
                    all_val_targets_epoch.append(batch_y_val_future_unnorm)
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            
            # Concatenate all predictions and targets for this epoch's validation
            final_val_preds_epoch = np.concatenate(all_val_preds_epoch, axis=0)
            final_val_targets_epoch = np.concatenate(all_val_targets_epoch, axis=0)
            
            current_val_metric = calculate_competition_metric(final_val_preds_epoch, final_val_targets_epoch)
            epoch_duration = time.time() - epoch_start_time

            print(f"Fold {fold+1} Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss (Norm): {avg_val_loss:.6f} | Val Metric (Unnorm): {current_val_metric:.6f} | Time: {epoch_duration:.2f}s")

            if current_val_metric < best_val_metric:
                best_val_metric = current_val_metric
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model to {best_model_path} with metric: {best_val_metric:.6f}")
        
        fold_metrics.append(best_val_metric)
        print(f"Best validation metric for Fold {fold+1}: {best_val_metric:.6f}")

    print("\n--- Cross-Validation Summary ---")
    for i, metric_val in enumerate(fold_metrics):
        print(f"Fold {i+1} Best Metric: {metric_val:.6f}")
    print(f"Average Best Metric across folds: {np.mean(fold_metrics):.6f} (+/- {np.std(fold_metrics):.6f})")

    # Example of loading the best model from a fold and making a prediction
    # model.load_state_dict(torch.load(f"transformer_model_fold_1_best.pth"))
    # model.eval()
    # with torch.no_grad():
    #     sample_input_raw = all_initial_states_raw[0:1] # Take one sample
    #     sample_input_norm = scaler_X.transform(sample_input_raw) # Use a fitted scaler
    #     sample_input_tensor = torch.tensor(sample_input_norm).to(device)
    #     prediction_norm = model(sample_input_tensor, num_forecast_steps=NUM_FORECAST_STEPS)
    #     prediction_unnorm = scaler_Y.inverse_transform(prediction_norm.cpu().numpy()) # Use a fitted scaler
    #     print(f"\nExample prediction shape (unnormalized): {prediction_unnorm.shape}") # (1, 12, 30, H, W)