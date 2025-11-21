# ============================================
# PyTorch full pipeline (holiday as feature, no masking)
# Outputs: Total_Energy_kWh (Energy), Mainland_Amount (Amount), predicted_price
# ============================================
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

# ========== User params ==========
FILE_PATH = r"ML_Feature_Engineered_LoadProfile_Mainland.xlsx"
SAVE_PATH = r"Forecast_Safe_PyTorch.xlsx"

SEQ = 48                # 48 intervals = 12 hours
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

# Safety margins (same as original)
ENERGY_SF = 0.10   # +10%
COST_SF = 0.05     # +5%

# ============================================
# 1. LOAD RAW DATA
# ============================================
df = pd.read_excel(FILE_PATH)
print("Loaded:", df.shape)

# Ensure UTC column is datetime (format example: "2024/1/1 0:15:00")
df['UTC'] = pd.to_datetime(df['UTC'], format="%Y/%m/%d %H:%M:%S")

# Sort by time just in case
df = df.sort_values('UTC').reset_index(drop=True)

# ============================================
# 2. ENGINEER TIME FEATURES
# ============================================
def month_to_season(m):
    if m in [12, 1, 2]:
        return 0  # winter
    if m in [3,4,5]:
        return 1  # spring
    if m in [6,7,8]:
        return 2  # summer
    return 3      # autumn

df['month'] = df['UTC'].dt.month
df['season'] = df['month'].apply(month_to_season)
season_dummies = pd.get_dummies(df['season'], prefix='season')
df = pd.concat([df, season_dummies], axis=1)

df['weekday'] = df['UTC'].dt.weekday
weekday_dummies = pd.get_dummies(df['weekday'], prefix='wd')
df = pd.concat([df, weekday_dummies], axis=1)

df['hour'] = df['UTC'].dt.hour
df['minute'] = df['UTC'].dt.minute
df['time_of_day_frac'] = (df['hour'] * 60 + df['minute']) / (24*60)
df['hour_sin'] = np.sin(2 * np.pi * df['time_of_day_frac'])
df['hour_cos'] = np.cos(2 * np.pi * df['time_of_day_frac'])

# ============================================
# 3. BUILD WEEKLY TARIFF PATTERN (expand to 15-min)
# ============================================
weekly_hourly_codes = {
    0: "6;6;6;6;6;6;6;6;2;1;1;1;1;1;2;2;2;2;1;1;1;1;2;2",  # Mon
    1: "6;6;6;6;6;6;6;6;2;1;1;1;1;1;2;2;2;2;1;1;1;1;2;2",  # Tue
    2: "6;6;6;6;6;6;6;6;2;1;1;1;1;1;2;2;2;2;1;1;1;1;2;2",  # Wed
    3: "6;6;6;6;6;6;6;6;2;1;1;1;1;1;2;2;2;2;1;1;1;1;2;2",  # Thu
    4: "6;6;6;6;6;6;6;6;2;1;1;1;1;1;2;2;2;2;1;1;1;1;2;2",  # Fri
    5: "6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6",  # Sat
    6: "6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6;6",  # Sun
}

def build_tariff_series(row):
    wd = int(row['weekday'])
    s = weekly_hourly_codes[wd]
    nums = [int(x) for x in s.split(';') if x.strip()!='']
    if len(nums) != 24:
        raise ValueError(f"weekly pattern for weekday {wd} does not have 24 entries")
    expanded = []
    for h in nums:
        expanded += [h]*4
    idx = int(row['hour']*4 + row['minute']//15)
    return expanded[idx]

df['tariff_code'] = df.apply(build_tariff_series, axis=1)
tariff_dummies = pd.get_dummies(df['tariff_code'], prefix='tariff')
df = pd.concat([df, tariff_dummies], axis=1)

# ============================================
# 4. HOLIDAY / ABNORMAL LOW-ENERGY DETECTION (but keep data — will use flag as feature)
#    daily total vs 7-day median method
# ============================================
ENERGY_COL = 'Total_Energy_kWh'   # user-specified energy target
AMOUNT_COL = 'Mainland_Amount'    # user-specified amount target

df['date'] = df['UTC'].dt.date
daily = df.groupby('date')[ENERGY_COL].sum().rename('daily_total').reset_index()
daily['daily_median_7'] = daily['daily_total'].rolling(7, center=True, min_periods=1).median()

HOLIDAY_THRESHOLD = 0.60
daily['is_low_day'] = daily['daily_total'] < (daily['daily_median_7'] * HOLIDAY_THRESHOLD)
low_day_set = set(daily.loc[daily['is_low_day'], 'date'].tolist())
df['holiday_flag'] = df['date'].apply(lambda d: 1 if d in low_day_set else 0)

print("Detected low-consumption (holiday-like) dates:", sorted(list(low_day_set))[:20])

# ============================================
# 5. FEATURE SELECTION (精简特征)
#    保留核心通道、少数 ratio、rolling、lags、calendar、tariff、holiday_flag
# ============================================
base_features = [
    'L1 Site','LPDC','Extraction','Supply Air','Chillers',
    'Trafo 1','Trafo 2','Compressor 1','Compressor 2',
    'L1 Site_ratio','Chillers_ratio','Compressor 1_ratio','Compressor 2_ratio',
    'lag_1','lag_96',
    'rolling_3h','rolling_24h',
    'Peak_Load_Flag','High_Tariff_Flag'
]

# keep only those that exist in df
base_features = [c for c in base_features if c in df.columns]

season_cols = [c for c in df.columns if c.startswith('season_')]
wd_cols = [c for c in df.columns if c.startswith('wd_')]
tariff_cols = [c for c in df.columns if c.startswith('tariff_')]

engineered = season_cols + wd_cols + tariff_cols + ['hour_sin','hour_cos','holiday_flag']

features = base_features + engineered
features = [f for i,f in enumerate(features) if f not in features[:i]]  # dedupe

print(f"Using {len(features)} features. Examples: {features[:15]}")

# targets
targets = [ENERGY_COL, AMOUNT_COL]

# drop rows with NaNs in necessary cols (e.g., due to lags/rolling)
df = df.dropna(subset=features + targets).reset_index(drop=True)
print("After dropna:", df.shape)

# ============================================
# 6. SCALE
# ============================================
X = df[features].values
y = df[targets].values

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# ============================================
# 7. SEQUENCE GENERATION (no mask — keep all sequences)
#    Each sequence length = SEQ, target = y[i+SEQ]
# ============================================
def make_seq(X_scaled, y_scaled, seq_len):
    Xs, ys = [], []
    for i in range(len(X_scaled) - seq_len):
        Xs.append(X_scaled[i:i+seq_len])
        ys.append(y_scaled[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_seq(X_scaled, y_scaled, SEQ)
print("X_seq:", X_seq.shape, "y_seq:", y_seq.shape)

# ============================================
# 8. TRAIN / VAL / TEST SPLIT
# ============================================
N = len(X_seq)
train_end = int(0.8 * N)
val_end = int(0.9 * N)

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# ============================================
# 9. PyTorch Dataset (no mask)
# ============================================
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ============================================
# 10. BUILD Conv1D + BiLSTM MODEL (PyTorch)
# ============================================
n_features = X_train.shape[2]
n_targets = y_train.shape[1]

class ConvBiLSTM(nn.Module):
    def __init__(self, n_features, n_targets):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(64, n_targets)
    def forward(self, x):
        # x: (B, SEQ, F)
        x = x.permute(0,2,1)            # (B, F, SEQ)
        x = self.relu(self.conv(x))     # (B, 64, SEQ)
        x = self.pool(x)                # (B, 64, SEQ/2)
        x = x.permute(0,2,1)            # (B, SEQ/2, 64)
        _, (h_n, _) = self.bilstm(x)    # h_n: (num_layers*2, B, hidden)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)   # (B, 256)
        x = self.dropout(self.relu(self.fc1(h)))
        return self.out(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConvBiLSTM(n_features=n_features, n_targets=n_targets).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================================
# 11. TRAINING LOOP
# ============================================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    n_batches = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    avg_train = train_loss / max(1, n_batches)

    # validation
    model.eval()
    val_loss = 0.0
    n_val_batches = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            val_loss += criterion(pred, yb).item()
            n_val_batches += 1
    avg_val = val_loss / max(1, n_val_batches)

    print(f"Epoch {epoch+1}/{EPOCHS} — train_loss: {avg_train:.6f} — val_loss: {avg_val:.6f}")

# ============================================
# 12. PREDICT ON TEST SET
# ============================================
model.eval()
preds_scaled = []
trues_scaled = []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        out = model(Xb).cpu().numpy()
        preds_scaled.append(out)
        trues_scaled.append(yb.numpy())

y_pred_scaled = np.vstack(preds_scaled)
y_true_scaled = np.vstack(trues_scaled)

# inverse scale
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_true_scaled)

# ============================================
# 13. APPLY FIXED SAFETY MARGINS
# ============================================
y_pred_safe = y_pred.copy()
y_pred_safe[:, 0] = y_pred_safe[:, 0] * (1 + ENERGY_SF)   # Energy
y_pred_safe[:, 1] = y_pred_safe[:, 1] * (1 + COST_SF)     # Amount

# ============================================
# 14. EVALUATION
# ============================================
def metrics(y_t, y_p):
    mse = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_t, y_p)
    return mse, rmse, mae

labels = ['Energy','Mainland_Amount']
print("\n===== FINAL RESULTS (AFTER SAFETY MARGIN) =====")
for i, name in enumerate(labels):
    mse, rmse, mae = metrics(y_true[:, i], y_pred_safe[:, i])
    print(f"\n{name}:")
    print("  MSE :", mse)
    print("  RMSE:", rmse)
    print("  MAE :", mae)

# ============================================
# 15. SAVE FORECAST FILE (Energy_safe, Mainland_amount_safe)
# ============================================
df_out = pd.DataFrame(y_pred_safe, columns=['Energy_safe','Mainland_amount_safe'])

# ============================================
# 16. COMPUTE PRICE FROM FORMULA (using TPAT -> 15-min)
#    Amount = Power * TPAT * Energy * Price  => Price = Amount / (Power * TPAT * Energy)
# ============================================
TPAT_year = {
    1: 15.403,
    2: 9.885,
    3: 6.439,
    4: 5.495,
    5: 1.062,
    6: 0.616
}
TPAT_15min = {k: v / 35040.0 for k, v in TPAT_year.items()}

# align test indices to original df rows:
# The sequence i has target at original df row (i + SEQ).
# The test sequences start at index val_end in sequence array, so corresponding df row index is val_end + SEQ
test_idx_start = val_end + SEQ
test_idx_end = test_idx_start + len(y_pred_safe)

test_tariff_code = df['tariff_code'].iloc[test_idx_start:test_idx_end].values
# use Calculated_Power_kW as Power; if absent, try POWER or Calculated_Power_kW
if 'Calculated_Power_kW' in df.columns:
    test_power = df['Calculated_Power_kW'].iloc[test_idx_start:test_idx_end].values
elif 'POWER' in df.columns:
    test_power = df['POWER'].iloc[test_idx_start:test_idx_end].values
else:
    # fallback to ones to avoid zero division (but better to have real power)
    test_power = np.ones(len(y_pred_safe))

prices = []
for i in range(len(y_pred_safe)):
    amount = y_pred_safe[i, 1]
    energy = y_pred_safe[i, 0]
    power = test_power[i]
    try:
        tcode = int(test_tariff_code[i])
        tpat = TPAT_15min.get(tcode, np.nan)
    except Exception:
        tpat = np.nan

    denom = power * tpat * energy
    if denom is None or tpat is None or tpat == 0 or denom <= 0 or np.isnan(denom):
        prices.append(np.nan)
    else:
        prices.append(amount / denom)

df_out['predicted_price'] = prices

df_out.to_excel(SAVE_PATH, index=False)
print("\nSaved:", SAVE_PATH)

