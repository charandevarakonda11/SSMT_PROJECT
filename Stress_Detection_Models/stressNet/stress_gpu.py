import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import json
from imblearn.over_sampling import SMOTE

# Load data
def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    X = np.array([[frame['f0'], frame['energy']] for frame in data])
    y = np.array([frame['label'] for frame in data])
    return X, y

# Dataset class
class StressDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TDNN block
class TDNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, context_size, dilation=1):
        super(TDNNLayer, self).__init__()
        self.tdnn = nn.Conv1d(input_dim, output_dim, kernel_size=context_size, dilation=dilation)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        return self.bn(self.activation(self.tdnn(x)))

# Model
class StressNet(nn.Module):
    def __init__(self, input_dim):
        super(StressNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.tdnn1 = TDNNLayer(64, 128, context_size=5)
        self.tdnn2 = TDNNLayer(128, 128, context_size=3, dilation=2)

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        return self.classifier(x)

# Training function

def train_model(model, train_loader, val_loader, epochs=150, lr=1e-4, device='cuda'):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

# def train_model(model, train_loader, val_loader, epochs=150, lr=1e-4, device='cuda'):
    
#     model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")

        if (epoch + 1) % 10 == 0:
            evaluate_model(model, val_loader, device)

# Evaluation function
def evaluate_model(model, loader, device='cuda'):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy().flatten()
            preds = 1 / (1 + np.exp(-logits))  # Sigmoid manually
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    print("Accuracy:", accuracy_score(all_labels, bin_preds))
    print(classification_report(all_labels, bin_preds))

# Main logic
# def main():
#     feature_json = "/scratch/data_temp/ssmt_pr/Json_files/f0_energy_mfcc_sdc.json"
#     print("Loading data...")
#     X, y = load_data(feature_json)
#     X = np.nan_to_num(X)
#     y = np.nan_to_num(y)
#     print(f"Initial shape of X: {X.shape}, y: {y.shape}")

#     time_steps = 15
#     num_samples = X.shape[0] // time_steps
#     X = X[:num_samples * time_steps]
#     y = y[:num_samples * time_steps]
#     X = X.reshape(num_samples, time_steps, X.shape[1])
#     y = y.reshape(num_samples, time_steps)
#     y = y[:, time_steps // 2].astype(int)
#     print(f"Unique labels after reshaping: {np.unique(y)}")
#     print(f"Final shape of X: {X.shape}, y: {y.shape}")

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#     train_loader = DataLoader(StressDataset(X_train, y_train), batch_size=256, shuffle=True)
#     val_loader = DataLoader(StressDataset(X_val, y_val), batch_size=256)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = StressNet(input_dim=X.shape[2])
#     train_model(model, train_loader, val_loader, device=device)

def main():
    feature_json = "/scratch/data_temp/ssmt_pr/Json_files/f0_energy_mfcc_sdc.json"
    print("Loading data...")
    X, y = load_data(feature_json)
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    print(f"Initial shape of X: {X.shape}, y: {y.shape}")

    time_steps = 15
    num_samples = X.shape[0] // time_steps
    X = X[:num_samples * time_steps]
    y = y[:num_samples * time_steps]
    X = X.reshape(num_samples, time_steps, X.shape[1])
    y = y.reshape(num_samples, time_steps)
    y = y[:, time_steps // 2].astype(int)

    print(f"Before SMOTE - Class distribution: {np.bincount(y)}")

    # Apply SMOTE
    X_2d = X.reshape((X.shape[0], -1))  # Flatten time dimension for SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_2d, y)

    # Reshape back to (samples, time, features)
    X_resampled = X_resampled.reshape((-1, time_steps, X.shape[2]))
    print(f"After SMOTE - Class distribution: {np.bincount(y_resampled)}")

    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    train_loader = DataLoader(StressDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(StressDataset(X_val, y_val), batch_size=256)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StressNet(input_dim=X.shape[2])
    train_model(model, train_loader, val_loader, device=device)


if __name__ == "__main__":
    main()
