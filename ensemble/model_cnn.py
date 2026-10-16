import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class ESM1DCNN(nn.Module):
    def __init__(self, input_dim=1280, dropout_rate=0.5):
        super(ESM1DCNN, self).__init__()

        # チャンネル数と全結合層をさらにコンパクト化して暗記を物理制御
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=12,
                      kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=24,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate)
        )

        self.fc_block = nn.Sequential(
            nn.Linear(24 * 320, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        out = self.fc_block(x)
        return out


def find_best_threshold(y_true, probs, metric="mcc"):
    best_thresh = 0.5
    best_score = -1.0
    for t in np.arange(0.15, 0.70, 0.01):
        preds = (probs >= t).astype(int)
        score = matthews_corrcoef(y_true, preds) if metric == "mcc" else f1_score(
            y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=5):
    """
    patience を 15 -> 5 に大幅短縮して厳格化
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        running_tr_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze(1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_tr_loss += loss.item() * batch_x.size(0)

        epoch_tr_loss = running_tr_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_tr_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze(1)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, train_loss_history, val_loss_history, best_val_loss


def predict_probs(model, data_loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch_x in data_loader:
            if isinstance(batch_x, (list, tuple)):
                batch_x = batch_x[0]
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze(1)
            sigmoid_probs = torch.sigmoid(outputs).cpu().numpy()
            probs.extend(sigmoid_probs)
    return np.array(probs)


def run_cnn(X_train, y_train, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relaxed_pos_weight = 3.0
    pos_weight_tensor = torch.tensor(
        [relaxed_pos_weight], dtype=torch.float32).to(device)

    scaler_full = StandardScaler()
    X_train_s_full = scaler_full.fit_transform(X_train)
    X_test_s_full = scaler_full.transform(X_test)

    test_ds = TensorDataset(torch.tensor(X_test_s_full, dtype=torch.float32))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_probs = np.zeros(len(y_train))
    test_probs_list = []
    train_aucs = []
    cv_aucs = []
    best_thresholds = []

    all_tr_losses = []
    all_val_losses = []
    min_val_losses = []

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[t_idx], X_train[v_idx]
        y_tr, y_val = y_train[t_idx], y_train[v_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        tr_ds = TensorDataset(torch.tensor(
            X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(
            X_val_s, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = ESM1DCNN(
            input_dim=X_train.shape[1], dropout_rate=0.5).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        # Weight Decay を 1e-1 に強化して正則化を徹底
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)

        # patience=5 で早期停止を厳格化
        model, tr_h, val_h, min_val_l = train_model(
            model, tr_loader, val_loader, criterion, optimizer, device, epochs=100, patience=5)

        all_tr_losses.append(tr_h)
        all_val_losses.append(val_h)
        min_val_losses.append(min_val_l)

        val_probs = predict_probs(model, val_loader, device)
        oof_probs[v_idx] = val_probs

        tr_loader_eval = DataLoader(tr_ds, batch_size=64, shuffle=False)
        tr_probs = predict_probs(model, tr_loader_eval, device)

        opt_thresh, _ = find_best_threshold(y_val, val_probs, metric="mcc")
        best_thresholds.append(opt_thresh)

        cv_aucs.append(roc_auc_score(y_val, val_probs))
        train_aucs.append(roc_auc_score(y_tr, tr_probs))

        fold_test_probs = predict_probs(model, test_loader, device)
        test_probs_list.append(fold_test_probs)

    max_ep = max(len(l) for l in all_tr_losses)
    padded_tr = np.array([l + [l[-1]] * (max_ep - len(l))
                         for l in all_tr_losses])
    padded_val = np.array([l + [l[-1]] * (max_ep - len(l))
                          for l in all_val_losses])

    return {
        "cv_probs": oof_probs,
        "test_probs": np.mean(test_probs_list, axis=0),
        "train_auc": np.mean(train_aucs),
        "cv_auc": np.mean(cv_aucs),
        "opt_thresh": np.mean(best_thresholds),
        "min_cv_loss": np.mean(min_val_losses),
        "mean_train_losses": np.mean(padded_tr, axis=0),
        "mean_val_losses": np.mean(padded_val, axis=0)
    }
