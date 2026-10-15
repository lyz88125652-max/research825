import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------
# 再現性のためのシード固定
# -------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------------------------------------------------
# 1D-CNN モデル定義 (改善版: 広域カーネル & チャンネル調整)
# -------------------------------------------------------------
class ESM1DCNN(nn.Module):

    def __init__(self, input_dim=1280, dropout_rate=0.4):
        super(ESM1DCNN, self).__init__()

        # Block 1: 広域コンテキスト (kernel_size=7)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # -> (batch, 16, 640)
            nn.Dropout(dropout_rate),
        )

        # Block 2: 局所コンテキスト (kernel_size=5)
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # -> (batch, 32, 320)
            nn.Dropout(dropout_rate),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(32 * 320, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        out = self.fc_block(x)
        return out


# -------------------------------------------------------------
# 最適閾値検索 (MCC ベースに強化)
# -------------------------------------------------------------
def find_best_threshold(y_true, probs, metric="mcc"):
    best_thresh = 0.5
    best_score = -1.0
    for t in np.arange(0.15, 0.70, 0.01):
        preds = (probs >= t).astype(int)
        if metric == "mcc":
            score = matthews_corrcoef(y_true, preds)
        elif metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        else:
            score = matthews_corrcoef(y_true, preds)

        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


# -------------------------------------------------------------
# 学習・評価関数 (Loss 履歴の取得対応)
# -------------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=100,
    patience=15,
):
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_weights = None

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze(1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze(1)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)

        # Early Stopping
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

    return model, train_loss_history, val_loss_history


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


# -------------------------------------------------------------
# Loss 曲線の描画・保存関数
# -------------------------------------------------------------
def plot_loss_curves(all_train_losses, all_val_losses, save_path):
    plt.figure(figsize=(10, 6))

    # 5-Fold の平均 Loss 曲線を計算
    max_epochs = max(len(l) for l in all_train_losses)

    # パディング処理（途中で早期終了した Fold を考慮）
    padded_train = np.array([
        l + [l[-1]] * (max_epochs - len(l)) for l in all_train_losses
    ])
    padded_val = np.array([
        l + [l[-1]] * (max_epochs - len(l)) for l in all_val_losses
    ])

    mean_train_loss = np.mean(padded_train, axis=0)
    mean_val_loss = np.mean(padded_val, axis=0)
    epochs = np.arange(1, max_epochs + 1)

    # 各 Fold の薄い線
    for i in range(len(all_train_losses)):
        plt.plot(
            np.arange(1, len(all_train_losses[i]) + 1),
            all_train_losses[i],
            color="blue",
            alpha=0.15,
            linestyle="--",
        )
        plt.plot(
            np.arange(1, len(all_val_losses[i]) + 1),
            all_val_losses[i],
            color="orange",
            alpha=0.15,
            linestyle="--",
        )

    # 平均曲線 (太線)
    plt.plot(
        epochs,
        mean_train_loss,
        label="Train Loss (Mean)",
        color="blue",
        linewidth=2.5,
    )
    plt.plot(
        epochs,
        mean_val_loss,
        label="Validation Loss (Mean)",
        color="orange",
        linewidth=2.5,
    )

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (BCEWithLogitsLoss)", fontsize=12)
    plt.title("1D-CNN Training and Validation Loss Curves (5-Fold CV)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Loss 曲線グラフを保存しました: '{save_path}'")


def main():
    print("=== 1D-CNN 評価スクリプト (Loss曲線出力 & Precision改善版) ===")

    train_path = input(
        "学習用Excelファイル名 (デフォルト: trainESMft.xlsx): "
    ).strip()
    if not train_path:
        train_path = "trainESMft.xlsx"

    test_path = input(
        "テスト用Excelファイル名 (デフォルト: testESMft.xlsx): "
    ).strip()
    if not test_path:
        test_path = "testESMft.xlsx"

    out_filename = input(
        "出力ファイル名 (デフォルト: cnn_esm_improved_result.xlsx): "
    ).strip()
    if not out_filename:
        out_filename = "cnn_esm_improved_result.xlsx"

    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    y_train = df_train.iloc[:, 1].values.astype(np.float32)
    X_train = df_train.iloc[:, 2:].values.astype(np.float32)

    y_test = df_test.iloc[:, 1].values.astype(np.float32)
    X_test = df_test.iloc[:, 2:].values.astype(np.float32)

    print(
        f"\nTrain データ数: {X_train.shape[0]} (正例:"
        f" {int(np.sum(y_train == 1))}/陰性: {int(np.sum(y_train == 0))})"
    )
    print(
        f"Test  データ数: {X_test.shape[0]} (正例:"
        f" {int(np.sum(y_test == 1))}/陰性: {int(np.sum(y_test == 0))})"
    )
    print(f"特徴量次元数 : {X_train.shape[1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス  : {device}")

    # -------------------------------------------------------------
    # 不均衡重み (pos_weight) の緩和: 6.15 ➔ 3.5 に緩和して Precision を保護
    # -------------------------------------------------------------
    relaxed_pos_weight = 3.5
    pos_weight_tensor = torch.tensor(
        [relaxed_pos_weight], dtype=torch.float32
    ).to(device)
    print(f"設定 pos_weight: {relaxed_pos_weight} (Precision 改善用緩和)")

    scaler_full = StandardScaler()
    X_train_s_full = scaler_full.fit_transform(X_train)
    X_test_s_full = scaler_full.transform(X_test)
    test_ds = TensorDataset(torch.tensor(X_test_s_full, dtype=torch.float32))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_aucs, cv_f1s, cv_sens, cv_precs, cv_mccs = [], [], [], [], []
    train_aucs = []
    best_thresholds = []
    test_probs_list = []

    all_fold_train_losses = []
    all_fold_val_losses = []

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[t_idx], X_train[v_idx]
        y_tr, y_val = y_train[t_idx], y_train[v_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        tr_ds = TensorDataset(
            torch.tensor(X_tr_s), torch.tensor(y_tr, dtype=torch.float32)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_s), torch.tensor(y_val, dtype=torch.float32)
        )

        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = ESM1DCNN(
            input_dim=X_train.shape[1], dropout_rate=0.4
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        model, tr_loss_hist, val_loss_hist = train_model(
            model,
            tr_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs=100,
            patience=15,
        )

        all_fold_train_losses.append(tr_loss_hist)
        all_fold_val_losses.append(val_loss_hist)

        val_probs = predict_probs(model, val_loader, device)

        tr_loader_eval = DataLoader(tr_ds, batch_size=64, shuffle=False)
        tr_probs = predict_probs(model, tr_loader_eval, device)

        # MCC を基準にした最適閾値検索
        opt_thresh, _ = find_best_threshold(y_val, val_probs, metric="mcc")
        best_thresholds.append(opt_thresh)

        val_preds = (val_probs >= opt_thresh).astype(int)

        cv_aucs.append(roc_auc_score(y_val, val_probs))
        train_aucs.append(roc_auc_score(y_tr, tr_probs))
        cv_f1s.append(f1_score(y_val, val_preds, zero_division=0))
        cv_sens.append(recall_score(y_val, val_preds, zero_division=0))
        cv_precs.append(precision_score(y_val, val_preds, zero_division=0))
        cv_mccs.append(matthews_corrcoef(y_val, val_preds))

        fold_test_probs = predict_probs(model, test_loader, device)
        test_probs_list.append(fold_test_probs)

    mean_opt_thresh = np.mean(best_thresholds)

    print("\n--- 5-Fold CV 評価結果 ---")
    print(f"Train AUC        : {np.mean(train_aucs):.4f}")
    print(f"ROC-AUC (CV)     : {np.mean(cv_aucs):.4f}")
    print(f"Sensitivity (CV) : {np.mean(cv_sens):.4f}")
    print(f"Precision (CV)   : {np.mean(cv_precs):.4f}")
    print(f"F1-Score (CV)    : {np.mean(cv_f1s):.4f}")
    print(f"MCC (CV)         : {np.mean(cv_mccs):.4f}")
    print(f"平均最適閾値     : {mean_opt_thresh:.4f}")

    # 5-Fold アンサンブル予測
    avg_test_probs = np.mean(test_probs_list, axis=0)
    test_preds = (avg_test_probs >= mean_opt_thresh).astype(int)

    test_auc = roc_auc_score(y_test, avg_test_probs)
    test_sens = recall_score(y_test, test_preds, zero_division=0)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_acc = accuracy_score(y_test, test_preds)
    test_mcc = matthews_corrcoef(y_test, test_preds)

    print("\n=== Test データ評価結果 (5-Fold Ensemble) ===")
    print(f"ROC-AUC (Test)   : {test_auc:.4f}")
    print(f"Sensitivity(Test): {test_sens:.4f}")
    print(f"Precision (Test) : {test_prec:.4f}")
    print(f"F1-Score (Test)  : {test_f1:.4f}")
    print(f"Accuracy (Test)  : {test_acc:.4f}")
    print(f"MCC (Test)       : {test_mcc:.4f}")

    # 出力・保存処理
    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Loss 曲線グラフの描画・保存
    loss_plot_path = os.path.join(output_dir, "cnn_loss_curve.png")
    plot_loss_curves(
        all_fold_train_losses, all_fold_val_losses, loss_plot_path
    )

    # 2. Excel スコアの保存
    res_df = pd.DataFrame([{
        "モデル": "1D-CNN (Improved Ensemble)",
        "特徴量": os.path.basename(train_path),
        "次元数": X_train.shape[1],
        "Train AUC": round(np.mean(train_aucs), 4),
        "ROC-AUC (CV)": round(np.mean(cv_aucs), 4),
        "Sensitivity (CV)": round(np.mean(cv_sens), 4),
        "Precision (CV)": round(np.mean(cv_precs), 4),
        "F1-Score (CV)": round(np.mean(cv_f1s), 4),
        "MCC (CV)": round(np.mean(cv_mccs), 4),
        "Opt Threshold": round(mean_opt_thresh, 4),
        "ROC-AUC (Test)": round(test_auc, 4),
        "Sensitivity (Test)": round(test_sens, 4),
        "Precision (Test)": round(test_prec, 4),
        "F1-Score (Test)": round(test_f1, 4),
        "Accuracy (Test)": round(test_acc, 4),
        "MCC (Test)": round(test_mcc, 4),
    }])

    if not out_filename.endswith(".xlsx"):
        out_filename += ".xlsx"
    output_path = os.path.join(output_dir, out_filename)
    res_df.to_excel(output_path, index=False)
    print(f"✅ 評価結果を出力しました: '{output_path}'")


if __name__ == "__main__":
    main()
