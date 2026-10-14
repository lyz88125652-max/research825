import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
# 1D-CNN モデル定義
# -------------------------------------------------------------
class ESM1DCNN(nn.Module):

    def __init__(self, input_dim=1280, dropout_rate=0.4):
        super(ESM1DCNN, self).__init__()
        # 入力形状: (batch_size, 1, 1280)

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # -> (batch_size, 32, 640)
            nn.Dropout(dropout_rate),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # -> (batch_size, 64, 320)
            nn.Dropout(dropout_rate),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(64 * 320, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),  # ロジット出力
        )

    def forward(self, x):
        # x: (batch_size, 1280) -> (batch_size, 1, 1280)
        x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)  # 平坦化 (Flatten)
        out = self.fc_block(x)
        return out


# -------------------------------------------------------------
# 最適閾値検索 (F1-Score ベース)
# -------------------------------------------------------------
def find_best_threshold(y_true, probs, metric='f1'):
    best_thresh = 0.5
    best_score = -1.0
    for t in np.arange(0.15, 0.65, 0.01):
        preds = (probs >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, preds)
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


# -------------------------------------------------------------
# 1エポック分の学習・評価関数
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
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(epochs):
        # 学習
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze(1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze(1)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)

        # Early Stopping 判断
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model


def predict_probs(model, data_loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze(1)
            sigmoid_probs = torch.sigmoid(outputs).cpu().numpy()
            probs.extend(sigmoid_probs)
    return np.array(probs)


def main():
    print('=== 1D-CNN 単体評価スクリプト (ESM-2 特徴量対応) ===')

    train_path = input(
        '学習用Excelファイル名 (デフォルト: trainESMft.xlsx): '
    ).strip()
    if not train_path:
        train_path = 'trainESMft.xlsx'

    test_path = input(
        'テスト用Excelファイル名 (デフォルト: testESMft.xlsx): '
    ).strip()
    if not test_path:
        test_path = 'testESMft.xlsx'

    out_filename = input(
        '出力ファイル名 (デフォルト: cnn_esm_result.xlsx): '
    ).strip()
    if not out_filename:
        out_filename = 'cnn_esm_result.xlsx'

    # データ読み込み
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    y_train = df_train.iloc[:, 1].values.astype(np.float32)
    X_train = df_train.iloc[:, 2:].values.astype(np.float32)

    y_test = df_test.iloc[:, 1].values.astype(np.float32)
    X_test = df_test.iloc[:, 2:].values.astype(np.float32)

    print(
        f'\nTrain データ数: {X_train.shape[0]} (正例:'
        f' {int(np.sum(y_train == 1))}/陰性: {int(np.sum(y_train == 0))})'
    )
    print(
        f'Test  データ数: {X_test.shape[0]} (正例:'
        f' {int(np.sum(y_test == 1))}/陰性: {int(np.sum(y_test == 0))})'
    )
    print(f'特徴量次元数 : {X_train.shape[1]}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用デバイス  : {device}')

    # 損失関数の不均衡重み (pos_weight)
    pos_weight_val = np.sum(y_train == 0) / np.sum(y_train == 1)
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(
        device
    )

    # -------------------------------------------------------------
    # STEP 1: 5-Fold 交差検証 (CV)
    # -------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_aucs, cv_f1s, cv_sens, cv_precs, cv_mccs = [], [], [], [], []
    train_aucs = []
    best_thresholds = []

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[t_idx], X_train[v_idx]
        y_tr, y_val = y_train[t_idx], y_train[v_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Tensor化 & DataLoader 化
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

        model = train_model(
            model,
            tr_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs=100,
            patience=15,
        )

        val_probs = predict_probs(model, val_loader, device)

        # Train AUC 算出用
        tr_loader_eval = DataLoader(tr_ds, batch_size=64, shuffle=False)
        tr_probs = predict_probs(model, tr_loader_eval, device)

        opt_thresh, _ = find_best_threshold(y_val, val_probs, metric='f1')
        best_thresholds.append(opt_thresh)

        val_preds = (val_probs >= opt_thresh).astype(int)

        cv_aucs.append(roc_auc_score(y_val, val_probs))
        train_aucs.append(roc_auc_score(y_tr, tr_probs))
        cv_f1s.append(f1_score(y_val, val_preds, zero_division=0))
        cv_sens.append(recall_score(y_val, val_preds, zero_division=0))
        cv_precs.append(precision_score(y_val, val_preds, zero_division=0))
        cv_mccs.append(matthews_corrcoef(y_val, val_preds))

    mean_opt_thresh = np.mean(best_thresholds)

    print('\n--- 5-Fold CV 評価結果 ---')
    print(f'Train AUC        : {np.mean(train_aucs):.4f}')
    print(f'ROC-AUC (CV)     : {np.mean(cv_aucs):.4f}')
    print(f'Sensitivity (CV) : {np.mean(cv_sens):.4f}')
    print(f'Precision (CV)   : {np.mean(cv_precs):.4f}')
    print(f'F1-Score (CV)    : {np.mean(cv_f1s):.4f}')
    print(f'MCC (CV)         : {np.mean(cv_mccs):.4f}')
    print(f'平均最適閾値     : {mean_opt_thresh:.4f}')

    # -------------------------------------------------------------
    # STEP 2: 全 Train データで再学習 & Test データ評価
    # -------------------------------------------------------------
    scaler_full = StandardScaler()
    X_train_s = scaler_full.fit_transform(X_train)
    X_test_s = scaler_full.transform(X_test)

    full_tr_ds = TensorDataset(
        torch.tensor(X_train_s), torch.tensor(y_train, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_s), torch.tensor(y_test, dtype=torch.float32)
    )

    full_tr_loader = DataLoader(full_tr_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model_full = ESM1DCNN(
        input_dim=X_train.shape[1], dropout_rate=0.4
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model_full.parameters(),
                            lr=1e-3, weight_decay=1e-2)

    # 全データ再学習 (検証セットなし、エポック数をCVでの早期終了平均程度に指定)
    model_full.train()
    for epoch in range(35):
        for batch_x, batch_y in full_tr_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model_full(batch_x).squeeze(1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    test_probs = predict_probs(model_full, test_loader, device)
    test_preds = (test_probs >= mean_opt_thresh).astype(int)

    test_auc = roc_auc_score(y_test, test_probs)
    test_sens = recall_score(y_test, test_preds, zero_division=0)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_acc = accuracy_score(y_test, test_preds)
    test_mcc = matthews_corrcoef(y_test, test_preds)

    print('\n=== Test データ評価結果 ===')
    print(f'ROC-AUC (Test)   : {test_auc:.4f}')
    print(f'Sensitivity(Test): {test_sens:.4f}')
    print(f'Precision (Test) : {test_prec:.4f}')
    print(f'F1-Score (Test)  : {test_f1:.4f}')
    print(f'Accuracy (Test)  : {test_acc:.4f}')
    print(f'MCC (Test)       : {test_mcc:.4f}')

    # -------------------------------------------------------------
    # STEP 3: 結果の保存
    # -------------------------------------------------------------
    res_df = pd.DataFrame([{
        'モデル': '1D-CNN',
        '特徴量': os.path.basename(train_path),
        '次元数': X_train.shape[1],
        'Train AUC': round(np.mean(train_aucs), 4),
        'ROC-AUC (CV)': round(np.mean(cv_aucs), 4),
        'Sensitivity (CV)': round(np.mean(cv_sens), 4),
        'Precision (CV)': round(np.mean(cv_precs), 4),
        'F1-Score (CV)': round(np.mean(cv_f1s), 4),
        'MCC (CV)': round(np.mean(cv_mccs), 4),
        'Opt Threshold': round(mean_opt_thresh, 4),
        'ROC-AUC (Test)': round(test_auc, 4),
        'Sensitivity (Test)': round(test_sens, 4),
        'Precision (Test)': round(test_prec, 4),
        'F1-Score (Test)': round(test_f1, 4),
        'Accuracy (Test)': round(test_acc, 4),
        'MCC (Test)': round(test_mcc, 4),
    }])

    output_dir = 'eval_results'
    os.makedirs(output_dir, exist_ok=True)

    if not out_filename.endswith('.xlsx'):
        out_filename += '.xlsx'

    output_path = os.path.join(output_dir, out_filename)
    res_df.to_excel(output_path, index=False)
    print(f"\n✅ 評価結果を出力しました: '{output_path}'")


if __name__ == '__main__':
    main()
