import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# シード固定


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# -------------------------------------------------------------
# 1. 中間層ベクトル出力対応 1D-CNN アーキテクチャ
# -------------------------------------------------------------


class Peptide1DCNN(nn.Module):

    def __init__(self, input_dim):
        super(Peptide1DCNN, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) if input_dim >= 2 else nn.Identity(),
            nn.Dropout(0.3),
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.3),
        )

        self.fc_net = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )

    def forward(self, x, return_features=False):
        x = x.unsqueeze(1)  # -> [Batch, 1, input_dim]
        features = self.conv_net(x)  # -> [Batch, 64, 1]
        features = features.squeeze(2)  # -> [Batch, 64] (中間層ベクトル)

        if return_features:
            return features  # 64次元の特徴量ベクトルを返す

        out = self.fc_net(features)  # -> [Batch, 1]
        return out


# -------------------------------------------------------------
# 2. メイン処理部 (特徴量抽出 & Excel保存)
# -------------------------------------------------------------


def main():
    print("=== 1D-CNN 中間層ベクトル抽出処理 ===")

    train_path = input(
        "学習用特徴量Excelファイル名 (例: train_features.xlsx): "
    ).strip()
    test_path = input(
        "テスト用特徴量Excelファイル名 (例: test_features.xlsx): "
    ).strip()

    output_dir = input(
        "抽出結果保存フォルダ名 [未入力時は cnn_extracted_features]: "
    ).strip()
    if not output_dir:
        output_dir = "cnn_extracted_features"
    os.makedirs(output_dir, exist_ok=True)

    # データ読み込み
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    seq_train = (
        df_train["SEQUENCE"].values
        if "SEQUENCE" in df_train.columns
        else np.arange(len(df_train))
    )
    y_train = df_train["LABEL"].values.astype(int)
    X_train_cols = df_train.drop(
        columns=["SEQUENCE", "LABEL"], errors="ignore"
    )

    seq_test = (
        df_test["SEQUENCE"].values
        if "SEQUENCE" in df_test.columns
        else np.arange(len(df_test))
    )
    y_test = df_test["LABEL"].values.astype(int)
    X_test_cols = df_test.drop(columns=["SEQUENCE", "LABEL"], errors="ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    max_epochs = 300
    patience = 25

    # 標準化
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_cols.values)
    X_te_scaled = scaler.transform(X_test_cols.values)

    # Train内での Early Stopping 評価用 Validation 分割
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, val_idx = next(skf.split(X_tr_scaled, y_train))

    ds_tr = TensorDataset(
        torch.tensor(X_tr_scaled[tr_idx], dtype=torch.float32),
        torch.tensor(y_train[tr_idx], dtype=torch.float32).unsqueeze(1),
    )
    ds_va = TensorDataset(
        torch.tensor(X_tr_scaled[val_idx], dtype=torch.float32),
        torch.tensor(y_train[val_idx], dtype=torch.float32).unsqueeze(1),
    )

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    # モデル定義
    model = Peptide1DCNN(input_dim=X_train_cols.shape[1]).to(device)
    n_neg, n_pos = np.sum(y_train[tr_idx] == 0), np.sum(y_train[tr_idx] == 1)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(
        device
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    # 1. 1D-CNN の学習 (Early Stopping 付き)
    print("\n⚙️ 1D-CNN を学習中...")
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        # Val Loss チェック
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in loader_va:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * len(by)
        val_loss /= len(ds_va)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)
    model.eval()

    # 2. 中間層ベクトル（64次元）の抽出関数
    def extract_features(data_scaled):
        tensor_data = torch.tensor(data_scaled, dtype=torch.float32).to(device)
        loader = DataLoader(
            TensorDataset(tensor_data), batch_size=batch_size, shuffle=False
        )
        features_list = []
        with torch.no_grad():
            for (bx,) in loader:
                feats = model(bx, return_features=True)
                features_list.append(feats.cpu().numpy())
        return np.vstack(features_list)

    print("📦 中間層ベクトルの抽出中...")
    cnn_feats_tr = extract_features(X_tr_scaled)
    cnn_feats_te = extract_features(X_te_scaled)

    # 3. Excel 用 DataFrame の作成（SEQUENCE + LABEL + CNN_00 ~ CNN_63）
    feat_cols = [f"CNN_feat_{i:02d}" for i in range(cnn_feats_tr.shape[1])]

    df_out_tr = pd.DataFrame(
        {"SEQUENCE": seq_train, "LABEL": y_train}
    )
    df_out_tr = pd.concat([df_out_tr, pd.DataFrame(
        cnn_feats_tr, columns=feat_cols)], axis=1)

    df_out_te = pd.DataFrame(
        {"SEQUENCE": seq_test, "LABEL": y_test}
    )
    df_out_te = pd.concat([df_out_te, pd.DataFrame(
        cnn_feats_te, columns=feat_cols)], axis=1)

    # 保存
    out_tr_path = os.path.join(output_dir, "cnn_extracted_train.xlsx")
    out_te_path = os.path.join(output_dir, "cnn_extracted_test.xlsx")

    df_out_tr.to_excel(out_tr_path, index=False)
    df_out_te.to_excel(out_te_path, index=False)

    print("\n✅ 中間層ベクトルの抽出・保存が完了しました！")
    print(f" 📄 Train用抽出データ: '{out_tr_path}'")
    print(f" 📄 Test用抽出データ : '{out_te_path}'")


if __name__ == "__main__":
    main()
