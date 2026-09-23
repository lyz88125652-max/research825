import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
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

# 乱数シード固定


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# --- 1. 3層隠れ層の Neural Network アーキテクチャ ---


class PeptideMLP(nn.Module):

    def __init__(self, input_dim, hidden1=128, hidden2=64, hidden3=32):
        super(PeptideMLP, self).__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 2
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 3
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Output Layer
            nn.Linear(hidden3, 1),
        )

    def forward(self, x):
        return self.net(x)


# 損失（Loss/Cost）計算用ヘルパー関数


def compute_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = criterion(out, by)
            total_loss += loss.item() * len(by)
            total_samples += len(by)
    return total_loss / max(total_samples, 1)


# --- 2. メイン処理部 ---


def main():
    print("=== 特徴量組み合わせ × 3層 Neural Network 評価 (NN.py) ===")

    # 1. 学習用ファイルの入力
    while True:
        train_path = input(
            "学習用特徴量Excelファイル名 (例: train_features.xlsx): "
        ).strip()
        if not train_path:
            continue
        if not os.path.exists(train_path):
            print(f"❌ エラー: 指定されたファイル '{train_path}' が見つかりません。")
        else:
            break

    # 2. テスト用ファイルの入力
    while True:
        test_path = input(
            "テスト用特徴量Excelファイル名 (例: test_features.xlsx): "
        ).strip()
        if not test_path:
            continue
        if not os.path.exists(test_path):
            print(f"❌ エラー: 指定されたファイル '{test_path}' が見つかりません。")
        else:
            break

    # 保存先フォルダの設定
    output_dir = input("結果保存フォルダ名 [未入力時は nn_results]: ").strip()
    if not output_dir:
        output_dir = "nn_results"

    # CNNと対比するための Y軸上限値の設定 (デフォルト 1.0)
    y_limit_input = input(
        "グラフの Y 軸上限値 (Loss の最大値) [未入力時は 1.0]: "
    ).strip()
    try:
        y_limit = float(y_limit_input) if y_limit_input else 1.0
    except ValueError:
        y_limit = 1.0

    curves_dir = os.path.join(output_dir, "loss_curves")
    os.makedirs(curves_dir, exist_ok=True)
    excel_output_path = os.path.join(output_dir, "nn_summary.xlsx")

    # データの読み込み
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    y_train_all = df_train["LABEL"].values.astype(int)
    X_train_all = df_train.drop(columns=["SEQUENCE", "LABEL"])

    y_test_all = df_test["LABEL"].values.astype(int)
    X_test_all = df_test.drop(columns=["SEQUENCE", "LABEL"])

    # 特徴量手法の抽出
    method_prefixes = ["AAC_", "DPC_", "CTDC_", "CTDT_", "MB_"]
    method_columns = {
        p.rstrip("_"): [c for c in X_train_all.columns if c.startswith(p)]
        for p in method_prefixes
    }
    method_names = [m for m in method_columns if len(method_columns[m]) > 0]

    # 全31パターンの組み合わせ作成
    all_combinations = []
    for r in range(1, len(method_names) + 1):
        for comb in itertools.combinations(method_names, r):
            all_combinations.append(comb)

    print(f"\n💡 検出された手法: {method_names}")
    print(
        f"⚙️ 全 {len(all_combinations)} パターンに対する学習・評価を開始します...\n"
    )

    results_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 30
    batch_size = 32

    for idx, comb in enumerate(all_combinations, 1):
        comb_name = " + ".join(comb)
        selected_cols = []
        for m in comb:
            selected_cols.extend(method_columns[m])

        X_train_sub = X_train_all[selected_cols].values
        X_test_sub = X_test_all[selected_cols].values

        # -------------------------------------------------------------
        # STEP 1: Trainデータ内での Stratified 5-Fold 交差検証
        # -------------------------------------------------------------
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_epoch_train_costs = np.zeros(num_epochs)
        cv_epoch_val_costs = np.zeros(num_epochs)

        auc_scores, f1_scores, acc_scores, rec_scores, prec_scores = (
            [],
            [],
            [],
            [],
            [],
        )

        for fold, (t_idx, v_idx) in enumerate(skf.split(X_train_sub, y_train_all)):
            X_tr, X_val = X_train_sub[t_idx], X_train_sub[v_idx]
            y_tr, y_val = y_train_all[t_idx], y_train_all[v_idx]

            # 標準化
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            # クラス不均衡補正重み (pos_weight)
            n_neg = np.sum(y_tr == 0)
            n_pos = np.sum(y_tr == 1)
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(
                device
            )

            # DataLoaders
            ds_tr = TensorDataset(
                torch.tensor(X_tr_scaled, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
            )
            ds_val = TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
            )

            loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
            loader_tr_eval = DataLoader(
                ds_tr, batch_size=batch_size, shuffle=False)
            loader_val = DataLoader(
                ds_val, batch_size=batch_size, shuffle=False)

            # モデルの初期化
            model = PeptideMLP(input_dim=len(selected_cols)).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.AdamW(
                model.parameters(), lr=0.001, weight_decay=1e-4)

            # エポック学習
            for epoch in range(num_epochs):
                model.train()
                for bx, by in loader_tr:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()

                # コスト（損失）記録
                tr_c = compute_loss(model, loader_tr_eval, criterion, device)
                val_c = compute_loss(model, loader_val, criterion, device)
                cv_epoch_train_costs[epoch] += tr_c / 5.0
                cv_epoch_val_costs[epoch] += val_c / 5.0

            # 各フォールドの最終評価指標の算出
            model.eval()
            with torch.no_grad():
                val_x = torch.tensor(
                    X_val_scaled, dtype=torch.float32).to(device)
                logits = model(val_x).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                auc_scores.append(roc_auc_score(y_val, probs))
                f1_scores.append(f1_score(y_val, preds, zero_division=0))
                acc_scores.append(accuracy_score(y_val, preds))
                rec_scores.append(recall_score(y_val, preds, zero_division=0))
                prec_scores.append(precision_score(
                    y_val, preds, zero_division=0))

        # -------------------------------------------------------------
        # STEP 2: Epochs vs Train/CV Loss グラフ描画・保存 (Y軸統一)
        # -------------------------------------------------------------
        plt.figure(figsize=(7, 4.5))
        plt.plot(
            range(1, num_epochs + 1),
            cv_epoch_train_costs,
            label="Train Loss",
            color="#1f77b4",
            linewidth=2,
        )
        plt.plot(
            range(1, num_epochs + 1),
            cv_epoch_val_costs,
            label="Validation Loss",
            color="#ff7f0e",
            linewidth=2,
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss (BCE Loss)")
        plt.title(f"NN Loss Curve: Pattern {idx:02d}\n({comb_name})")

        # 【重要】CNNとの比較のため Y軸の範囲・刻み幅を固定設定
        plt.ylim(0.0, y_limit)
        plt.yticks(np.arange(0.0, y_limit + 0.05, 0.1))

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(
            os.path.join(curves_dir, f"pattern_{idx:02d}_loss_curve.png"), dpi=200
        )
        plt.close()

        # -------------------------------------------------------------
        # STEP 3: Train全体で再学習し、Test Loss および Test 評価指標を算出
        # -------------------------------------------------------------
        scaler_full = StandardScaler()
        X_train_full_scaled = scaler_full.fit_transform(X_train_sub)
        X_test_full_scaled = scaler_full.transform(X_test_sub)

        n_neg_full = np.sum(y_train_all == 0)
        n_pos_full = np.sum(y_train_all == 1)
        pos_weight_full = torch.tensor(
            [n_neg_full / max(n_pos_full, 1)], dtype=torch.float32
        ).to(device)

        ds_train_full = TensorDataset(
            torch.tensor(X_train_full_scaled, dtype=torch.float32),
            torch.tensor(y_train_all, dtype=torch.float32).unsqueeze(1),
        )
        ds_test_full = TensorDataset(
            torch.tensor(X_test_full_scaled, dtype=torch.float32),
            torch.tensor(y_test_all, dtype=torch.float32).unsqueeze(1),
        )

        loader_train_full = DataLoader(
            ds_train_full, batch_size=batch_size, shuffle=True
        )
        loader_test_full = DataLoader(
            ds_test_full, batch_size=batch_size, shuffle=False
        )

        model_full = PeptideMLP(input_dim=len(selected_cols)).to(device)
        criterion_full = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
        optimizer_full = optim.AdamW(
            model_full.parameters(), lr=0.001, weight_decay=1e-4
        )

        for epoch in range(num_epochs):
            model_full.train()
            for bx, by in loader_train_full:
                bx, by = bx.to(device), by.to(device)
                optimizer_full.zero_grad()
                loss = criterion_full(model_full(bx), by)
                loss.backward()
                optimizer_full.step()

        # Test Loss 計算
        test_loss = compute_loss(
            model_full, loader_test_full, criterion_full, device
        )

        # Testデータに対する推論と各種評価指標の計算
        model_full.eval()
        with torch.no_grad():
            x_test_tensor = torch.tensor(
                X_test_full_scaled, dtype=torch.float32
            ).to(device)
            test_logits = model_full(x_test_tensor).squeeze(1)
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            test_preds = (test_probs >= 0.5).astype(int)

            test_auc = roc_auc_score(y_test_all, test_probs)
            test_f1 = f1_score(y_test_all, test_preds, zero_division=0)
            test_rec = recall_score(y_test_all, test_preds, zero_division=0)
            test_prec = precision_score(
                y_test_all, test_preds, zero_division=0)
            test_acc = accuracy_score(y_test_all, test_preds)

        # 集計結果の追加
        mean_auc = np.mean(auc_scores)
        mean_f1 = np.mean(f1_scores)
        mean_acc = np.mean(acc_scores)
        mean_rec = np.mean(rec_scores)
        mean_prec = np.mean(prec_scores)
        train_loss_final = cv_epoch_train_costs[-1]
        cv_loss_final = cv_epoch_val_costs[-1]
        loss_diff = cv_loss_final - train_loss_final

        results_list.append({
            "パターンNo": idx,
            "手法数": len(comb),
            "手法の組み合わせ": comb_name,
            "次元数": len(selected_cols),
            "ROC-AUC (CV)": round(mean_auc, 4),
            "F1-Score (CV)": round(mean_f1, 4),
            "Sensitivity (CV)": round(mean_rec, 4),
            "Precision (CV)": round(mean_prec, 4),
            "Accuracy (CV)": round(mean_acc, 4),
            "Train Loss": round(train_loss_final, 4),
            "Validation Loss": round(cv_loss_final, 4),
            "Loss_Diff (Val - Train)": round(loss_diff, 4),
            "ROC-AUC (Test)": round(test_auc, 4),
            "F1-Score (Test)": round(test_f1, 4),
            "Sensitivity (Test)": round(test_rec, 4),
            "Precision (Test)": round(test_prec, 4),
            "Accuracy (Test)": round(test_acc, 4),
            "Test Loss": round(test_loss, 4),
        })

        print(
            f" [{idx:02d}/31] {comb_name:<35} | CV AUC: {mean_auc:.4f} | Test AUC:"
            f" {test_auc:.4f} | Val Loss: {cv_loss_final:.4f}"
        )

    # -------------------------------------------------------------
    # STEP 4: 総合スコア（Composite Score）による並び替えと出力
    # -------------------------------------------------------------
    res_df = pd.DataFrame(results_list)

    auc_min, auc_max = res_df["ROC-AUC (CV)"].min(
    ), res_df["ROC-AUC (CV)"].max()
    val_l_min, val_l_max = (
        res_df["Validation Loss"].min(),
        res_df["Validation Loss"].max(),
    )
    diff_min, diff_max = (
        res_df["Loss_Diff (Val - Train)"].min(),
        res_df["Loss_Diff (Val - Train)"].max(),
    )

    # 正規化 (0~1)
    auc_norm = (res_df["ROC-AUC (CV)"] - auc_min) / (auc_max - auc_min + 1e-8)
    val_l_norm = (res_df["Validation Loss"] - val_l_min) / (
        val_l_max - val_l_min + 1e-8
    )
    diff_norm = (res_df["Loss_Diff (Val - Train)"] - diff_min) / (
        diff_max - diff_min + 1e-8
    )

    # CNNと基準を統一: AUC重視(0.6) - Val Lossペナルティ(0.2) - 過学習差分ペナルティ(0.2)
    res_df["Composite_Score"] = round(
        0.6 * auc_norm - 0.2 * val_l_norm - 0.2 * diff_norm, 4
    )

    # 総合スコア（Composite_Score）順に降順ソート
    res_df = res_df.sort_values(by="Composite_Score", ascending=False)
    res_df.to_excel(excel_output_path, index=False)

    # 上位10案の複合スコア比較グラフ出力
    top10 = res_df.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(
        top10["手法の組み合わせ"][::-1],
        top10["Composite_Score"][::-1],
        color="#2b5c8f",
    )
    plt.xlabel(
        "Composite Score (0.6*AUC_norm - 0.2*ValLoss_norm - 0.2*LossDiff_norm)"
    )
    plt.title(
        "Top 10 Feature Combinations by Neural Network (Composite Score Ranking)"
    )
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "top10_nn_composite_score.png"), dpi=200
    )
    plt.close()

    print("\n✅ 全31パターンの処理が正常に完了しました！")
    print(f" 📊 総合スコア順の集計Excel: '{excel_output_path}'")
    print(
        f" 📈 上位10案の総合スコアグラフ: '{output_dir}/top10_nn_composite_score.png'"
    )
    print(
        f" 📈 各パターンの学習曲線画像: '{curves_dir}/pattern_XX_loss_curve.png'"
    )


if __name__ == "__main__":
    main()
