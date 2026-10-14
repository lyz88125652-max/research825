import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    matthews_corrcoef,
)


def find_best_threshold(y_true, probs):
    """Youden's J Index (Sensitivity + Specificity - 1) を最大化する閾値を探索."""
    best_thresh = 0.5
    best_j = -1.0
    for t in np.arange(0.10, 0.60, 0.01):
        preds = (probs >= t).astype(int)
        sens = recall_score(y_true, preds, zero_division=0)

        tn = np.sum((y_true == 0) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        j_index = sens + spec - 1
        if j_index > best_j:
            best_j = j_index
            best_thresh = t
    return best_thresh, best_j


def main():
    print("=== LightGBM 単体評価スクリプト (ESM-2 特徴量対応) ===")

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

    # 1. データ読み込み
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    y_train = df_train["LABEL"].values.astype(int)
    X_train = df_train.drop(columns=["SEQUENCE", "LABEL"]).values

    y_test = df_test["LABEL"].values.astype(int)
    X_test = df_test.drop(columns=["SEQUENCE", "LABEL"]).values

    print(
        f"Train データ数: {X_train.shape[0]} (正例:"
        f" {np.sum(y_train == 1)}/陰性: {np.sum(y_train == 0)})"
    )
    print(
        f"Test  データ数: {X_test.shape[0]} (正例:"
        f" {np.sum(y_test == 1)}/陰性: {np.sum(y_test == 0)})"
    )
    print(f"特徴量次元数 : {X_train.shape[1]}")

    # 不均衡重み（陰性数 / 陽性数）
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    print(f"不均衡重み (scale_pos_weight): {scale_pos_weight:.2f}")

    # -------------------------------------------------------------
    # STEP 1: 5-Fold 交差検証 (CV)
    # -------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_aucs, cv_f1s, cv_sens, cv_precs, cv_mccs = [], [], [], [], []
    train_aucs = []
    best_thresholds = []

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[t_idx], X_train[v_idx]
        y_tr, y_val = y_train[t_idx], y_train[v_idx]

        # 特徴量の標準化
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # LightGBM モデル構築 (過学習抑制パラメータ設定)
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,  # 小さめの学習率でじっくり学習
            max_depth=4,  # 木の深さを制限して過学習抑制
            num_leaves=15,  # 葉ノード数制限
            subsample=0.8,  # 行のサンプリング
            colsample_bytree=0.8,  # 列（特徴量）のサンプリング
            scale_pos_weight=scale_pos_weight,  # 不均衡重み
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(X_tr_s, y_tr)

        # 予測確率
        val_probs = model.predict_proba(X_val_s)[:, 1]
        tr_probs = model.predict_proba(X_tr_s)[:, 1]

        # 最適閾値の検索
        opt_thresh, _ = find_best_threshold(y_val, val_probs)
        best_thresholds.append(opt_thresh)

        val_preds = (val_probs >= opt_thresh).astype(int)

        cv_aucs.append(roc_auc_score(y_val, val_probs))
        train_aucs.append(roc_auc_score(y_tr, tr_probs))
        cv_f1s.append(f1_score(y_val, val_preds, zero_division=0))
        cv_sens.append(recall_score(y_val, val_preds, zero_division=0))
        cv_precs.append(precision_score(y_val, val_preds, zero_division=0))
        cv_mccs.append(matthews_corrcoef(y_val, val_preds))

    mean_opt_thresh = np.mean(best_thresholds)

    print("\n--- 5-Fold CV 評価結果 ---")
    print(f"Train AUC        : {np.mean(train_aucs):.4f}")
    print(f"ROC-AUC (CV)     : {np.mean(cv_aucs):.4f}")
    print(f"Sensitivity (CV) : {np.mean(cv_sens):.4f}")
    print(f"Precision (CV)   : {np.mean(cv_precs):.4f}")
    print(f"F1-Score (CV)    : {np.mean(cv_f1s):.4f}")
    print(f"MCC (CV)         : {np.mean(cv_mccs):.4f}")
    print(f"平均最適閾値     : {mean_opt_thresh:.4f}")

    # -------------------------------------------------------------
    # STEP 2: 全 Train データで再学習 & Test データ評価
    # -------------------------------------------------------------
    scaler_full = StandardScaler()
    X_train_s = scaler_full.fit_transform(X_train)
    X_test_s = scaler_full.transform(X_test)

    model_full = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model_full.fit(X_train_s, y_train)

    test_probs = model_full.predict_proba(X_test_s)[:, 1]
    test_preds = (test_probs >= mean_opt_thresh).astype(int)

    test_auc = roc_auc_score(y_test, test_probs)
    test_sens = recall_score(y_test, test_preds, zero_division=0)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_acc = accuracy_score(y_test, test_preds)
    test_mcc = matthews_corrcoef(y_test, test_preds)

    print("\n=== Test データ評価結果 ===")
    print(f"ROC-AUC (Test)   : {test_auc:.4f}")
    print(f"Sensitivity(Test): {test_sens:.4f}")
    print(f"Precision (Test) : {test_prec:.4f}")
    print(f"F1-Score (Test)  : {test_f1:.4f}")
    print(f"Accuracy (Test)  : {test_acc:.4f}")
    print(f"MCC (Test)       : {test_mcc:.4f}")

    # -------------------------------------------------------------
    # STEP 3: 結果の保存
    # -------------------------------------------------------------
    res_df = pd.DataFrame([{
        "モデル": "LightGBM",
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

    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "lgbm_esm_result.xlsx")
    res_df.to_excel(output_path, index=False)
    print(f"\n✅ 評価結果を出力しました: '{output_path}'")


if __name__ == "__main__":
    main()
