import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


def find_best_threshold(y_true, probs, metric="f1"):
    """F1-Score または MCC を最大化する最適な判定閾値を探索."""
    best_thresh = 0.5
    best_score = -1.0

    for t in np.arange(0.15, 0.65, 0.01):
        preds = (probs >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, preds)
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score


def main():
    print("=== Random Forest 評価スクリプト (ESM-2 特徴量対応) ===")

    # ファイル入力
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
        "出力ファイル名 (デフォルト: rf_esm_result.xlsx): "
    ).strip()
    if not out_filename:
        out_filename = "rf_esm_result.xlsx"

    # 1. データ読み込み (1列目=配列, 2列目=ラベル, 3列目以降=1280次元)
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    y_train = df_train.iloc[:, 1].values.astype(int)
    X_train = df_train.iloc[:, 2:].values

    y_test = df_test.iloc[:, 1].values.astype(int)
    X_test = df_test.iloc[:, 2:].values

    print(
        f"\nTrain データ数: {X_train.shape[0]} (正例:"
        f" {np.sum(y_train == 1)}/陰性: {np.sum(y_train == 0)})"
    )
    print(
        f"Test  データ数: {X_test.shape[0]} (正例:"
        f" {np.sum(y_test == 1)}/陰性: {np.sum(y_test == 0)})"
    )
    print(f"特徴量次元数 : {X_train.shape[1]}")

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

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Random Forest モデル構築 (過学習抑制 & 不均衡対策)
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,  # 木の深さを制限して過学習抑制 (Train AUC 1.0の防止)
            min_samples_split=10,  # 分割に必要な最小サンプル数
            min_samples_leaf=5,  # 葉に必要な最小サンプル数
            max_features="sqrt",  # 1280次元のうち sqrt(1280) ≈ 35 次元をランダム選択
            class_weight="balanced",  # 不均衡データ対策
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_tr_s, y_tr)

        val_probs = model.predict_proba(X_val_s)[:, 1]
        tr_probs = model.predict_proba(X_tr_s)[:, 1]

        # F1-Score 基準による最適閾値検索
        opt_thresh, _ = find_best_threshold(y_val, val_probs, metric="f1")
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

    model_full = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
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
        "モデル": "Random Forest",
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

    if not out_filename.endswith(".xlsx"):
        out_filename += ".xlsx"

    output_path = os.path.join(output_dir, out_filename)
    res_df.to_excel(output_path, index=False)
    print(f"\n✅ 評価結果を出力しました: '{output_path}'")


if __name__ == "__main__":
    main()
