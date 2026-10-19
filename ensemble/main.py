import os
import random
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

# 各モデルモジュールのインポート
from model_cnn import run_cnn
from model_lgb import run_lgb
from model_rf import run_rf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def calculate_metrics(y_true, probs, thresh=0.5):
    """評価指標を安全に計算する関数（全正例/全負例の単一クラスデータセットに対応）"""
    preds = (probs >= thresh).astype(int)
    unique_classes = np.unique(y_true)

    # テストデータに正例・負例の両方が揃っている場合のみ AUC と MCC を計算
    if len(unique_classes) > 1:
        roc_auc = round(roc_auc_score(y_true, probs), 4)
        mcc = round(matthews_corrcoef(y_true, preds), 4)
    else:
        # クラスが1つしかない場合（例: 全て正例）は計算不可のため NaN
        roc_auc = np.nan
        mcc = np.nan

    return {
        "ROC-AUC": roc_auc,
        "Sensitivity": round(recall_score(y_true, preds, zero_division=0), 4),
        "Precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, preds, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_true, preds), 4),
        "MCC": mcc,
    }


def load_data(train_file, test_file):
    """Excel ファイルからデータとラベルを読み込む（文字列列を自動排除）"""
    print(f"\n学習データの読み込み中: {train_file}")
    df_train = pd.read_excel(train_file)
    y_train = df_train["LABEL"].values

    # LABEL 列と、アミノ酸配列などの文字列列を除外して数値特徴量のみ抽出
    X_train_df = df_train.drop(columns=["LABEL"]).select_dtypes(
        include=[np.number]
    )
    X_train = X_train_df.values

    print(f"テストデータの読み込み中: {test_file}")
    df_test = pd.read_excel(test_file)
    y_test = df_test["LABEL"].values

    # テストデータも同様に数値特徴量のみ抽出（列順を学習データに合わせる）
    X_test_df = df_test[X_train_df.columns]
    X_test = X_test_df.values

    pos_tr = np.sum(y_train == 1)
    neg_tr = np.sum(y_train == 0)
    pos_te = np.sum(y_test == 1)
    neg_te = np.sum(y_test == 0)

    print(f"特徴量数: {X_train.shape[1]} 次元")
    print(f"Train データ数: {len(y_train)} (正例: {pos_tr} / 陰性: {neg_tr})")
    print(f"Test  データ数: {len(y_test)} (正例: {pos_te} / 陰性: {neg_te})")

    return X_train, y_train, X_test, y_test


def main():
    print("=" * 58)
    print("  マルチモデル・アンサンブル評価パイプライン (ESM-2 対応)")
    print("=" * 58)

    train_file = (
        input("学習用Excelファイル名 (デフォルト: trainESMft.xlsx): ").strip()
        or "trainESMft.xlsx"
    )
    test_file = (
        input("テスト用Excelファイル名 (デフォルト: testESMft.xlsx): ").strip()
        or "testESMft.xlsx"
    )
    output_file = (
        input("出力Excelファイル名 (デフォルト: ensemble_eval_result.xlsx): ").strip()
        or "ensemble_eval_result.xlsx"
    )

    X_train, y_train, X_test, y_test = load_data(train_file, test_file)

    # 1. LightGBM の実行
    print("\n>>> [1/3] LightGBM の実行...")
    res_lgb = run_lgb(X_train, y_train, X_test, y_test)

    # 2. Random Forest の実行
    print("\n>>> [2/3] Random Forest の実行...")
    res_rf = run_rf(X_train, y_train, X_test, y_test)

    # 3. 1D-CNN の実行
    print("\n>>> [3/3] 1D-CNN (PyTorch) の実行...")
    res_cnn = run_cnn(X_train, y_train, X_test, y_test)

    # 4. Soft Voting アンサンブル
    print("\n>>> アンサンブル処理中 (Soft Voting)...")
    ens_cv_probs = (
        res_lgb["cv_probs"] + res_rf["cv_probs"] + res_cnn["cv_probs"]
    ) / 3.0
    ens_test_probs = (
        res_lgb["test_probs"] + res_rf["test_probs"] + res_cnn["test_probs"]
    ) / 3.0

    # アンサンブル最適閾値の探索（CV OOF）
    best_thresh_ens = 0.5
    best_mcc_ens = -1.0
    unique_train_classes = np.unique(y_train)

    if len(unique_train_classes) > 1:
        for t in np.arange(0.15, 0.70, 0.01):
            preds = (ens_cv_probs >= t).astype(int)
            mcc = matthews_corrcoef(y_train, preds)
            if mcc > best_mcc_ens:
                best_mcc_ens = mcc
                best_thresh_ens = t
        ens_cv_auc = round(roc_auc_score(y_train, ens_cv_probs), 4)
    else:
        ens_cv_auc = np.nan

    ens_train_auc = round(
        (res_lgb["train_auc"] + res_rf["train_auc"] +
         res_cnn["train_auc"]) / 3.0,
        4,
    )

    # 各モデルの指標算出（テストデータ）
    m_lgb = calculate_metrics(
        y_test, res_lgb["test_probs"], res_lgb["opt_thresh"]
    )
    m_rf = calculate_metrics(
        y_test, res_rf["test_probs"], res_rf["opt_thresh"])
    m_cnn = calculate_metrics(
        y_test, res_cnn["test_probs"], res_cnn["opt_thresh"]
    )
    m_ens = calculate_metrics(y_test, ens_test_probs, best_thresh_ens)

    # 結果テーブルの作成
    df_summary = pd.DataFrame([
        {
            "モデル": "LightGBM",
            "Train AUC": res_lgb["train_auc"],
            "ROC-AUC (CV)": res_lgb["cv_auc"],
            "Opt Threshold": round(res_lgb["opt_thresh"], 3),
            "ROC-AUC (Test)": m_lgb["ROC-AUC"],
            "Sensitivity (Test)": m_lgb["Sensitivity"],
            "Precision (Test)": m_lgb["Precision"],
            "F1-Score (Test)": m_lgb["F1-Score"],
            "Accuracy (Test)": m_lgb["Accuracy"],
            "MCC (Test)": m_lgb["MCC"],
            "Min CV Loss": np.nan,
        },
        {
            "モデル": "Random Forest",
            "Train AUC": res_rf["train_auc"],
            "ROC-AUC (CV)": res_rf["cv_auc"],
            "Opt Threshold": round(res_rf["opt_thresh"], 3),
            "ROC-AUC (Test)": m_rf["ROC-AUC"],
            "Sensitivity (Test)": m_rf["Sensitivity"],
            "Precision (Test)": m_rf["Precision"],
            "F1-Score (Test)": m_rf["F1-Score"],
            "Accuracy (Test)": m_rf["Accuracy"],
            "MCC (Test)": m_rf["MCC"],
            "Min CV Loss": np.nan,
        },
        {
            "モデル": "1D-CNN (5-Fold Ens)",
            "Train AUC": round(res_cnn["train_auc"], 4),
            "ROC-AUC (CV)": round(res_cnn["cv_auc"], 4),
            "Opt Threshold": round(res_cnn["opt_thresh"], 3),
            "ROC-AUC (Test)": m_cnn["ROC-AUC"],
            "Sensitivity (Test)": m_cnn["Sensitivity"],
            "Precision (Test)": m_cnn["Precision"],
            "F1-Score (Test)": m_cnn["F1-Score"],
            "Accuracy (Test)": m_cnn["Accuracy"],
            "MCC (Test)": m_cnn["MCC"],
            "Min CV Loss": round(res_cnn["min_cv_loss"], 3),
        },
        {
            "モデル": "Ensemble (LGBM+RF+CNN)",
            "Train AUC": ens_train_auc,
            "ROC-AUC (CV)": ens_cv_auc,
            "Opt Threshold": round(best_thresh_ens, 3),
            "ROC-AUC (Test)": m_ens["ROC-AUC"],
            "Sensitivity (Test)": m_ens["Sensitivity"],
            "Precision (Test)": m_ens["Precision"],
            "F1-Score (Test)": m_ens["F1-Score"],
            "Accuracy (Test)": m_ens["Accuracy"],
            "MCC (Test)": m_ens["MCC"],
            "Min CV Loss": np.nan,
        },
    ])

    # テストデータごとの予測確率シートの作成
    df_preds = pd.DataFrame({
        "LABEL": y_test,
        "LGBM_Prob": res_lgb["test_probs"],
        "RF_Prob": res_rf["test_probs"],
        "CNN_Prob": res_cnn["test_probs"],
        "Ensemble_Prob": ens_test_probs,
        "Ensemble_Pred": (ens_test_probs >= best_thresh_ens).astype(int),
    })

    # CNN の Loss 履歴シートの作成
    df_cnn_loss = pd.DataFrame({
        "Epoch": np.arange(1, len(res_cnn["mean_train_losses"]) + 1),
        "Mean_Train_Loss": res_cnn["mean_train_losses"],
        "Mean_CV_Loss": res_cnn["mean_val_losses"],
    })

    # Excel へ保存
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Model_Comparison", index=False)
        df_preds.to_excel(writer, sheet_name="Test_Predictions", index=False)
        df_cnn_loss.to_excel(
            writer, sheet_name="CNN_Loss_History", index=False)

    print(f"\n【処理完了】評価結果を保存しました: {output_file}")


if __name__ == "__main__":
    main()
