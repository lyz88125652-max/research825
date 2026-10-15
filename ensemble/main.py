import os
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

# -------------------------------------------------------------
# 各モデルファイルの関数をインポート
# （各モジュール側に train_and_eval 関数を定義しておきます）
# -------------------------------------------------------------
try:
    from model_cnn import run_cnn
    from model_lgb import run_lgbm
    from model_rf import run_rf
except ImportError as e:
    print(
        f"⚠️ モジュールのインポートエラー: {e}\n`model_lgb.py`, `model_rf.py`,"
        " `model_cnn.py` が同じディレクトリにあるか確認してください。"
    )


def find_best_threshold(y_true, probs, metric="mcc"):
    """MCC または F1-Score を最大化する最適判定閾値を探索."""
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


def calculate_metrics(y_true, probs, threshold):
    """予測確率と閾値から各種評価指標を算出."""
    preds = (probs >= threshold).astype(int)
    return {
        "ROC-AUC": round(roc_auc_score(y_true, probs), 4),
        "Sensitivity": round(
            recall_score(y_true, preds, zero_division=0), 4
        ),
        "Precision": round(
            precision_score(y_true, preds, zero_division=0), 4
        ),
        "F1-Score": round(f1_score(y_true, preds, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_true, preds), 4),
        "MCC": round(matthews_corrcoef(y_true, preds), 4),
    }


def main():
    print("==========================================================")
    print("  マルチモデル・アンサンブル評価パイプライン (ESM-2 対応)")
    print("==========================================================")

    # 1. 入力ファイル設定
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
        "出力Excelファイル名 (デフォルト: ensemble_eval_result.xlsx): "
    ).strip()
    if not out_filename:
        out_filename = "ensemble_eval_result.xlsx"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ 入力ファイルが存在しません。処理を中断します。")
        return

    # 2. データ読み込み（位置指定: 1列目=配列, 2列目=ラベル, 3列目以降=1280次元）
    print("\nデータを読み込み中...")
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    y_train = df_train.iloc[:, 1].values.astype(int)
    X_train = df_train.iloc[:, 2:].values

    y_test = df_test.iloc[:, 1].values.astype(int)
    X_test = df_test.iloc[:, 2:].values

    print(
        f"Train データ数: {X_train.shape[0]} (正例:"
        f" {np.sum(y_train == 1)}/陰性: {np.sum(y_train == 0)})"
    )
    print(
        f"Test  データ数: {X_test.shape[0]} (正例:"
        f" {np.sum(y_test == 1)}/陰性: {np.sum(y_test == 0)})"
    )
    print(f"特徴量次元数 : {X_train.shape[1]}")

    # -------------------------------------------------------------
    # 3. 各単体モデルの実行（個別の python ファイルから呼び出し）
    # -------------------------------------------------------------
    # 各関数は dict を返す仕様とします:
    # {
    #   "cv_probs": ...,      # Validation (CV) の予測確率
    #   "test_probs": ...,    # Test データの予測確率
    #   "train_auc": ...,     # Train AUC の平均値
    #   "opt_thresh": ...     # 各モデルが決定した CV 最適閾値
    # }

    print("\n>>> [1/3] LightGBM の実行...")
    res_lgb = run_lgbm(X_train, y_train, X_test, y_test)

    print("\n>>> [2/3] Random Forest の実行...")
    res_rf = run_rf(X_train, y_train, X_test, y_test)

    print("\n>>> [3/3] 1D-CNN (PyTorch) の実行...")
    res_cnn = run_cnn(X_train, y_train, X_test, y_test)

    # -------------------------------------------------------------
    # 4. アンサンブル (Soft Voting: 確率平均)
    # -------------------------------------------------------------
    print("\n>>> アンサンブル処理中 (Soft Voting)...")

    # CV 段階でのアンサンブル予測確率と最適閾値決定
    cv_ensemble_probs = (
        res_lgb["cv_probs"] + res_rf["cv_probs"] + res_cnn["cv_probs"]
    ) / 3.0
    ens_opt_thresh, _ = find_best_threshold(
        y_train, cv_ensemble_probs, metric="mcc"
    )

    # Test 段階でのアンサンブル予測確率
    test_ensemble_probs = (
        res_lgb["test_probs"] + res_rf["test_probs"] + res_cnn["test_probs"]
    ) / 3.0

    # -------------------------------------------------------------
    # 5. 各モデルおよびアンサンブルの各種評価指標算出
    # -------------------------------------------------------------
    models_data = [
        ("LightGBM", res_lgb, res_lgb["test_probs"], res_lgb["opt_thresh"]),
        ("Random Forest", res_rf, res_rf["test_probs"], res_rf["opt_thresh"]),
        (
            "1D-CNN (5-Fold Ens)",
            res_cnn,
            res_cnn["test_probs"],
            res_cnn["opt_thresh"],
        ),
    ]

    results_summary = []

    for name, res_dict, t_probs, thresh in models_data:
        metrics = calculate_metrics(y_test, t_probs, thresh)
        results_summary.append({
            "モデル": name,
            "Train AUC": round(res_dict["train_auc"], 4),
            "ROC-AUC (CV)": round(res_dict["cv_auc"], 4),
            "Opt Threshold": round(thresh, 4),
            "ROC-AUC (Test)": metrics["ROC-AUC"],
            "Sensitivity (Test)": metrics["Sensitivity"],
            "Precision (Test)": metrics["Precision"],
            "F1-Score (Test)": metrics["F1-Score"],
            "Accuracy (Test)": metrics["Accuracy"],
            "MCC (Test)": metrics["MCC"],
        })

    # アンサンブルモデルの結果を追加
    ens_metrics = calculate_metrics(
        y_test, test_ensemble_probs, ens_opt_thresh
    )
    ens_cv_auc = roc_auc_score(y_train, cv_ensemble_probs)
    ens_train_auc = np.mean(
        [res_lgb["train_auc"], res_rf["train_auc"], res_cnn["train_auc"]]
    )

    results_summary.append({
        "モデル": "Ensemble (LGBM+RF+CNN)",
        "Train AUC": round(ens_train_auc, 4),
        "ROC-AUC (CV)": round(ens_cv_auc, 4),
        "Opt Threshold": round(ens_opt_thresh, 4),
        "ROC-AUC (Test)": ens_metrics["ROC-AUC"],
        "Sensitivity (Test)": ens_metrics["Sensitivity"],
        "Precision (Test)": ens_metrics["Precision"],
        "F1-Score (Test)": ens_metrics["F1-Score"],
        "Accuracy (Test)": ens_metrics["Accuracy"],
        "MCC (Test)": ens_metrics["MCC"],
    })

    df_results = pd.DataFrame(results_summary)

    # 画面表示
    print("\n=================== 最終評価結果比較 ===================")
    print(
        df_results[
            [
                "モデル",
                "ROC-AUC (Test)",
                "Sensitivity (Test)",
                "Precision (Test)",
                "F1-Score (Test)",
                "MCC (Test)",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------
    # 6. Excelファイルへ出力保存
    # -------------------------------------------------------------
    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)

    if not out_filename.endswith(".xlsx"):
        out_filename += ".xlsx"
    output_path = os.path.join(output_dir, out_filename)

    # 個別確率なども確認できるよう複数シートで出力
    df_probs = pd.DataFrame({
        "LABEL": y_test,
        "LGBM_Prob": res_lgb["test_probs"],
        "RF_Prob": res_rf["test_probs"],
        "CNN_Prob": res_cnn["test_probs"],
        "Ensemble_Prob": test_ensemble_probs,
        "Ensemble_Pred": (test_ensemble_probs >= ens_opt_thresh).astype(int),
    })

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Model_Comparison", index=False)
        df_probs.to_excel(writer, sheet_name="Test_Predictions", index=False)

    print(f"\n✅ すべての評価結果を保存しました: '{output_path}'")


if __name__ == "__main__":
    main()
