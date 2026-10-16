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

try:
    from model_cnn import run_cnn
    from model_lgb import run_lgbm
    from model_rf import run_rf
except ImportError as e:
    print(f"⚠️ モジュールのインポートエラー: {e}")


def find_best_threshold(y_true, probs, metric="mcc"):
    best_thresh = 0.5
    best_score = -1.0
    for t in np.arange(0.15, 0.70, 0.01):
        preds = (probs >= t).astype(int)
        score = (
            matthews_corrcoef(y_true, preds)
            if metric == "mcc"
            else f1_score(y_true, preds, zero_division=0)
        )
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


def calculate_metrics(y_true, probs, threshold):
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

    print("\n>>> [1/3] LightGBM の実行...")
    res_lgb = run_lgbm(X_train, y_train, X_test, y_test)

    print("\n>>> [2/3] Random Forest の実行...")
    res_rf = run_rf(X_train, y_train, X_test, y_test)

    print("\n>>> [3/3] 1D-CNN (PyTorch) の実行...")
    res_cnn = run_cnn(X_train, y_train, X_test, y_test)

    print("\n>>> アンサンブル処理中 (Soft Voting)...")
    cv_ensemble_probs = (
        res_lgb["cv_probs"] + res_rf["cv_probs"] + res_cnn["cv_probs"]
    ) / 3.0
    ens_opt_thresh, _ = find_best_threshold(
        y_train, cv_ensemble_probs, metric="mcc"
    )

    test_ensemble_probs = (
        res_lgb["test_probs"] + res_rf["test_probs"] + res_cnn["test_probs"]
    ) / 3.0

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
        row = {
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
        }
        if "min_cv_loss" in res_dict:
            row["Min CV Loss"] = round(res_dict["min_cv_loss"], 4)
        results_summary.append(row)

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

    # CNN Loss 履歴データフレームを作成
    df_cnn_loss = pd.DataFrame({
        "Epoch": np.arange(1, len(res_cnn["mean_train_losses"]) + 1),
        "Mean_Train_Loss": res_cnn["mean_train_losses"],
        "Mean_CV_Loss": res_cnn["mean_val_losses"],
    })

    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)

    if not out_filename.endswith(".xlsx"):
        out_filename += ".xlsx"
    output_path = os.path.join(output_dir, out_filename)

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
        df_cnn_loss.to_excel(
            writer, sheet_name="CNN_Loss_History", index=False)

    print(f"\n✅ すべての評価結果を保存しました: '{output_path}'")
    print("   - シート 1: 'Model_Comparison'")
    print("   - シート 2: 'Test_Predictions'")
    print("   - シート 3: 'CNN_Loss_History' (エポック毎の平均Train/CV Loss)")
    print(f"   - グラフ画像: '{os.path.join(output_dir, 'cnn_loss_curve.png')}'")


if __name__ == "__main__":
    main()
