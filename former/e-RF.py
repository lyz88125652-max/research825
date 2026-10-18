import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 乱数シード固定


def set_seed(seed=42):
    np.random.seed(seed)


set_seed(42)


def main():
    print(
        "=== 1D-CNN抽出ベクトル × Random Forest 評価 (OOB / CV / Test) ==="
    )

    # 1. 入力ファイルの指定
    while True:
        train_path = input(
            "1D-CNN抽出Trainファイル名 (例: cnn_extracted_train.xlsx): "
        ).strip()
        if not train_path:
            continue
        if not os.path.exists(train_path):
            print(f"❌ エラー: 指定されたファイル '{train_path}' が見つかりません。")
        else:
            break

    while True:
        test_path = input(
            "1D-CNN抽出Testファイル名 (例: cnn_extracted_test.xlsx): "
        ).strip()
        if not test_path:
            continue
        if not os.path.exists(test_path):
            print(f"❌ エラー: 指定されたファイル '{test_path}' が見つかりません。")
        else:
            break

    output_dir = input(
        "結果保存フォルダ名 [未入力時は rf_results]: "
    ).strip()
    if not output_dir:
        output_dir = "rf_results"
    os.makedirs(output_dir, exist_ok=True)

    excel_output_path = os.path.join(output_dir, "rf_evaluation_summary.xlsx")

    # 2. データの読み込み
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    y_train = df_train["LABEL"].values.astype(int)
    X_train = df_train.drop(
        columns=["SEQUENCE", "LABEL"], errors="ignore"
    ).values

    y_test = df_test["LABEL"].values.astype(int)
    X_test = df_test.drop(
        columns=["SEQUENCE", "LABEL"], errors="ignore").values

    print(
        f"\n⚙️ データ読み込み完了: Train {X_train.shape[0]}件, Test"
        f" {X_test.shape[0]}件 | 特徴量数: {X_train.shape[1]}次元\n"
    )

    # -------------------------------------------------------------
    # STEP 1: 5-Fold Cross Validation (CV) 評価
    # -------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc_list, cv_rec_list, cv_f1_list = [], [], []

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[t_idx], X_train[v_idx]
        y_tr, y_val = y_train[t_idx], y_train[v_idx]

        rf_cv = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        rf_cv.fit(X_tr, y_tr)

        val_probs = rf_cv.predict_proba(X_val)[:, 1]
        val_preds = rf_cv.predict(X_val)

        cv_auc_list.append(roc_auc_score(y_val, val_probs))
        cv_rec_list.append(recall_score(y_val, val_preds, zero_division=0))
        cv_f1_list.append(f1_score(y_val, val_preds, zero_division=0))

    # -------------------------------------------------------------
    # STEP 2: Train全体での学習と OOB (Out-of-Bag) & Test 評価
    # -------------------------------------------------------------
    # oob_score=True に設定することでOOB予測確率を取得可能にする
    rf_full = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        oob_score=True,
        n_jobs=-1,
    )
    rf_full.fit(X_train, y_train)

    # --- OOB 評価の計算 ---
    # oob_decision_function_ は [N_samples, 2] の確率配列
    oob_probs = rf_full.oob_decision_function_[:, 1]
    oob_preds = (oob_probs >= 0.5).astype(int)

    oob_auc = roc_auc_score(y_train, oob_probs)
    oob_rec = recall_score(y_train, oob_preds, zero_division=0)
    oob_f1 = f1_score(y_train, oob_preds, zero_division=0)

    # --- Test 評価の計算 ---
    test_probs = rf_full.predict_proba(X_test)[:, 1]
    test_preds = rf_full.predict(X_test)

    test_auc = roc_auc_score(y_test, test_probs)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)

    # -------------------------------------------------------------
    # STEP 3: 結果の集計と Excel 出力
    # -------------------------------------------------------------
    results = [{
        "Model": "1D-CNN + Random Forest",
        "Feature_Dim": X_train.shape[1],
        "ROC-AUC (CV)": round(np.mean(cv_auc_list), 4),
        "ROC-AUC (Test)": round(test_auc, 4),
        "ROC-AUC (OOB)": round(oob_auc, 4),
        "Sensitivity (CV)": round(np.mean(cv_rec_list), 4),
        "Sensitivity (Test)": round(test_rec, 4),
        "Sensitivity (OOB)": round(oob_rec, 4),
        "F1-Score (CV)": round(np.mean(cv_f1_list), 4),
        "F1-Score (Test)": round(test_f1, 4),
        "F1-Score (OOB)": round(oob_f1, 4),
    }]

    df_res = pd.DataFrame(results)
    df_res.to_excel(excel_output_path, index=False)

    # コンソール画面への結果表示
    print("=" * 60)
    print("📊 評価結果一覧 (Summary)")
    print("=" * 60)
    print(
        f" [CV]   ROC-AUC: {np.mean(cv_auc_list):.4f} | Sensitivity: {np.mean(cv_rec_list):.4f} | F1: {np.mean(cv_f1_list):.4f}")
    print(
        f" [Test] ROC-AUC: {test_auc:.4f} | Sensitivity: {test_rec:.4f} | F1:"
        f" {test_f1:.4f}"
    )
    print(
        f" [OOB]  ROC-AUC: {oob_auc:.4f} | Sensitivity: {oob_rec:.4f} | F1:"
        f" {oob_f1:.4f}"
    )
    print("=" * 60)
    print(f"\n✅ Excelファイルへの出力が完了しました: '{excel_output_path}'")


if __name__ == "__main__":
    main()
