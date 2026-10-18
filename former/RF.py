import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# --- [追加] F1-Scoreを最大化する最適閾値を探索する関数 ---
def find_best_threshold(y_true, probs):
    best_thresh = 0.5
    best_f1 = 0.0
    # 0.05 から 0.50 まで 0.01 刻みで探索
    for t in np.arange(0.05, 0.50, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def main():
    print(
        "=== 特徴量組み合わせ × Random Forest 評価 (閾値最適化 & 構造改善版) ==="
    )

    train_path = input(
        "学習用特徴量Excelファイル名 (例: train_features.xlsx): "
    ).strip()
    test_path = input(
        "テスト用特徴量Excelファイル名 (例: test_features.xlsx): "
    ).strip()

    output_dir = "rf_results"
    os.makedirs(output_dir, exist_ok=True)
    default_output_name = "rf_summary_oob.xlsx"
    output_name = input(
        f"出力ファイル名 (空欄で {default_output_name}): "
    ).strip()
    if not output_name:
        output_name = default_output_name
    elif not output_name.lower().endswith(".xlsx"):
        output_name += ".xlsx"
    excel_output_path = os.path.join(output_dir, output_name)

    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    y_train_all = df_train["LABEL"].values.astype(int)
    X_train_all = df_train.drop(columns=["SEQUENCE", "LABEL"])

    y_test_all = df_test["LABEL"].values.astype(int)
    X_test_all = df_test.drop(columns=["SEQUENCE", "LABEL"])

    method_prefixes = ["AAC_", "DPC_", "CTDC_", "CTDT_", "MB_"]
    method_columns = {
        p.rstrip("_"): [c for c in X_train_all.columns if c.startswith(p)]
        for p in method_prefixes
    }
    method_names = [m for m in method_columns if len(method_columns[m]) > 0]

    all_combinations = []
    for r in range(1, len(method_names) + 1):
        for comb in itertools.combinations(method_names, r):
            all_combinations.append(comb)

    results_list = []

    for idx, comb in enumerate(all_combinations, 1):
        comb_name = " + ".join(comb)
        selected_cols = []
        for m in comb:
            selected_cols.extend(method_columns[m])

        X_train_sub = X_train_all[selected_cols].values
        X_test_sub = X_test_all[selected_cols].values

        # -------------------------------------------------------------
        # STEP 1: 5-Fold 交差検証 (CV)
        # -------------------------------------------------------------
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        auc_scores, f1_scores, acc_scores, rec_scores, prec_scores = (
            [],
            [],
            [],
            [],
            [],
        )
        train_auc_scores, oob_auc_scores = [], []
        best_thresholds = []

        for t_idx, v_idx in skf.split(X_train_sub, y_train_all):
            X_tr, X_val = X_train_sub[t_idx], X_train_sub[v_idx]
            y_tr, y_val = y_train_all[t_idx], y_train_all[v_idx]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            # --- [変更 1] ハイパーパラメータの調整 (深さを5に制限, サンプル制限強化) ---
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=5,  # 10 -> 5 に変更して過学習を強力抑制
                min_samples_split=10,  # 5 -> 10 に変更
                min_samples_leaf=5,  # 2 -> 5 に変更
                max_features="sqrt",
                class_weight="balanced_subsample",  # 木ごとに重みを再計算
                oob_score=True,
                random_state=42,
                n_jobs=-1,
            )

            rf.fit(X_tr_scaled, y_tr)

            val_probs = rf.predict_proba(X_val_scaled)[:, 1]

            # --- [変更 2] Validation データ上で最適閾値を検索 ---
            opt_thresh, _ = find_best_threshold(y_val, val_probs)
            best_thresholds.append(opt_thresh)

            # 最適閾値を用いて二値判定
            val_preds = (val_probs >= opt_thresh).astype(int)

            tr_probs = rf.predict_proba(X_tr_scaled)[:, 1]
            oob_probs = rf.oob_decision_function_[:, 1]

            auc_scores.append(roc_auc_score(y_val, val_probs))
            train_auc_scores.append(roc_auc_score(y_tr, tr_probs))
            oob_auc_scores.append(roc_auc_score(y_tr, oob_probs))
            f1_scores.append(f1_score(y_val, val_preds, zero_division=0))
            acc_scores.append(accuracy_score(y_val, val_preds))
            rec_scores.append(recall_score(y_val, val_preds, zero_division=0))
            prec_scores.append(precision_score(
                y_val, val_preds, zero_division=0
            ))

        # CV全体の平均最適閾値を取得
        mean_opt_thresh = np.mean(best_thresholds)

        # -------------------------------------------------------------
        # STEP 2: Train全データで再学習して Test 評価 & OOB 算出
        # -------------------------------------------------------------
        scaler_full = StandardScaler()
        X_train_full_scaled = scaler_full.fit_transform(X_train_sub)
        X_test_full_scaled = scaler_full.transform(X_test_sub)

        rf_full = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,  # 10 -> 5
            min_samples_split=10,  # 5 -> 10
            min_samples_leaf=5,  # 2 -> 5
            max_features="sqrt",
            class_weight="balanced_subsample",
            oob_score=True,
            random_state=42,
            n_jobs=-1,
        )
        rf_full.fit(X_train_full_scaled, y_train_all)

        full_oob_probs = rf_full.oob_decision_function_[:, 1]
        full_oob_auc = roc_auc_score(y_train_all, full_oob_probs)

        test_probs = rf_full.predict_proba(X_test_full_scaled)[:, 1]

        # --- [変更 3] テストデータ判定にも CV で決定した平均最適閾値を適用 ---
        test_preds = (test_probs >= mean_opt_thresh).astype(int)

        test_auc = roc_auc_score(y_test_all, test_probs)
        test_f1 = f1_score(y_test_all, test_preds, zero_division=0)
        test_rec = recall_score(y_test_all, test_preds, zero_division=0)
        test_prec = precision_score(y_test_all, test_preds, zero_division=0)
        test_acc = accuracy_score(y_test_all, test_preds)

        mean_auc = np.mean(auc_scores)
        mean_tr_auc = np.mean(train_auc_scores)
        mean_oob_auc = np.mean(oob_auc_scores)

        train_oob_diff = mean_tr_auc - mean_oob_auc

        results_list.append({
            "パターンNo": idx,
            "手法数": len(comb),
            "手法の組み合わせ": comb_name,
            "次元数": len(selected_cols),
            "Opt_Threshold": round(mean_opt_thresh, 4),  # 最適閾値を記録
            "ROC-AUC (CV)": round(mean_auc, 4),
            "OOB ROC-AUC (CV Avg)": round(mean_oob_auc, 4),
            "Full Train OOB ROC-AUC": round(full_oob_auc, 4),
            "F1-Score (CV)": round(np.mean(f1_scores), 4),
            "Sensitivity (CV)": round(np.mean(rec_scores), 4),
            "Precision (CV)": round(np.mean(prec_scores), 4),
            "Accuracy (CV)": round(np.mean(acc_scores), 4),
            "Train AUC": round(mean_tr_auc, 4),
            "AUC_Diff (Train - OOB)": round(train_oob_diff, 4),
            "ROC-AUC (Test)": round(test_auc, 4),
            "F1-Score (Test)": round(test_f1, 4),
            "Sensitivity (Test)": round(test_rec, 4),
            "Precision (Test)": round(test_prec, 4),
            "Accuracy (Test)": round(test_acc, 4),
        })

        print(
            f" [{idx:02d}/31] {comb_name:<35} | Thresh: {mean_opt_thresh:.2f} | CV"
            f" Sens: {np.mean(rec_scores):.4f} | Test Sens: {test_rec:.4f}"
        )

    # -------------------------------------------------------------
    # STEP 3: OOB考慮型の Composite Score による並び替え
    # -------------------------------------------------------------
    res_df = pd.DataFrame(results_list)

    auc_min, auc_max = (
        res_df["ROC-AUC (CV)"].min(),
        res_df["ROC-AUC (CV)"].max(),
    )
    oob_min, oob_max = (
        res_df["Full Train OOB ROC-AUC"].min(),
        res_df["Full Train OOB ROC-AUC"].max(),
    )
    diff_min, diff_max = (
        res_df["AUC_Diff (Train - OOB)"].min(),
        res_df["AUC_Diff (Train - OOB)"].max(),
    )

    auc_norm = (res_df["ROC-AUC (CV)"] - auc_min) / (auc_max - auc_min + 1e-8)
    oob_norm = (res_df["Full Train OOB ROC-AUC"] - oob_min) / (
        oob_max - oob_min + 1e-8
    )
    diff_norm = (res_df["AUC_Diff (Train - OOB)"] - diff_min) / (
        diff_max - diff_min + 1e-8
    )

    res_df["Composite_Score"] = round(
        0.4 * auc_norm + 0.4 * oob_norm - 0.2 * diff_norm, 4
    )
    res_df = res_df.sort_values(by="Composite_Score", ascending=False)
    res_df.to_excel(excel_output_path, index=False)

    print(f"\n✅ OOBスコア・最適閾値付き集計完了: '{excel_output_path}'")


if __name__ == "__main__":
    main()
