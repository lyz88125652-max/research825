import os
import sys
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA


def run_all_combinations_kmeans():
    print("=== 特徴量手法の全組み合わせ k-means 評価スクリプト ===")

    # 1. 入力ファイル名の受付けと存在チェック
    while True:
        input_path = input(
            "解析対象の特徴量Excelファイル名（例: train_features.xlsx）を入力してください: ").strip()
        if not input_path:
            print("⚠️ ファイル名を入力してください。")
            continue
        if not os.path.exists(input_path):
            print(f"❌ エラー: ファイル '{input_path}' が見つかりません。再入力してください。\n")
        else:
            break

    # 2. 結果出力用ディレクトリ（フォルダ）の受付け
    output_dir = input(
        "グラフ画像の保存先フォルダ名（例: kmeans_results）を入力してください [未入力時は kmeans_results]: ").strip()
    if not output_dir:
        output_dir = "kmeans_results"

    os.makedirs(output_dir, exist_ok=True)
    excel_output_path = os.path.join(output_dir, "kmeans_summary.xlsx")

    # 3. データ読み込みとチェック
    try:
        print(f"\n📂 '{input_path}' を読み込んでいます...")
        df = pd.read_excel(input_path)

        if 'SEQUENCE' not in df.columns or 'LABEL' not in df.columns:
            print("❌ エラー: ファイル内に 'SEQUENCE' または 'LABEL' 列が存在しません。")
            return

        labels = df['LABEL'].values
        X_all = df.drop(columns=['SEQUENCE', 'LABEL'])

    except Exception as e:
        print(f"❌ データ読み込み中にエラーが発生しました: {e}")
        return

    # 4. 手法（プレフィックス）ごとの列の自動グループ化
    # 列名の接頭辞 (AAC_, DPC_, CTDC_, CTDT_, MB_) に基づいて分類
    method_prefixes = ['AAC_', 'DPC_', 'CTDC_', 'CTDT_', 'MB_']
    method_columns = {}

    for prefix in method_prefixes:
        cols = [c for c in X_all.columns if c.startswith(prefix)]
        if cols:
            method_columns[prefix.rstrip('_')] = cols

    if not method_columns:
        print("❌ エラー: 対象の手法プレフィックス（AAC_, DPC_ 等）を持つ列が見つかりません。")
        return

    method_names = list(method_columns.keys())
    print(f"💡 検出された手法 ({len(method_names)}種類): {method_names}")

    # 5. 手法の全組み合わせパターンの生成 (1手法〜全手法)
    all_combinations = []
    for r in range(1, len(method_names) + 1):
        for comb in itertools.combinations(method_names, r):
            all_combinations.append(comb)

    print(f"⚙️ 合計 {len(all_combinations)} パターンの組み合わせで k-means (k=2) を実行します...\n")

    results_list = []

    # 6. 各組み合わせでの k-means 実行ループ
    for idx, comb in enumerate(all_combinations, 1):
        comb_name = " + ".join(comb)

        # 該当する手法の列を結合
        selected_cols = []
        for m in comb:
            selected_cols.extend(method_columns[m])

        X_sub = X_all[selected_cols].values

        # 標準化 (Standardization)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)

        # k-means 実行 (k=2)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # 評価指標の算出
        # 正解ラベルとの一致度 (-1 ~ 1, 1が完全一致)
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(
            labels, cluster_labels)  # 相互情報量 (0 ~ 1)
        # クラスタのまとまり具合 (-1 ~ 1)
        sil = silhouette_score(X_scaled, cluster_labels)

        results_list.append({
            "組み合わせ番号": idx,
            "手法数": len(comb),
            "手法の組み合わせ": comb_name,
            "次元数": len(selected_cols),
            "ARI (正解一致度)": round(ari, 4),
            "NMI (相互情報量)": round(nmi, 4),
            "Silhouette Score": round(sil, 4)
        })

        # 2次元（PCA）に投影してクラスタリング結果を画像保存
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # (左) k-means による予測クラスタ結果
        scatter1 = ax1.scatter(
            X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='coolwarm', alpha=0.6, s=20)
        ax1.set_title(f"k-means Cluster (k=2)\nARI: {ari:.4f}")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.grid(True, linestyle='--', alpha=0.5)

        # (右) 実際の正解ラベル (1:活性, 0:不活性)
        colors = ['blue' if l == 0 else 'red' for l in labels]
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=20)
        ax2.set_title("True Labels\n(Red: Active(1), Blue: Inactive(0))")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(
            f"Pattern {idx}: {comb_name} (Dim: {len(selected_cols)})", fontsize=12)
        plt.tight_layout()

        # 画像保存
        file_label = "_".join(comb)
        img_path = os.path.join(
            output_dir, f"pattern_{idx:02d}_{file_label}.png")
        plt.savefig(img_path, dpi=200)
        plt.close()

        print(
            f"  [{idx}/{len(all_combinations)}] {comb_name} -> ARI: {ari:.4f}, NMI: {nmi:.4f}")

    # 7. 全組み合わせの評価一覧を Excel 出力
    res_df = pd.DataFrame(results_list)
    res_df = res_df.sort_values(by="ARI (正解一致度)", ascending=False)  # 精度順にソート
    res_df.to_excel(excel_output_path, index=False)

    print(f"\n✅ 全処理完了！")
    print(f" 📊 一覧結果Excel: '{excel_output_path}'")
    print(f" 🖼️ 散布図グラフ画像: '{output_dir}/' フォルダ内に全パターン保存されました。")

    # 最高精度（ARI最高値）の表示
    best = res_df.iloc[0]
    print("\n--- 🏆 最も正解ラベルに近かった組み合わせ ---")
    print(f" 手法: {best['手法の組み合わせ']}")
    print(f" ARI : {best['ARI (正解一致度)']}")
    print(f" NMI : {best['NMI (相互情報量)']}")


# 実行
if __name__ == "__main__":
    run_all_combinations_kmeans()
