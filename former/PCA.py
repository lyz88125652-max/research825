import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_pca_interactive():
    print("=== 主成分分析 (PCA) 実行スクリプト ===")

    # 1. 入力ファイル名の受付けとチェック
    while True:
        feature_excel_path = input(
            "解析対象の特徴量Excelファイル名（例: train_features.xlsx）を入力してください: ").strip()
        if not feature_excel_path:
            print("⚠️ ファイル名が入力されていません。もう一度入力してください。")
            continue

        if not os.path.exists(feature_excel_path):
            print(
                f"❌ エラー: 指定されたファイル '{feature_excel_path}' が見つかりません。パスを確認してください。\n")
        else:
            break

    # 2. 出力ファイル名の受付け
    while True:
        pca_output_path = input(
            "PCA結果の保存先Excelファイル名（例: train_pca.xlsx）を入力してください: ").strip()
        if not pca_output_path:
            print("⚠️ 保存先のファイル名を入力してください。")
            continue
        if not pca_output_path.endswith('.xlsx'):
            pca_output_path += '.xlsx'  # 拡張子の補正
        break

    # 3. 主成分数の受付け（デフォルト 10）
    n_components_input = input("算出する主成分数（デフォルト: 10 / 未入力時は10）: ").strip()
    if n_components_input == "":
        n_components = 10
    else:
        try:
            n_components = int(n_components_input)
            if n_components <= 0:
                print("⚠️ 1以上の整数を指定してください。デフォルトの 10 に設定します。")
                n_components = 10
        except ValueError:
            print("⚠️ 無効な入力です。デフォルトの 10 に設定します。")
            n_components = 10

    # 4. データ読み込みとエラーチェック
    try:
        print(f"\n📂 '{feature_excel_path}' を読み込んでいます...")
        df = pd.read_excel(feature_excel_path)

        # 必須列の存在確認
        required_cols = ['SEQUENCE', 'LABEL']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ エラー: ファイル内に必要な列 {missing_cols} が存在しません。")
            print(f"   現在の列名: {list(df.columns)}")
            return

        # 特徴量データの抽出（SEQUENCE, LABEL 以外の列）
        labels = df['LABEL']
        sequences = df['SEQUENCE']
        X = df.drop(columns=['SEQUENCE', 'LABEL'])

        if X.empty or X.shape[1] == 0:
            print("❌ エラー: 解析対象となる数値特徴量列が見つかりません。")
            return

        # 要求主成分数が特徴量数を超えていないか調整
        max_possible_components = min(X.shape[0], X.shape[1])
        if n_components > max_possible_components:
            print(
                f"⚠️ 指定された主成分数 ({n_components}) が最大可能数 ({max_possible_components}) を超えています。{max_possible_components} に変更します。")
            n_components = max_possible_components

    except Exception as e:
        print(f"❌ ファイル読み込み中にエラーが発生しました: {e}")
        return

    # 5. PCA解析の実行
    try:
        print("⚙️ 標準化および PCA を実行中...")

        # 標準化（Standardization: 平均0、分散1）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 主成分分析実行
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(X_scaled)

        # 寄与率の表示
        evr = pca.explained_variance_ratio_
        cum_evr = np.cumsum(evr)

        print("\n--- PCA 寄与率解析結果 ---")
        for i, (ratio, cum_ratio) in enumerate(zip(evr, cum_evr), 1):
            print(
                f"  第 {i} 主成分 (PC{i}): 個別寄与率 = {ratio*100:.2f}%, 累積寄与率 = {cum_ratio*100:.2f}%")

        # 6. 結果 Excel への出力
        pca_cols = [f"PC{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(pca_features, columns=pca_cols)
        result_df = pd.concat([sequences, labels, pca_df], axis=1)

        result_df.to_excel(pca_output_path, index=False)
        print(f"\n✅ PCA結果保存完了: '{pca_output_path}'")

        # 7. 散布図（PC1 vs PC2）の作成と保存
        print("📊 散布図グラフを作成しています...")
        plt.figure(figsize=(8, 6))
        for label, color, name in [(1, 'red', 'Antioxidant (1)'), (0, 'blue', 'Non-Antioxidant (0)')]:
            mask = (labels == label)
            plt.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                        c=color, label=name, alpha=0.6, s=25)

        plt.xlabel(f"PC1 ({evr[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({evr[1]*100:.1f}%)")
        plt.title("PCA Projection (PC1 vs PC2)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        scatter_img_name = 'pca_scatter_plot.png'
        plt.savefig(scatter_img_name, dpi=300)
        plt.show()
        print(f"✅ グラフ画像保存完了: '{scatter_img_name}'")

    except Exception as e:
        print(f"❌ PCA計算または描画中にエラーが発生しました: {e}")


# 実行
if __name__ == "__main__":
    run_pca_interactive()
