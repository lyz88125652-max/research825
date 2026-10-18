import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler


def main():
    print(
        "=== Train / Test データの重複・類似度チェック (Data Leakage Verification) ==="
    )

    # 1. ファイルパスの入力
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

    # 保存先の指定
    output_dir = "leakage_check_results"
    os.makedirs(output_dir, exist_ok=True)

    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    print(f"\n📊 データサイズ:")
    print(f"  - Train: {len(df_train)} サンプル")
    print(f"  - Test : {len(df_test)} サンプル")

    # -------------------------------------------------------------
    # CHECK 1: SEQUENCE（アミノ酸配列）の完全一致重複チェック
    # -------------------------------------------------------------
    print("\n🔍 [CHECK 1] アミノ酸配列 (SEQUENCE) の重複チェック...")
    if "SEQUENCE" in df_train.columns and "SEQUENCE" in df_test.columns:
        train_seqs = set(df_train["SEQUENCE"].dropna().values)
        test_seqs = set(df_test["SEQUENCE"].dropna().values)

        exact_matches = train_seqs.intersection(test_seqs)

        print(f"  - Train 内のユニーク配列数: {len(train_seqs)}")
        print(f"  - Test 内のユニーク配列数 : {len(test_seqs)}")
        print(f"  - ⚠️ 完全重複配列数 (Data Leak): {len(exact_matches)}")

        if len(exact_matches) > 0:
            print(
                f"  ⚠️ 警告: {len(exact_matches)} 個の配列が Train と Test"
                " の両方に完全に重複して存在します！"
            )
            pd.DataFrame({"Duplicated_Sequence": list(exact_matches)}).to_csv(
                os.path.join(output_dir, "exact_duplicate_sequences.csv"),
                index=False,
            )
        else:
            print("  ✅ アミノ酸配列の完全重複はありません。")
    else:
        print(
            "  ℹ️ 'SEQUENCE' 列が存在しないため、配列一致チェックをスキップします。"
        )

    # -------------------------------------------------------------
    # CHECK 2: 特徴量ベクトルのコサイン類似度 (Cosine Similarity)
    # -------------------------------------------------------------
    print("\n🔍 [CHECK 2] 特徴量ベクトルのコサイン類似度解析...")

    # 数値特徴量のみ抽出
    feature_cols = [
        c for c in df_train.columns if c not in ["SEQUENCE", "LABEL"]
    ]
    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values

    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cosine Similarity 行列の計算 [Test数 × Train数]
    sim_matrix = cosine_similarity(X_test_scaled, X_train_scaled)

    # 各 Test サンプルに対する「最も類似している Train サンプル」の類似度
    max_sim_per_test = np.max(sim_matrix, axis=1)

    print(f"  - 平均 最大コサイン類似度 : {np.mean(max_sim_per_test):.4f}")
    print(f"  - 中央値 最大コサイン類似度: {np.median(max_sim_per_test):.4f}")
    print(f"  - 最大 コサイン類似度     : {np.max(max_sim_per_test):.4f}")

    # 類似度が非常に高いペア (例: > 0.95 または > 0.99) をカウント
    high_sim_95 = np.sum(max_sim_per_test >= 0.95)
    high_sim_99 = np.sum(max_sim_per_test >= 0.99)

    print(
        f"  - 類似度 0.95 以上の Test サンプル数:"
        f" {high_sim_95} / {len(df_test)} ({high_sim_95 / len(df_test)*100:.1f}%)"
    )
    print(
        f"  - 類似度 0.99 以上の Test サンプル数 (ほぼ同一):"
        f" {high_sim_99} / {len(df_test)} ({high_sim_99 / len(df_test)*100:.1f}%)"
    )

    # コサイン類似度の分布ヒストグラムを保存
    plt.figure(figsize=(7, 4.5))
    plt.hist(
        max_sim_per_test,
        bins=30,
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.8,
    )
    plt.axvline(
        np.mean(max_sim_per_test),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(max_sim_per_test):.3f}",
    )
    plt.xlabel("Max Cosine Similarity to any Train sample")
    plt.ylabel("Test Sample Count")
    plt.title("Distribution of Max Similarity (Test vs Train)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "test_to_train_similarity_distribution.png"),
        dpi=200,
    )
    plt.close()

    # -------------------------------------------------------------
    # CHECK 3: 高類似度ペアの詳細抽出
    # -------------------------------------------------------------
    high_sim_pairs = []
    for test_idx in range(len(df_test)):
        train_idx = np.argmax(sim_matrix[test_idx])
        sim_val = sim_matrix[test_idx, train_idx]

        if sim_val >= 0.90:  # 90%以上似ているものを保存
            item = {
                "Test_Index": test_idx,
                "Train_Index": train_idx,
                "Similarity": round(float(sim_val), 4),
                "Test_Label": df_test.iloc[test_idx]["LABEL"],
                "Train_Label": df_train.iloc[train_idx]["LABEL"],
            }
            if "SEQUENCE" in df_train.columns:
                item["Test_Sequence"] = df_test.iloc[test_idx]["SEQUENCE"]
                item["Train_Sequence"] = df_train.iloc[train_idx]["SEQUENCE"]
            high_sim_pairs.append(item)

    df_high_sim = pd.DataFrame(high_sim_pairs)
    if not df_high_sim.empty:
        df_high_sim = df_high_sim.sort_values(by="Similarity", ascending=False)
        high_sim_path = os.path.join(output_dir, "high_similarity_pairs.csv")
        df_high_sim.to_csv(high_sim_path, index=False)
        print(
            f"  📁 類似度0.90以上の類似ペア情報を '{high_sim_path}' に保存しました。"
        )

    print(f"\n✅ チェック処理が完了しました。")
    print(f"   画像・csvは '{output_dir}' フォルダに保存されています。")


if __name__ == "__main__":
    main()
