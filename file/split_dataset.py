from collections import defaultdict
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def get_kmers(seq, k=3):
    """配列から k-mer のリストを抽出"""
    return [seq[i: i + k] for i in range(len(seq) - k + 1)]


def calculate_sequence_identity(seq1, seq2):
    """k-mer Dice 係数に基づく配列類似度（Sequence Identity）の計算"""
    len1, len2 = len(seq1), len(seq2)
    min_len = min(len1, len2)
    max_len = max(len1, len2)

    # 長さ比率が 0.35 未満の場合は類似度 40% 以上になり得ないため即座に 0
    if min_len / max_len < 0.35:
        return 0.0

    kmers1 = get_kmers(seq1, k=3)
    kmers2 = get_kmers(seq2, k=3)

    if not kmers1 or not kmers2:
        return 0.0

    counts1 = defaultdict(int)
    for kmer in kmers1:
        counts1[kmer] += 1

    intersection = 0
    for kmer in kmers2:
        if counts1[kmer] > 0:
            intersection += 1
            counts1[kmer] -= 1

    dice_sim = (2.0 * intersection) / (len(kmers1) + len(kmers2))
    return dice_sim


def cluster_sequences(sequences, threshold=0.40):
    """類似度 40% 基準による CD-HIT 方式のクラスタリング"""
    print(">>> 配列のクラスタリング処理中 (閾値: 40%)...")
    sorted_indices = sorted(
        range(len(sequences)), key=lambda i: len(sequences[i]), reverse=True
    )

    cluster_ids = np.zeros(len(sequences), dtype=int)
    representatives = []

    for idx in sorted_indices:
        seq = sequences[idx]
        assigned_cluster = -1

        for c_id, rep_seq in enumerate(representatives):
            sim = calculate_sequence_identity(seq, rep_seq)
            if sim >= threshold:
                assigned_cluster = c_id
                break

        if assigned_cluster == -1:
            new_c_id = len(representatives)
            representatives.append(seq)
            cluster_ids[idx] = new_c_id
        else:
            cluster_ids[idx] = assigned_cluster

    print(
        f"  完了: 全 {len(sequences)} 件を {len(representatives)} 個のクラスタに集約しました。"
    )
    return cluster_ids, representatives


def compute_similarity_matrices(df_train, df_test):
    """分割された Train と Test 間の最大類似度マトリックスを算出"""
    print(">>> 分割結果に基づく詳細な類似度計算中...")

    train_seqs = df_train["SEQUENCE"].tolist()
    test_seqs = df_test["SEQUENCE"].tolist()

    max_sim_to_train = []
    for t_seq in test_seqs:
        max_sim = 0.0
        for tr_seq in train_seqs:
            sim = calculate_sequence_identity(t_seq, tr_seq)
            if sim > max_sim:
                max_sim = sim
            if max_sim >= 0.40:
                break
        max_sim_to_train.append(max_sim)

    df_test_sim_summary = pd.DataFrame({
        "Test_Index": df_test.index,
        "SEQUENCE": test_seqs,
        "LABEL": df_test["LABEL"].values,
        "Max_Similarity_to_Train": max_sim_to_train,
        "Pass_40pct_Check": [sim <= 0.40 for sim in max_sim_to_train],
    })

    train_clusters = set(df_train["CLUSTER_ID"])
    test_clusters = set(df_test["CLUSTER_ID"])

    df_cluster_info = pd.DataFrame({
        "Metric": [
            "Total Sequences",
            "Train Sequences",
            "Test Sequences",
            "Total Clusters Created",
            "Train Clusters Count",
            "Test Clusters Count",
            "Overlapping Clusters Count (Must be 0)",
            "Max Test-Train Identity Observed",
        ],
        "Value": [
            len(df_train) + len(df_test),
            len(df_train),
            len(df_test),
            len(train_clusters | test_clusters),
            len(train_clusters),
            len(test_clusters),
            len(train_clusters & test_clusters),
            max(max_sim_to_train) if max_sim_to_train else 0.0,
        ],
    })

    return df_test_sim_summary, df_cluster_info


def main():
    input_file = "unified.xlsx"
    output_test_file = "independent_test.xlsx"
    output_train_file = "train_cvtest.xlsx"
    output_sim_file = "similarity.xlsx"

    if not os.path.exists(input_file):
        print(f"エラー: 入力ファイル '{input_file}' が見つかりません。")
        return

    print(f"データ読み込み中: {input_file}")
    df = pd.read_excel(input_file)

    sequences = df["SEQUENCE"].tolist()
    labels = df["LABEL"].values

    # 1. 類似度 40% に基づくクラスタリング
    cluster_ids, _ = cluster_sequences(sequences, threshold=0.40)
    df["CLUSTER_ID"] = cluster_ids

    # 2. Stratified Group 分割 (同じクラスタが分散しないように抽出)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(
        df, y=labels, groups=df["CLUSTER_ID"]))

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # 3. 類似度テーブルの計算
    df_test_sim_summary, df_cluster_info = compute_similarity_matrices(
        df_train, df_test
    )

    # 余分なメタ列の除去
    df_train_save = df_train.drop(columns=["CLUSTER_ID"])
    df_test_save = df_test.drop(columns=["CLUSTER_ID"])

    # 4. 3つの Excel ファイルに出力
    df_train_save.to_excel(output_train_file, index=False)
    df_test_save.to_excel(output_test_file, index=False)

    with pd.ExcelWriter(output_sim_file, engine="openpyxl") as writer:
        df_cluster_info.to_excel(writer, sheet_name="Summary", index=False)
        df_test_sim_summary.to_excel(
            writer, sheet_name="Test_vs_Train_Similarity", index=False
        )

    print("\n" + "=" * 60)
    print("  データ分割 & 類似度算出 完了サマリー")
    print("=" * 60)
    print(f"1. {output_train_file} : {len(df_train_save)} 件 (Train/CV用)")
    print(f"2. {output_test_file}  : {len(df_test_save)} 件 (Independent Test用)")
    print(
        f"3. {output_sim_file}     : 類似度レポート (Summary &"
        " Test_vs_Train_Similarity)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
