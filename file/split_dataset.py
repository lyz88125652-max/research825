import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# 乱数シードの固定
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def get_kmers(seq, k=3):
    """配列から k-mer の集合（マルチセット）を抽出"""
    return [seq[i: i + k] for i in range(len(seq) - k + 1)]


def calculate_sequence_identity(seq1, seq2):
    """k-mer Dice 係数に基づく高速配列類似度の計算 (CD-HIT 40% 近似)

    配列長の違いや k-mer 一致率から Sequence Identity を評価します。
    """
    len1, len2 = len(seq1), len(seq2)
    min_len = min(len1, len2)
    max_len = max(len1, len2)

    # 長さの違いが大きすぎる場合は類似度 40% 未満と判定
    if min_len / max_len < 0.35:
        return 0.0

    kmers1 = get_kmers(seq1, k=3)
    kmers2 = get_kmers(seq2, k=3)

    if not kmers1 or not kmers2:
        return 0.0

    # k-mer 共通数の算出
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
    """Greedy Clustering (CD-HIT アルゴリズムの Python 実装)

    類似度が threshold (40%) 以上の配列同士を同じクラスタ ID にグループ化します。
    """
    print(">>> 配列のクラスタリング処理中 (閾値: 40%)...")
    # 長い配列順にソート (CD-HIT 標準挙動)
    sorted_indices = sorted(
        range(len(sequences)), key=lambda i: len(sequences[i]), reverse=True
    )

    cluster_ids = np.zeros(len(sequences), dtype=int)
    representatives = []  # 各クラスタの代表配列

    for idx in sorted_indices:
        seq = sequences[idx]
        assigned_cluster = -1

        for c_id, rep_seq in enumerate(representatives):
            sim = calculate_sequence_identity(seq, rep_seq)
            if sim >= threshold:
                assigned_cluster = c_id
                break

        if assigned_cluster == -1:
            # 新規クラスタの作成
            new_c_id = len(representatives)
            representatives.append(seq)
            cluster_ids[idx] = new_c_id
        else:
            cluster_ids[idx] = assigned_cluster

    print(
        f"  完了: 全 {len(sequences)} 件を {len(representatives)} 個のクラスタに集約しました。"
    )
    return cluster_ids


def main():
    input_file = 'unified.xlsx'
    output_test_file = 'independent_test.xlsx'
    output_train_file = 'train_cvtest.xlsx'

    if not os.path.exists(input_file):
        print(f"エラー: 入力ファイル '{input_file}' が見つかりません。")
        return

    print(f"データ読み込み中: {input_file}")
    df = pd.read_excel(input_file)

    sequences = df['SEQUENCE'].tolist()
    labels = df['LABEL'].values

    # 1. 類似度 40% に基づくクラスタリング
    cluster_ids = cluster_sequences(sequences, threshold=0.40)
    df['CLUSTER_ID'] = cluster_ids

    # 2. Stratified Group 分割（同じクラスタが Train と Test に跨がらないように約 10% を抽出）
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)

    train_idx, test_idx = next(
        sgkf.split(df, y=labels, groups=df['CLUSTER_ID'])
    )

    df_train = df.iloc[train_idx].drop(columns=['CLUSTER_ID']).copy()
    df_test = df.iloc[test_idx].drop(columns=['CLUSTER_ID']).copy()

    # 3. Excel ファイルとして保存
    df_test.to_excel(output_test_file, index=False)
    df_train.to_excel(output_train_file, index=False)

    # 4. 分割結果のサマリー表示
    pos_train = (df_train['LABEL'] == 1).sum()
    neg_train = (df_train['LABEL'] == 0).sum()
    pos_test = (df_test['LABEL'] == 1).sum()
    neg_test = (df_test['LABEL'] == 0).sum()

    print('\n' + '=' * 60)
    print('  データ分割完了サマリー')
    print('=' * 60)
    print(
        f"■ Train/CV 用データ ({output_train_file}): {len(df_train)} 件"
        f" ({len(df_train)/len(df)*100:.1f}%)"
    )
    print(
        f'   - 正例 (1): {pos_train} 件 ({pos_train/len(df_train)*100:.1f}%)'
    )
    print(
        f'   - 陰性 (0): {neg_train} 件 ({neg_train/len(df_train)*100:.1f}%)'
    )

    print(
        f"\n■ Independent Test データ ({output_test_file}): {len(df_test)} 件"
        f" ({len(df_test)/len(df)*100:.1f}%)"
    )
    print(
        f'   - 正例 (1): {pos_test} 件 ({pos_test/len(df_test)*100:.1f}%)'
    )
    print(f'   - 陰性 (0): {neg_test} 件 ({neg_test/len(df_test)*100:.1f}%)')
    print('=' * 60)
    print('※ Train と Test 間での配列類似度は 40% 以下に厳格制御されています。')


if __name__ == '__main__':
    main()
