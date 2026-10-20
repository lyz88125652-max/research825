import os
import pandas as pd


def check_sequence_overlap(file1, file2, file3):
    """3つのExcelファイル（SEQUENCE列を含む）の間で

    重複している配列がないかをチェックする関数
    """
    # ファイルの存在確認
    for f in [file1, file2, file3]:
        if not os.path.exists(f):
            print(f"❌ エラー: ファイル '{f}' が見つかりません。")
            return

    # 各ファイルの読み込みと配列の抽出
    def load_sequences(filepath):
        df = pd.read_excel(filepath)
        # カラム名が 'SEQUENCE' であることを前提（小文字等の揺れにも対応）
        seq_col = next(
            (c for c in df.columns if c.upper() == "SEQUENCE"), None)
        if not seq_col:
            raise ValueError(f"ファイル '{filepath}' に 'SEQUENCE' 列が見つかりません。")
        # 空白の除去や大文字化で正規化してセット（集合）で保持
        return set(df[seq_col].dropna().astype(str).str.strip())

    print("📂 ファイルを読み込んでいます...")
    set1 = load_sequences(file1)
    set2 = load_sequences(file2)
    set3 = load_sequences(file3)

    print(f"\n📊 各ファイルの配列数:")
    print(f"  - {file1}: {len(set1)} 件")
    print(f"  - {file2}: {len(set2)} 件")
    print(f"  - {file3}: {len(set3)} 件")

    # 2者間・3者間の共通集合（積集合）の計算
    inter_12 = set1.intersection(set2)
    inter_13 = set1.intersection(set3)
    inter_23 = set2.intersection(set3)
    inter_all = set1.intersection(set2).intersection(set3)

    print("\n" + "=" * 50)
    print("🔍 配列の重複チェック結果")
    print("=" * 50)
    print(f"・{file1} と {file2} の共通配列数: {len(inter_12)} 件")
    print(f"・{file1} と {file3} の共通配列数: {len(inter_13)} 件")
    print(f"・{file2} と {file3} の共通配列数: {len(inter_23)} 件")
    print(f"・3つのファイルすべてに共通する配列数: {len(inter_all)} 件")
    print("=" * 50)

    # もし重複があれば、詳細をファイルに出力またはコンソールに表示
    if len(inter_12) > 0 or len(inter_13) > 0 or len(inter_23) > 0:
        print(
            "\n⚠️ 注意: ファイル間で重複している配列が見つかりました。"
            "独立したテストデータとして評価する場合、訓練データとの重複に注意が必要です。"
        )
    else:
        print(
            "\n✅ すべてのファイル間で重複している配列はありません（完全に独立してい"
            "ます）。"
        )


if __name__ == "__main__":
    # ファイル名の指定（必要に応じて _cleaned.xlsx や元のファイル名に変更してください）
    f_train = "trainESM.xlsx"  # または training_cleaned.xlsx
    f_test1 = "test1ESM.xlsx"  # または test1_cleaned.xlsx
    f_test2 = "test2ESM.xlsx"  # または test2_cleaned.xlsx

    check_sequence_overlap(f_train, f_test1, f_test2)
