import os
import pandas as pd


def unify_datasets(file1, file2, file3, output_path):
    """3つのExcelファイルを読み込んで結合し、

    SEQUENCEの重複行を削除した上で unified.xlsx として保存する関数
    """
    files = [file1, file2, file3]

    # 1. すべてのファイルの存在確認
    for f in files:
        if not os.path.exists(f):
            print(f"❌ エラー: ファイル '{f}' が見つかりません。")
            return

    dfs = []
    total_raw_rows = 0

    # 2. 各ファイルの読み込み
    for f in files:
        df = pd.read_excel(f)
        total_raw_rows += len(df)
        dfs.append(df)
        print(f"📂 読み込み成功: {f} ({len(df)} 行)")

    # 3. データフレームの結合（縦方向に連結）
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n🔗 結合後の総行数: {total_raw_rows} 行")

    # 4. SEQUENCE列を基準に重複行を削除（最初に出現した行を保持）
    # ※ 大文字・小文字や前後の空白の揺れを考慮して安全に重複判定するため、
    #    一時的な正規化列を使って重複チェックを行います
    if "SEQUENCE" in combined_df.columns:
        # 前後の空白を削除した正規化配列を作成して重複判定
        combined_df["_TEMP_SEQ_"] = (
            combined_df["SEQUENCE"].astype(str).str.strip()
        )
        cleaned_df = combined_df.drop_duplicates(
            subset=["_TEMP_SEQ_"], keep="first"
        ).copy()
        cleaned_df = cleaned_df.drop(columns=["_TEMP_SEQ_"])
    else:
        # 万が一列名が大文字小文字などで異なる場合のフォールバック
        seq_col = next(
            (c for c in combined_df.columns if c.upper() == "SEQUENCE"), None
        )
        if seq_col:
            cleaned_df = combined_df.drop_duplicates(
                subset=[seq_col], keep="first"
            ).copy()
        else:
            raise ValueError("❌ エラー: 'SEQUENCE' 列が見つかりません。")

    removed_count = total_raw_rows - len(cleaned_df)

    # 5. 結果の保存（列の内容・順序は変更しない）
    cleaned_df.to_excel(output_path, index=False)

    print("=" * 50)
    print("✨ 統合と重複削除が完了しました！")
    print("=" * 50)
    print(f"・削除された重複行数: {removed_count} 行")
    print(f"・最終的な unified.xlsx の行数: {len(cleaned_df)} 行")
    print(f"・保存先: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    # 対象のファイル名
    f_train = "trainESM.xlsx"
    f_test1 = "test1ESM.xlsx"
    f_test2 = "test2ESM.xlsx"
    output_file = "unified.xlsx"

    unify_datasets(f_train, f_test1, f_test2, output_file)
