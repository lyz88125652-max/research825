import os
import pandas as pd


def remove_duplicate_sequences(train_path="train1.xlsx", test_path="test1.xlsx", output_path="test2.xlsx"):
    # ファイルの存在確認
    if not os.path.exists(train_path):
        print(f"❌ エラー: '{train_path}' が見つかりません。")
        return
    if not os.path.exists(test_path):
        print(f"❌ エラー: '{test_path}' が見つかりません。")
        return

    # データ読み込み
    print(f"📖 '{train_path}' と '{test_path}' を読み込んでいます...")
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    # 配列列のチェック（大文字・小文字を統一、前後の空白除去を行って比較）
    train_seqs = set(df_train['SEQUENCE'].dropna().str.strip().str.upper())

    # test1 内の各配列が train1 に存在するか確認
    is_duplicate = df_test['SEQUENCE'].astype(
        str).str.strip().str.upper().isin(train_seqs)

    duplicate_count = is_duplicate.sum()

    print("\n--- チェック結果 ---")
    print(f"学習データ (train1) の件数 : {len(df_train)} 件")
    print(f"テストデータ (test1) の件数: {len(df_test)} 件")
    print(f"重複している配列の数       : {duplicate_count} 件")

    if duplicate_count > 0:
        # 重複データを除去
        df_test_cleaned = df_test[~is_duplicate].copy()

        # 重複除去後の保存
        df_test_cleaned.to_excel(output_path, index=False)
        print(f"\n✅ 重複する {duplicate_count} 件を除去し、'{output_path}' に保存しました。")
        print(f"除去後のテストデータ件数: {len(df_test_cleaned)} 件")
    else:
        # 重複がない場合はそのままコピー保存
        df_test.to_excel(output_path, index=False)
        print(f"\n✅ 重複する配列はありませんでした。そのまま '{output_path}' として保存しました。")


if __name__ == "__main__":
    remove_duplicate_sequences()
