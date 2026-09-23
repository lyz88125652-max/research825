import pandas as pd


def verify_train_data(raw_excel_path, clean_excel_path):
    """train.xlsx (FASTA風形式) と train1.xlsx の一致チェック"""
    raw_df = pd.read_excel(raw_excel_path)
    clean_df = pd.read_excel(clean_excel_path)

    # 原データの解析・抽出
    col = raw_df.columns[0]
    lines = [col] + raw_df[col].dropna().tolist()

    raw_extracted = []
    curr_seq = []
    curr_label = None

    for line in lines:
        s = str(line).strip()
        if s.startswith(">"):
            if curr_seq and curr_label is not None:
                raw_extracted.append(("".join(curr_seq), curr_label))
                curr_seq = []
            curr_label = (
                0 if ("nonanti" in s.lower() or "|0|" in s or "|-1|" in s) else 1
            )
        else:
            curr_seq.append(s)
    if curr_seq and curr_label is not None:
        raw_extracted.append(("".join(curr_seq), curr_label))

    # 1. データ件数の検証
    if len(raw_extracted) != len(clean_df):
        print(
            f"❌ [Train] 件数不一致: 原データ={len(raw_extracted)}件, 変換後={len(clean_df)}件"
        )
        return False

    # 2. 配列とラベルの1件ずつの厳密検証
    mismatches = 0
    for idx, (raw_seq, raw_lbl) in enumerate(raw_extracted):
        clean_seq = str(clean_df.iloc[idx]["SEQUENCE"]).strip()
        clean_lbl = clean_df.iloc[idx]["LABEL"]

        if raw_seq != clean_seq or raw_lbl != clean_lbl:
            mismatches += 1
            print(
                f"  - [Train Error] 行 index {idx}: 原データ({raw_lbl}) != 変換後({clean_lbl})"
            )

    if mismatches == 0:
        print(
            f"✅ [Train] 検証成功: 全{len(clean_df)}件の配列とラベルが完全一致しています。"
        )
        return True
    else:
        print(f"❌ [Train] データ不一致検出: 計 {mismatches} 件")
        return False


def verify_test_data(raw_excel_path, clean_excel_path):
    """test.xlsx (表形式) と test1.xlsx の一致チェック"""
    raw_df = pd.read_excel(raw_excel_path)
    clean_df = pd.read_excel(clean_excel_path)

    label_map = {"antioxidant": 1, "non-antioxidant": 0}

    # 1. データ件数の検証
    if len(raw_df) != len(clean_df):
        print(
            f"❌ [Test] 件数不一致: 原データ={len(raw_df)}件, 変換後={len(clean_df)}件"
        )
        return False

    # 2. 配列とラベルの1件ずつの厳密検証
    mismatches = 0
    for idx in range(len(raw_df)):
        raw_seq = str(raw_df.iloc[idx]["SEQUENCE"]).strip()
        raw_lbl_str = str(raw_df.iloc[idx]["SEQCLASS"]).strip()
        expected_lbl = label_map.get(raw_lbl_str)

        clean_seq = str(clean_df.iloc[idx]["SEQUENCE"]).strip()
        clean_lbl = clean_df.iloc[idx]["LABEL"]

        if raw_seq != clean_seq or expected_lbl != clean_lbl:
            mismatches += 1
            print(
                f"  - [Test Error] 行 index {idx}: 原データ({raw_lbl_str}->{expected_lbl}) != 変換後({clean_lbl})"
            )

    if mismatches == 0:
        print(
            f"✅ [Test] 検証成功: 全{len(clean_df)}件の配列とラベルが完全一致しています。"
        )
        return True
    else:
        print(f"❌ [Test] データ不一致検出: 計 {mismatches} 件")
        return False


# --- 実行 ---
print("--- データ整合性検証実行 ---")
verify_train_data("train.xlsx", "train1.xlsx")
verify_test_data("test.xlsx", "test1.xlsx")
