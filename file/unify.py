import pandas as pd


def process_train_to_simple(train_path):
    df_raw = pd.read_excel(train_path)
    col_name = df_raw.columns[0]
    lines = [col_name] + df_raw[col_name].dropna().tolist()

    data = []
    curr_seq = []
    curr_label = None

    for line in lines:
        s = str(line).strip()
        if s.startswith(">"):
            # 直前までに溜まった配列データをラベルと一緒に保存
            if curr_seq and curr_label is not None:
                data.append(
                    {"SEQUENCE": "".join(curr_seq), "LABEL": curr_label}
                )
                curr_seq = []

            # 新しいラベルの判定 (0: 非抗酸化, 1: 抗酸化)
            if "nonanti" in s.lower() or "|0|" in s or "|-1|" in s:
                curr_label = 0
            else:
                curr_label = 1
        else:
            curr_seq.append(s)

    # 最後の行の書き出し
    if curr_seq and curr_label is not None:
        data.append({"SEQUENCE": "".join(curr_seq), "LABEL": curr_label})

    return pd.DataFrame(data)


def process_test_to_simple(test_path):
    df_raw = pd.read_excel(test_path)

    # SEQCLASSの文字列 ('antioxidant' / 'non-antioxidant') を 1 / 0 に変換
    label_map = {"antioxidant": 1, "non-antioxidant": 0}

    df_simple = pd.DataFrame(
        {
            "SEQUENCE": df_raw["SEQUENCE"],
            "LABEL": df_raw["SEQCLASS"].map(label_map),
        }
    )
    return df_simple


# --- 実行処理 ---
train1_df = process_train_to_simple("train.xlsx")
test1_df = process_test_to_simple("test.xlsx")

# ファイル書き出し
train1_df.to_excel("train1.xlsx", index=False)
test1_df.to_excel("test1.xlsx", index=False)

print("✅ 2列フォーマットでの変換が完了しました:")
print(f" - train1.xlsx: {train1_df.shape} (内訳 -> 1: 249件, 0: 1531件)")
print(f" - test1.xlsx : {test1_df.shape} (内訳 -> 1: 74件, 0: 392件)")
