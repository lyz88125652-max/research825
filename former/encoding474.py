import os
import sys
import pandas as pd
import numpy as np

# --- 1. 定義部 ---
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
DPC_LIST = [a1 + a2 for a1 in AA_LIST for a2 in AA_LIST]

GROUP1 = {
    "hydrophobicity": set("RKEDQN"), "normwaalsvolume": set("GASTPDC"),
    "polarity": set("LIFWCMVY"), "polarizability": set("GASDT"),
    "charge": set("KR"), "secondarystruct": set("EALMKRH"), "solventaccess": set("ALFCGIVW")
}
GROUP2 = {
    "hydrophobicity": set("GASTPHY"), "normwaalsvolume": set("NVEQIL"),
    "polarity": set("PATGS"), "polarizability": set("CPNVEQIL"),
    "charge": set("ANCQGHILMFPSTWYV"), "secondarystruct": set("VIYCWFT"), "solventaccess": set("RKQEND")
}
GROUP3 = {
    "hydrophobicity": set("CLVIMFW"), "normwaalsvolume": set("MHKFRYW"),
    "polarity": set("HQRKNED"), "polarizability": set("KMHFRYW"),
    "charge": set("DE"), "secondarystruct": set("GNPSD"), "solventaccess": set("MSPTHY")
}
PROPERTIES = list(GROUP1.keys())

PROPERTIES_VALS = {
    'hydrophobicity': {'A': -0.40, 'R': -0.59, 'N': -0.92, 'D': -1.31, 'C': 0.17, 'Q': -0.91, 'E': -1.22, 'G': -0.68, 'H': -0.62, 'I': 1.25, 'L': 1.22, 'K': -0.67, 'M': 1.02, 'F': 1.92, 'P': -0.49, 'S': -0.55, 'T': -0.28, 'W': 0.50, 'Y': 1.67, 'V': 0.91},
    'volume': {'A': -0.36, 'R': 1.34, 'N': 0.18, 'D': -0.04, 'C': 0.04, 'Q': 0.60, 'E': 0.28, 'G': -0.83, 'H': 0.55, 'I': 0.81, 'L': 0.81, 'K': 1.10, 'M': 0.97, 'F': 1.39, 'P': -0.06, 'S': -0.46, 'T': -0.15, 'W': 2.11, 'Y': 1.63, 'V': 0.35},
    'polarity': {'A': -0.45, 'R': 1.54, 'N': 0.82, 'D': 1.18, 'C': -0.45, 'Q': 0.73, 'E': 1.09, 'G': -0.45, 'H': 0.46, 'I': -1.18, 'L': -1.18, 'K': 1.27, 'M': -0.73, 'F': -1.09, 'P': -0.18, 'S': -0.18, 'T': -0.09, 'W': -0.73, 'Y': -0.27, 'V': -0.91},
    'charge': {'A': 0.0, 'R': 1.0, 'N': 0.0, 'D': -1.0, 'C': 0.0, 'Q': 0.0, 'E': -1.0, 'G': 0.0, 'H': 0.5, 'I': 0.0, 'L': 0.0, 'K': 1.0, 'M': 0.0, 'F': 0.0, 'P': 0.0, 'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0}
}

# --- 2. 特徴量計算関数 ---


def calc_aac(seq):
    L = len(seq)
    return [seq.count(aa)/L if L > 0 else 0.0 for aa in AA_LIST]


def calc_dpc(seq):
    L = len(seq)
    if L < 2:
        return [0.0] * 400
    counts = {dp: 0 for dp in DPC_LIST}
    for i in range(L - 1):
        dp = seq[i:i+2]
        if dp in counts:
            counts[dp] += 1
    return [counts[dp] / (L - 1) for dp in DPC_LIST]


def calc_ctdc(seq):
    L = len(seq)
    res = []
    for prop in PROPERTIES:
        if L == 0:
            res.extend([0.0, 0.0, 0.0])
            continue
        res.extend([
            sum(1 for aa in seq if aa in GROUP1[prop]) / L,
            sum(1 for aa in seq if aa in GROUP2[prop]) / L,
            sum(1 for aa in seq if aa in GROUP3[prop]) / L
        ])
    return res


def calc_ctdt(seq):
    L = len(seq)
    res = []
    for prop in PROPERTIES:
        if L < 2:
            res.extend([0.0, 0.0, 0.0])
            continue
        g_seq = [1 if aa in GROUP1[prop] else (2 if aa in GROUP2[prop] else (
            3 if aa in GROUP3[prop] else 0)) for aa in seq]
        t12, t13, t23 = 0, 0, 0
        for i in range(L - 1):
            pair = tuple(sorted([g_seq[i], g_seq[i+1]]))
            if pair == (1, 2):
                t12 += 1
            elif pair == (1, 3):
                t13 += 1
            elif pair == (2, 3):
                t23 += 1
        res.extend([t12/(L-1), t13/(L-1), t23/(L-1)])
    return res


def calc_moreau_broto(seq, max_lag=3):
    L = len(seq)
    res = []
    for prop_name, prop_dict in PROPERTIES_VALS.items():
        vals = [prop_dict.get(aa, 0.0) for aa in seq]
        for lag in range(1, max_lag + 1):
            if L <= lag:
                res.append(0.0)
            else:
                res.append(sum(vals[i] * vals[i + lag]
                           for i in range(L - lag)) / (L - lag))
    return res


# カラム名リスト生成 (計 474 次元)
feature_cols = (
    [f"AAC_{aa}" for aa in AA_LIST] +
    [f"DPC_{dp}" for dp in DPC_LIST] +
    [f"CTDC_{prop}_G{g}" for prop in PROPERTIES for g in [1, 2, 3]] +
    [f"CTDT_{prop}_T{t}" for prop in PROPERTIES for t in ["12", "13", "23"]] +
    [f"MB_{prop}_lag{lag}" for prop in PROPERTIES_VALS.keys()
     for lag in range(1, 4)]
)

# --- 3. メイン処理 ---


def main():
    print("=== 特徴量抽出処理 (encoding.py) ===")
    while True:
        input_path = input("入力Excelファイル名 (例: train1.xlsx): ").strip()
        if not input_path:
            continue
        if not os.path.exists(input_path):
            print(f"❌ エラー: '{input_path}' が見つかりません。")
        else:
            break

    while True:
        output_path = input("出力Excelファイル名 (例: train_features.xlsx): ").strip()
        if not output_path:
            continue
        if not output_path.endswith('.xlsx'):
            output_path += '.xlsx'
        break

    try:
        print(f"\n📂 '{input_path}' を読み込み中...")
        df = pd.read_excel(input_path)
        required_cols = ['SEQUENCE', 'LABEL']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ エラー: 必要な列 {missing_cols} がありません。")
            return

        print("⚙️ 474次元の特徴量を計算中...")
        features = []
        for seq in df['SEQUENCE']:
            s = str(seq).strip().upper()
            row = calc_aac(s) + calc_dpc(s) + calc_ctdc(s) + \
                calc_ctdt(s) + calc_moreau_broto(s)
            features.append(row)

        feat_df = pd.DataFrame(features, columns=feature_cols)
        out_df = pd.concat([df[['SEQUENCE', 'LABEL']], feat_df], axis=1)

        out_df.to_excel(output_path, index=False)
        print(
            f"✅ 計算・保存完了: '{output_path}' (データ数: {out_df.shape[0]}行, カラム数: {out_df.shape[1]}列)")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
