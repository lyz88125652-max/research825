import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel


def extract_and_save_esm2_features(
    input_xlsx,
    output_xlsx="trainESMft.xlsx",
    model_name="facebook/esm2_t33_650M_UR50D",
    batch_size=16,
):
    """ESM-2 (650M) を用いてペプチド配列から1280次元のEmbeddingを抽出し、Excel形式で保存する関数."""
    print(f"\n=== ESM-2 特徴量抽出開始 ===")
    print(f"入力ファイル: {input_xlsx}")
    print(f"使用モデル  : {model_name}")
    print(f"出力ファイル: {output_xlsx}")

    # 1. データ読み込み
    if not os.path.exists(input_xlsx):
        raise FileNotFoundError(f"ファイルが見つかりません: {input_xlsx}")

    df = pd.read_excel(input_xlsx)
    if "SEQUENCE" not in df.columns or "LABEL" not in df.columns:
        raise ValueError("入力Excelに 'SEQUENCE' および 'LABEL' 列が必要です。")

    sequences = df["SEQUENCE"].astype(str).tolist()
    labels = df["LABEL"].values

    # 2. モデルとトークナイザーのロード
    print("モデルとトークナイザーをロード中 (時間がかかる場合があります)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    model.to(device)

    # 3. バッチ処理による特徴抽出
    embeddings_list = []
    total_seqs = len(sequences)

    print(f"合計 {total_seqs} 件の配列の特徴抽出を開始...")

    with torch.no_grad():
        for start_idx in range(0, total_seqs, batch_size):
            end_idx = min(start_idx + batch_size, total_seqs)
            batch_seqs = sequences[start_idx:end_idx]

            # トークン化（バッチ内でパディング）
            inputs = tokenizer(
                batch_seqs, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            outputs = model(**inputs)

            # 最終隠れ層からアミノ酸方向（Mean Pooling）で1本の固定長ベクトルを作成
            # outputs.last_hidden_state: (batch_size, seq_len, hidden_dim)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(
                outputs.last_hidden_state * attention_mask, dim=1
            )
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings_list.append(mean_embeddings)

            if end_idx % 100 == 0 or end_idx == total_seqs:
                print(f"  進捗: {end_idx}/{total_seqs} 完了")

    # 4. 特徴量DataFrameの作成
    all_embeddings = np.vstack(embeddings_list)
    dim = all_embeddings.shape[1]  # 1280 次元

    esm_cols = [f"ESM_{i+1}" for i in range(dim)]
    df_esm = pd.DataFrame(all_embeddings, columns=esm_cols)

    # 元の SEQUENCE, LABEL を先頭に配置
    result_df = pd.concat(
        [df[["SEQUENCE", "LABEL"]].reset_index(drop=True), df_esm], axis=1
    )

    # 5. Excelファイルに出力
    print(
        f"Excelファイルへ保存中 ({result_df.shape[0]} 行 x {result_df.shape[1]} 列)..."
    )
    result_df.to_excel(output_xlsx, index=False)
    print(f"✅ 保存完了: {output_xlsx}\n")

    return output_xlsx


if __name__ == "__main__":
    # --- 1. 訓練データ（trainESMft.xlsx）の作成 ---
    train_input = input(
        "学習用Excelファイル名を入力してください (例: train_features.xlsx): "
    ).strip()
    extract_and_save_esm2_features(
        input_xlsx=train_input,
        output_xlsx="trainESMft.xlsx",  # デフォルト名
        model_name="facebook/esm2_t33_650M_UR50D",
        batch_size=16,
    )

    # --- 2. テストデータ（testESMft.xlsx）の作成 ---
    test_input = input(
        "テスト用Excelファイル名を入力してください (例: test_features.xlsx): "
    ).strip()
    extract_and_save_esm2_features(
        input_xlsx=test_input,
        output_xlsx="testESMft.xlsx",
        model_name="facebook/esm2_t33_650M_UR50D",
        batch_size=16,
    )
