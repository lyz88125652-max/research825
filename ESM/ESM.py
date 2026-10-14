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
    """入力Excelの1列目(配列)と2列目(ラベル)からESM-2 (650M) 1280次元Embeddingを抽出し保存する."""
    print(f"\n=== ESM-2 特徴量抽出開始 ===")
    print(f"入力ファイル: {input_xlsx}")
    print(f"使用モデル  : {model_name}")
    print(f"出力ファイル: {output_xlsx}")

    # 1. データ読み込み（1列目を配列、2列目をラベルとして取得）
    if not os.path.exists(input_xlsx):
        raise FileNotFoundError(f"ファイルが見つかりません: {input_xlsx}")

    df_raw = pd.read_excel(input_xlsx)

    if df_raw.shape[1] < 2:
        raise ValueError(
            "入力Excelには少なくとも2つの列（1列目: 配列, 2列目: ラベル）が必要です。"
        )

    # 列名に依存せず、位置（0列目＝配列、1列目＝ラベル）で確実に取得
    sequences = df_raw.iloc[:, 0].astype(str).str.strip().tolist()
    labels = df_raw.iloc[:, 1].values

    print(f"データ件数  : {len(sequences)} 件")
    print(f"配列サンプル: {sequences[0][:15]}...")
    print(f"ラベル分布  : 1={np.sum(labels == 1)}, 0={np.sum(labels == 0)}")

    # 2. モデルとトークナイザーのロード
    print("\nモデルとトークナイザーをロード中 (初回はダウンロードが発生します)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    model.to(device)

    # 3. バッチ処理による特徴抽出 (Mean Pooling)
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

            # パディングを除外した正確な Mean Pooling 計算
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(
                outputs.last_hidden_state * attention_mask, dim=1
            )
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings_list.append(mean_embeddings)

            if end_idx % 100 == 0 or end_idx == total_seqs:
                print(f"  進捗: {end_idx}/{total_seqs} 完了")

    # 4. 出力用DataFrameの構築
    all_embeddings = np.vstack(embeddings_list)
    dim = all_embeddings.shape[1]  # 1280次元

    esm_cols = [f"ESM_{i+1}" for i in range(dim)]
    df_esm = pd.DataFrame(all_embeddings, columns=esm_cols)

    # 1列目を SEQUENCE、2列目を LABEL と命名を統一して先頭に配置
    df_meta = pd.DataFrame({"SEQUENCE": sequences, "LABEL": labels})
    result_df = pd.concat([df_meta, df_esm], axis=1)

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
        "学習用Excelファイル名を入力してください (例: train_data.xlsx): "
    ).strip()
    if not train_input:
        train_input = "train_data.xlsx"

    extract_and_save_esm2_features(
        input_xlsx=train_input,
        output_xlsx="trainESMft.xlsx",  # デフォルト名
        model_name="facebook/esm2_t33_650M_UR50D",
        batch_size=16,
    )

    # --- 2. テストデータ（testESMft.xlsx）の作成 ---
    test_input = input(
        "テスト用Excelファイル名を入力してください (例: test_data.xlsx): "
    ).strip()
    if not test_input:
        test_input = "test_data.xlsx"

    extract_and_save_esm2_features(
        input_xlsx=test_input,
        output_xlsx="testESMft.xlsx",  # デフォルト名
        model_name="facebook/esm2_t33_650M_UR50D",
        batch_size=16,
    )
