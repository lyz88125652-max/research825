import os
import matplotlib.pyplot as plt
import pandas as pd


def analyze_unified_labels(unified_path="unified.xlsx"):
    """unified.xlsx を読み込み、活性（1）と不活性（0）の件数と割合を統計・集計し、

    コンソール出力および円グラフの保存を行う関数
    """
    # 1. ファイルの存在確認
    if not os.path.exists(unified_path):
        print(f"❌ エラー: 指定されたファイル '{unified_path}' が見つかりません。")
        return

    # 2. データの読み込み
    df = pd.read_excel(unified_path)

    # 列名のゆらぎに対応（大文字小文字を無視して 'LABEL' を探す）
    label_col = next((c for c in df.columns if c.upper() == "LABEL"), None)
    if not label_col:
        print(
            f"❌ エラー: '{unified_path}' 内に 'LABEL' 列が見つかりません。"
            f"現在のカラム一覧: {df.columns.tolist()}"
        )
        return

    # 3. 統計データの算出
    total_samples = len(df)
    label_counts = df[label_col].value_counts()

    active_count = label_counts.get(1, 0)  # 活性 (1)
    inactive_count = label_counts.get(0, 0)  # 不活性 (0)

    active_ratio = (
        (active_count / total_samples) * 100 if total_samples > 0 else 0
    )
    inactive_ratio = (
        (inactive_count / total_samples) * 100 if total_samples > 0 else 0
    )

    # 4. コンソールへの詳細な統計出力
    print("=" * 50)
    print(f"📊 {unified_path} の活性・不活性 統計サマリー")
    print("=" * 50)
    print(f"・総サンプル数 (Total): {total_samples:,} 件")
    print(
        f"・活性データ (Active / Label=1):   {active_count:,} 件"
        f" ({active_ratio:.2f}%)"
    )
    print(
        f"・不活性データ (Inactive / Label=0): {inactive_count:,} 件"
        f" ({inactive_ratio:.2f}%)"
    )
    print("=" * 50)

    # 5. 割合を視覚化する円グラフの作成・保存
    plt.figure(figsize=(6, 6))
    labels = [
        f"Active (1)\n{active_count:,} ({active_ratio:.1f}%)",
        f"Inactive (0)\n{inactive_count:,} ({inactive_ratio:.1f}%)",
    ]
    sizes = [active_count, inactive_count]
    colors = ["#ff9999", "#66b3ff"]
    explode = (0.05, 0)  # 活性側を少し強調

    plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="",
        startangle=140,
        textprops={"fontsize": 12},
    )
    plt.title(
        f"Class Distribution in {unified_path}\n(Total: {total_samples:,} samples)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_img = "unified_label_distribution.png"
    plt.savefig(output_img, dpi=200)
    plt.close()

    print(f"📈 統計グラフを保存しました: '{output_img}'")


if __name__ == "__main__":
    analyze_unified_labels("unified.xlsx")
