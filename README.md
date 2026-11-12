# ESM-2 ペプチド分類

Python 3.10以上で、プロジェクト直下から実行します。

```bash
python -m pip install -r requirements.txt
python main.py
```

`main.py` が次の処理を呼び出します。

1. `data/training_cv.xlsx` を読み込みます。
2. 特徴量ファイルがなければ `ESM-2.py` で650Mモデルの特徴量を抽出します。
3. 同じ5分割でCNN・Random Forest・LightGBMを学習、評価します。
4. `results/run_日時/` にモデルごとの結果と比較表を保存します。

初回はHugging FaceからESM-2モデルをダウンロードします。既存の特徴量は、配列・ラベル・元の行番号を入力Excelと照合して再利用します。入力を変更した場合は `python main.py --reextract` で再抽出します。

## ファイル構成

```text
main.py                         実行の入口、設定、処理の呼び出し
ESM-2.py                        ESM-2 650Mの特徴量抽出
ml_pipeline/
    data.py                     データ確認、ラベル重複確認、CV分割
    cnn.py                      PyTorchの1D CNN
    random_forest.py            Random Forest
    lightgbm_model.py           LightGBM
    evaluation.py               共通の指標計算、予測・結果保存
requirements.txt                必要なライブラリ
```

## 入力と学習方法

ラベルは抗酸化活性=1、抗菌活性=2、毒性活性=3です。今回のプログラムは3クラス分類です。ラベルはそれぞれのファイルに記録された活性であり、他の活性がないことを意味しません。

ESM-2の最終層の残基表現を平均し、配列あたり1,280次元を使います。特殊表記36件と小文字表記3件は自動変換せず、抽出時に `.skipped.csv` に記録します。現在の入力では8,720件が特徴量抽出の対象です。

**CNNは平均済み特徴ベクトルの座標方向に畳み込むモデルです。** アミノ酸位置に沿ったCNNではありません。残基単位のモチーフを学習するCNNには、平均前のESM残基表現が別途必要です。

クラス数の偏りには各モデルのクラス重みを使います。CNNの標準化は各学習分割内だけで行います。CNNのエポック数は、学習側の配列グループをさらに分けた内部検証で決定し、その回数で外側の学習データ全体を使って学習し直します。外側の評価データはエポック数の決定に使いません。

同一配列をまとめた `StratifiedGroupKFold` を使い、全モデルで共通の分割を使います。厳密に同一の配列間の混入を防ぎますが、類似配列のクラスタ分割は行いません。乱数シードは42が既定です。GPUやライブラリの違いによって結果が完全一致しない場合があります。

## 複数のラベルを持つ配列

元データには同じ配列に複数のラベルが付いた例があります。既定ではその配列の全行を学習・評価から除外します。`duplicated.xlsx` に配列、元のラベル、元Excelの行番号、観測されたラベルの組み合わせを保存します。`label_conflicts.csv` にも該当行を保存します。同じラベル内だけの重複行は保持し、同一配列は同じfoldにまとめます。

複数ラベルの配列をすべて除外する場合は、次を実行します。

```bash
python main.py --conflicts drop
```

これは既定の設定です。元ファイルを変更せず、学習対象を絞ります。除外後の分類は、今回のデータでラベルが重複していない配列を対象とした評価です。他の活性が生物学的に存在しないことを保証するものではありません。比較のため元ラベルを保持したまま実行する場合は `--conflicts keep` を指定できます。複数活性を同時に予測するマルチラベル分類は、この3クラス分類とは別の学習課題になります。

## 出力

`results/run_日時/` 以下に保存します。

| ファイル | 内容 |
|---|---|
| `model_comparison.csv` | 各モデルのCV平均・標準偏差 |
| `run_config.json` | 設定、件数、依存ライブラリのバージョン、特徴量ハッシュ |
| `data_audit.json` | 重複・特殊表記による除外件数、学習対象件数 |
| `fold_assignments.csv` | 全モデル共通の分割 |
| `label_conflicts.csv` | 複数ラベルの配列 |
| `duplicated.xlsx` | 除外対象の配列・ラベル・元行番号・ラベルの組み合わせ |
| 各モデルの `fold_metrics.csv` | foldごとの評価指標 |
| 各モデルの `summary.json` | CV平均・標準偏差と全OOF予測の指標 |
| 各モデルの `predictions.csv` | 各行の未学習foldによる予測、3クラスの確率 |
| 各モデルの `confusion_matrix.csv` | 混同行列（行=正解、列=予測） |
| 各モデルの `classification_report.json` | クラス別precision・recall・F1 |
| 各モデルの `fold_*/model.*` | foldごとの学習済みモデル |
| CNNの `fold_*/history.csv` | 内部検証と再学習の学習履歴 |

評価指標はAccuracy、Balanced Accuracy、Macro Precision/Recall/F1、Weighted F1、MCC、Log Loss、Macro ROC-AUC（one-vs-rest）です。標準偏差はfold間の標本標準偏差であり、信頼区間ではありません。最終モデルの選定やハイパーパラメータ探索は行いません。保存モデルはfoldごとのモデルで、全データでの最終学習モデルは別途作成する必要があります。

## 実行例

```bash
# Random Forestだけ
python main.py --models rf

# Random ForestとLightGBM
python main.py --models rf lightgbm

# CPUでCNN。小さいバッチで実行
python main.py --models cnn --device cpu --cnn-batch-size 32

# 評価fold数と木の本数の変更
python main.py --folds 3 --trees 100

# ESM抽出時にGPUメモリが不足する場合
python main.py --esm-batch-size 1

# 重複一覧と対象件数だけを確認（モデルは読み込まない）
python main.py --prepare-only
```

macOSでLightGBMの `libomp` 読み込みエラーが出た場合は `brew install libomp` を実行してください。

現在のデータでは複数ラベルの747配列（1,496行）を除外します。さらにESM-2で扱わない39行を除くと、学習・評価対象は7,224件（抗酸化998件、抗菌2,559件、毒性3,667件）になります。入力を更新した場合は、件数を毎回計算し直します。確認用として、現在の入力に基づく一覧を `data/duplicated.xlsx` に作成しています。通常実行時には、その回の一覧を結果フォルダにも出力します。

## 動作確認

```bash
python -m unittest discover -s tests -v
```

合成特徴量を用いた3モデルの学習・予測・保存モデルの再読み込み、配列グループの分離、複数ラベルの除外とExcel出力、キャッシュの整合性検査をテストします。合成データの評価値は実データの性能ではありません。

## APIの参照先

- [ESM-2公式実装](https://github.com/facebookresearch/esm)
- [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [LightGBM LGBMClassifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
- [PyTorch Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
