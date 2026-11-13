# ESM-2 ペプチド分類

Python 3.10以上で、プロジェクト直下から実行します。

```bash
python -m pip install -r requirements.txt
python main.py
```

`main.py` が次の処理を呼び出します。

1. `data/training_cv.xlsx` を読み込みます。
2. 特徴量ファイルがなければ `ESM-2.py` で650Mモデルの特徴量を抽出します。
3. ラベル重複・使用できない表記を除いたデータから、ラベル比率を保って約10%を独立testとして確保します。
4. 残り約90%を使い、共通の5分割でCNN・Random Forest・LightGBMを学習、評価します。
5. 各モデルを約90%の全体で再学習し、独立testを評価します。
6. `results/run_日時/` に既存形式の結果、`summary.xlsx`、CNNのloss図を保存します。

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

評価指標はAccuracy、Balanced Accuracy、Macro Precision/Recall/F1、Weighted F1、MCC、Log Loss、Macro ROC-AUC（one-vs-rest）です。標準偏差はfold間の標本標準偏差であり、信頼区間ではありません。モデルの選定やハイパーパラメータ探索は行いません。foldごとのモデルに加え、約90%のtraining全体で学習したモデルを各モデルの `final_model/` に保存します。

独立testは特徴量抽出・学習の前に、使用可能な入力配列から乱数seedで確保します。同一配列を一つのグループとし、元のクラス比率を保つ層化ランダム分割を使います。グループ単位のため行数は厳密な90%/10%とならない場合があります。`--conflicts keep` でグループ内に複数ラベルがある場合は層化グループ10分割の先頭をtestに使います。独立testはCV・標準化の学習・CNNのepoch選択には使いません。

`holdout_assignments.csv` に全使用可能行のtraining/test割当、`fold_assignments.csv` にtraining側だけのCV割当を保存します。`run_config.json` にtraining/test件数、クラス別件数、実際のtest比率を記録します。各モデルの `test_predictions.csv`、`test_metrics.json`、`test_confusion_matrix.csv`、`test_classification_report.json` が独立testの結果です。

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

## Excel集計とCNNのloss図

通常の `python main.py` の学習終了後、`ml_pipeline/reporting.py` が自動で以下を追加します。

- `results/run_日時/summary.xlsx`：モデル別のCV平均・標本標準偏差、training側OOF指標、独立test指標、fold別・クラス別の成績。
- `cnn/loss_curves.png`：各foldの内部学習・検証lossを並べた一覧。
- `cnn/fold_*/loss_curve.png`：内部学習・検証と、外側学習データ全体での再学習を分けた図。
- `cnn/final_model/loss_curve.png`：最終モデルの内部学習・検証と、training全体での再学習の図。

Excelの指標はROC-AUC、Accuracy、Sensitivity、Specificity、MCC、Precision、F1、Log Lossです。
Sensitivity・Specificityは各クラス対残り（one-vs-rest）で求め、全体表にはマクロ平均を表示します。
MCCは全体表では多クラス、クラス別表では二値です。定義は `Notes` シートに記載します。

`OOF_pooled` はtraining側の各評価foldの予測を結合した成績です。`Independent_test` が最初に確保した約10%の独立testの成績です。既存のCVのみの実験にレポートを追加しても独立testの成績は作れないため、独立testが必要な場合は `python main.py` で再学習してください。

既存結果にも再学習なしで追加できます。

```bash
python -m ml_pipeline.reporting results/run_20261112_154234_404164
```

CNNの横軸はEpoch、縦軸はLoss、破線は採用epochです。
今後の学習はepoch終了時の重みなし学習CEをevalモードで記録し、同条件の検証CEと比較します。
既存履歴の学習lossはクラス重み付きで、重みなしの検証lossとは条件が異なるため図に注記します。
再学習のlossは別パネルに表示します。学習lossの低下に対して検証lossが上昇し続けるかを確認できます。
`--prepare-only` はデータ監査のみのため、Excel集計と図の生成は行いません。

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
