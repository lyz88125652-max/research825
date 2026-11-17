# Multi-label ペプチド活性予測

配列ごとに抗酸化・抗菌・毒性の3つの確率を予測します。複数の活性を同時に1と判定でき、確率の合計を1にする制約はありません。

Python 3.10以上で、プロジェクト直下から実行します。

```bash
python -m pip install -r requirements_multi_label.txt
python main_multi_label.py
```

macOSでLightGBMの `libomp` 読み込みエラーが出る場合は、`brew install libomp` を実行してください。

## 入力とESM-2特徴量

入力は `data/multi-label.csv` です。列の順序は固定です。

```csv
sequence,antioxidant,antimicrobial,toxic
EL,1,1,0
FLKPLFNAALKLLP,1,1,1
```

1は元の活性データに記載あり、0は記載なし（陰性または未知）です。このベースラインは0を学習上の陰性として扱います。実験で陰性を確認したことを意味せず、評価値も元の注釈に対する予測性能として解釈します。欠損したセルを自動で0に置換しません。

同じ配列の複数活性を保持し、multi-class版のように重複ラベルを理由に除外しません。CSVは配列ごとに1件に統合済みである必要があります。重複があれば入力エラーにします。特殊表記や小文字は化学的な意味を推測して変換せず、除外一覧へ記録します。

現在の8,002件のうち、ESM-2が扱えない39件を除いた7,963件が対象です。複数活性を持つ747件も保持しています。対象データの陽性件数は抗酸化995件、抗菌3,306件、毒性4,411件です。

共通の `ESM-2.py` は次の入力に対応します。

| 形式 | 必須列 | 学習ラベルの形 |
|---|---|---|
| Multi-label CSVまたはExcel | `sequence,antioxidant,antimicrobial,toxic` | `(N, 3)` の0/1行列 |
| 既存multi-class CSVまたはExcel | `sequence,label` | `(N,)` のラベル1・2・3 |

ヘッダーから自動判定します。既定入力は `data/multi-label.csv`。入力のラベル形式だけが異なり、ESM-2 650Mの最終層の残基表現を平均するエンコーディングは共通です。BOS・EOS・paddingを平均から除外し、配列ごとに1,280次元を出力します。

```bash
# 新しいmulti-label入力の確認のみ
python ESM-2.py --validate-only

# 通常の特徴量抽出
python ESM-2.py

# 既存のmulti-class特徴量を配列で照合して再利用
python ESM-2.py --reuse-features data/multi_class/training_cv_esm2_650M.npz

# 従来のmulti-class入力も利用可能
python ESM-2.py --input multi_class/data/training_cv.xlsx
```

multi-label出力の既定保存先は `data/multi_label/multi_label_esm2_650M.npz` です。ラベルは常に現在の入力から作り直し、古いキャッシュの単一ラベルを流用しません。モデルID、層、平均方法が一致するキャッシュだけを再利用し、不足する配列だけを追加抽出します。

| NPZの配列 | 内容 |
|---|---|
| `X` | `(N, 1280)` のfloat32特徴量 |
| `y` | `(N, 3)` の0/1ラベル |
| `sequences` | 各行の元の配列文字列 |
| `source_rows` | 元CSVまたはExcelのデータ位置（ヘッダーを1とした番号） |
| `label_names` | `antioxidant, antimicrobial, toxic` の順序 |
| `task` | `multilabel` |

`excel_rows` も従来との互換性のため保存します。`.metadata.json` に抽出条件・再利用件数、`.skipped.csv` に除外した配列と元の3列ラベルを保存します。`np.load(path, allow_pickle=False)` で読み込めます。

## 学習と独立テスト

`main_multi_label.py` が以下を実行します。

1. 入力とラベル順序を確認し、使えない配列を記録します。
2. 各活性の陽性比率を考慮するiterative stratificationで約10%を独立テストに確保します。
3. 残りに共通の5分割CVを適用し、全モデルを比較します。
4. 各モデルを学習データ全体で学習し直し、独立テストを評価します。
5. CSV・JSON・Excelの集計表、評価図、学習済みモデルを保存します。

入力は配列ごとに1件なので、同一配列が学習・評価の両方へ分かれません。類似配列のクラスタ分割や別取得元での外部検証は行いません。seedは42です。現在のデータとseedでは学習7,176件、独立テスト787件になります。分割は活性ごとの比率を考慮するため、指定割合から件数が少し変わる場合があります。

ラベルの組み合わせを10進数のクラスに変換して層化しません。例えば `(1,1,1)` が2件しかなくても、それを除外して5分割を成立させることはしません。ただし、すべての組み合わせが各foldに存在する保証はありません。

## 各モデル

| モデル | Multi-labelでの学習方法 |
|---|---|
| CNN | 3出力logitと `BCEWithLogitsLoss`。推論は独立Sigmoid |
| Random Forest | 活性ごとに1つ、合計3つの二値分類器（binary relevance） |
| LightGBM | `objective=binary` の分類器を活性ごとに学習（binary relevance） |
| Logistic Regression `lr` | 標準化＋L2正則化、活性ごとに二値分類器。`class_weight=balanced` |
| MLP `mlp` | 全結合1280→256→64→3、GELU・Dropout 0.3、BCE・Sigmoid |
| Classifier Chain `chain` | 標準化＋LRを連鎖。既定順序は抗酸化→抗菌→毒性 |
| CatBoost `catboost` | ネイティブmulti-labelの `MultiLogloss`、深さ6、学習率0.05 |
| XGBoost `xgboost` | 活性ごとの `binary:logistic`、深さ6、学習率0.05、`tree_method=hist` |
| SVM `svm` | 標準化＋活性ごとのRBF SVC、`C=1`・`gamma=scale`、確率出力 |

既定では9モデルすべてを実行します。`--models` で一部だけ選択できます。今回の追加はコードの作成であり、実データの特徴量抽出・学習・性能比較は実行していません。

LR・SVMは各外側の学習データだけで標準化します。LR・Chainの正則化は `--linear-c`、SVMは `--svm-c` で指定します。SVMは `probability=True` により学習データ内で確率推定を行い、共通評価ではその確率に閾値を適用します。SVC自身の `predict()` と一致しない場合があります。RBF SVMの確率推定は学習時間が増えることがあります。

Chainは各リンク内に標準化とLRを置き、内部3分割CVで得た予測を次のリンクの学習に使います。推論では前のリンクの予測を順番に使います。正解ラベルを独立テストから渡しません。順序は既定で `0 1 2` に固定し、`--chain-order` で変更できます。順序を比較する場合は学習側のCVで選び、独立テストで選びません。観測された共起にはデータの収集・注釈の偏りも含まれます。

XGBoostは各活性の学習側の陰性数/陽性数から `scale_pos_weight` を計算します。CatBoostの今回の設定は重みなしのMultiLoglossです。全モデルで損失関数や不均衡対策が同一という意味ではありません。木の本数・反復数は共通の `--trees`（既定300）で指定します。

MLPはCNNと同じ、学習側の約20%によるエポック選択と全学習データでの再学習を使います。`--mlp-epochs`（既定30）と `--mlp-batch-size`（既定64）で指定します。保存した `model.pt` の `architecture` は `cnn` または `mlp` です。再読み込み時は対応するモジュールの `build_network()` を使います。

RFは `balanced_subsample`、LightGBMは `balanced`、CNNは学習データ内の陰性数/陽性数を使った `pos_weight` で件数の偏りに対応します。RF・LightGBMはこの実装ではラベル間の依存関係を直接モデル化しません。

CNNは平均済みESMベクトルの座標方向に畳み込みます。アミノ酸位置ごとのCNNではありません。標準化は各学習側だけで適合します。学習側のさらに約20%を内部検証としてエポック数を決め、選んだ回数で外側の学習データ全体を使って新しいモデルを学習します。検証lossは重みなし・評価モードのbinary log lossで、独立テストを早停や回数選択に使いません。

判定閾値は既定で各活性0.5。確率が閾値以上なら1とします。`000` も `111` も予測できます。閾値は独立テストに合わせて最適化しません。変更する場合も、学習側の内部検証だけで決定してください。

## 出力と評価指標

保存先は `results_multi_label/run_日時/` です。

| 出力 | 内容 |
|---|---|
| `data_audit.json` | 採用・除外件数、陽性件数、ラベル組み合わせの件数 |
| `skipped_sequences.csv` | 除外配列と元ラベル・理由 |
| `holdout_assignments.csv` | 学習・独立テストへの割り当て |
| `fold_assignments.csv` | 全モデル共通のCV分割 |
| `run_config.json` | ラベル順序、閾値、環境、入力と特徴量のハッシュ |
| `model_comparison_cv.csv` | CV指標の平均・標本標準偏差と有効fold数 |
| `model_comparison_test.csv` | 独立テストのモデル比較 |
| `summary.xlsx` | Summary、CV_folds、Label_metrics、Notes |
| 各モデルの `oof_predictions.csv` | CVでその配列を学習していないモデルによる予測 |
| 各モデルの `test_predictions.csv` | 独立テストの正解3列・予測3列・確率3列 |
| 各モデルの `oof_label_metrics.csv` と `test_label_metrics.csv` | 活性別の指標、陽性・陰性件数、TP/FN/FP/TN |
| 各モデルの `fold_*/model.*` と `final_model/model.*` | foldごと・最終学習のモデル |
| `test_roc.png` と `test_pr.png` | 活性別のROC・Precision–Recall曲線 |
| 各モデルの `test_confusion_matrices.png` | 活性ごとの2×2混同行列 |
| `model_comparison.png` | CVと独立テストの比較 |
| CNN・MLPの `history.csv` と `loss_curves.png` | 学習履歴と内部検証曲線 |

全体指標はSubset Accuracy（3ラベルすべての完全一致率）、ラベルごとの平均正解率、Hamming loss、Micro/Macro/Samples F1、Samples Jaccard、Macro ROC-AUC、Macro Average Precision、Macro Sensitivity・Specificity・Precision、Macro/Micro MCC、Binary Log Lossです。

**multi-classのAccuracyとSubset Accuracyは同じ定義ではありません。** ラベルごとの平均正解率は `1 - Hamming loss` です。Macro MCCは各活性の二値MCCを平均し、Micro MCCはすべての0/1判定をまとめて計算します。Multi-labelのMacro Balanced Accuracyは、活性ごとに感度と特異度を平均したものをさらに平均します。

ROC-AUCなどが計算できない評価集合では、0や1で埋めず `null`（Excelでは空白）にします。CV平均には有効fold数も付けます。`average_precision` はAPであり、PR曲線を台形積分した面積と区別します。

## 実行例

```bash
python main_multi_label.py --prepare-only
python main_multi_label.py --models rf lightgbm
python main_multi_label.py --models lr mlp chain catboost xgboost svm
python main_multi_label.py --models mlp --mlp-epochs 30 --device cpu
python main_multi_label.py --models chain --chain-order 0 1 2 --chain-folds 3
python main_multi_label.py --models cnn --device cpu --cnn-batch-size 32
python main_multi_label.py --folds 3 --trees 100
python main_multi_label.py --threshold 0.5
python main_multi_label.py --reextract --no-feature-reuse
python -m unittest discover -s tests_multi_label -v
```

通常実行では、multi-label特徴量がなければ既存のmulti-classキャッシュの利用を試みます。`--no-feature-reuse` はこの自動再利用を無効化します。特徴量が既にある場合、再抽出には `--reextract` も指定します。

今回の実験用としてESM-2を新規に動かす場合は、次を後日実行します（この変更作業では実行していません）。既存キャッシュを上書きするため、必要なら先に別名で保存してください。

```bash
python ESM-2.py --task multilabel --input data/multi-label.csv --output data/multi_label/multi_label_esm2_650M.npz
python main_multi_label.py
```

2026年11月16日の集計・構造・実行状況・計画は [実験ログ](experiment_logs_multi_label/2026-11-16_experiment_log.md) に記録しています。

## 参照先

- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [scikit-learn MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)
- [LightGBM LGBMClassifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
- [iterative-stratification](https://github.com/trent-b/iterative-stratification)
- [scikit-learn ClassifierChain](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html)
- [scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [CatBoost MultiLogloss](https://catboost.ai/docs/en/concepts/loss-functions-multilabel-classification)
- [XGBoost XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
