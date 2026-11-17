# research825 ペプチド活性予測

既存の3クラス分類を、今後のmulti-label分類と区別できる名称に整理しています。

## Multi-label 実験

抗酸化・抗菌・毒性をそれぞれ0/1で予測します。複数活性を同時に予測でき、確率の合計は1に制約しません。

```bash
python -m pip install -r requirements_multi_label.txt
python main_multi_label.py
```

| 内容 | 保存先 |
|---|---|
| 実行の入口 | `main_multi_label.py` |
| 入力データ | `data/multi-label.csv` |
| 特徴量 | `data/multi_label/` |
| 学習・評価コード | `multi_label_pipeline/` |
| 結果 | `results_multi_label/` |
| テスト | `tests_multi_label/` |
| 詳しい使い方 | [README_multi_label.md](README_multi_label.md) |
| Word実験ログと集計 | `experiment_logs_multi_label/` |

CNN・RF・LightGBMに加え、LR・MLP・Classifier Chain・CatBoost・XGBoost・SVMのコードを用意しています。`--models` で選択し、指定しなければ9モデルが対象です。追加コードの作成時には実データの学習を実行していません。

`ESM-2.py` は両方式で共用します。既定入力はmulti-label CSVで、従来のmulti-class Excelもヘッダーから自動判定します。

## Multi-class 実験

1配列に対して抗酸化・抗菌・毒性のうち1つを予測します。3クラスの出力確率の合計は1です。

```bash
python -m pip install -r requirements_multi_class.txt
python main_multi_class.py
```

| 内容 | 保存先 |
|---|---|
| 実行の入口 | `main_multi_class.py` |
| ESM-2特徴量抽出 | `ESM-2.py` |
| 学習・評価のコード | `multi_class_pipeline/` |
| 学習用データと特徴量 | `data/multi_class/` |
| 実験結果と学習済みモデル | `results_multi_class/` |
| テスト | `tests_multi_class/` |
| Word実験ログと図 | `experiment_logs_multi_class/` |
| 詳しい使い方 | [README_multi_class.md](README_multi_class.md) |

テストは次のコマンドで実行します。

```bash
python -m unittest discover -s tests_multi_class -v
```

旧名称からの変更一覧と過去の記録内のパスについては、[MULTI_CLASS_PATHS.md](MULTI_CLASS_PATHS.md) を参照してください。

## 今後のmulti-label用に共有するデータ

`data/` 直下には元の活性別CSV・FASTAを残しています。これらはmulti-labelデータの作成にも使用できます。

`git.py` はプロジェクト全体のGit操作用です。
