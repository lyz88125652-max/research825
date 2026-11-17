# research825 ペプチド活性予測

抗酸化・抗菌・毒性のmulti-label予測を開発しています。旧multi-class実験はmulti_classフォルダに保管しています。

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

## Multi-class 実験の保管

旧multi-class専用のコード・結果・モデル・テスト・Wordログ・学習用Excelは [multi_class/](multi_class/README.md) に集約しました。

```bash
python multi_class/main_multi_class.py
python -m unittest discover -s multi_class/tests_multi_class -v
```

元の活性別CSV・FASTA、共通の `ESM-2.py`、multi-label側でも再利用する旧特徴量キャッシュ `data/multi_class/training_cv_esm2_650M.*` は現在の場所に残しています。

移動先の対応は [MULTI_CLASS_PATHS.md](multi_class/MULTI_CLASS_PATHS.md) を参照してください。過去のWordログ・モデル・メタデータに記録された実験当時のパスは、そのまま保持しています。

## 今後のmulti-label用に共有するデータ

`data/` 直下には元の活性別CSV・FASTAを残しています。これらはmulti-labelデータの作成にも使用できます。

`git.py` はプロジェクト全体のGit操作用です。
