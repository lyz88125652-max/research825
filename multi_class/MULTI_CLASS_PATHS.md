# Multi-class 専用ファイルの保管先

2026年11月17日、今後のmulti-label実験と区別するため、multi-class専用ファイルをプロジェクト直下のmulti_classフォルダに集約しました。ファイルを削除せず、保存済みの実験結果・モデル・Wordログ・図・Excelの内容を維持しています。

| 移動前 | 現在の場所 |
|---|---|
| main_multi_class.py | multi_class/main_multi_class.py |
| multi_class_pipeline/ | multi_class/multi_class_pipeline/ |
| tests_multi_class/ | multi_class/tests_multi_class/ |
| results_multi_class/ | multi_class/results_multi_class/ |
| experiment_logs_multi_class/ | multi_class/experiment_logs_multi_class/ |
| requirements_multi_class.txt | multi_class/requirements_multi_class.txt |
| README_multi_class.md | multi_class/README.md |
| MULTI_CLASS_PATHS.md | multi_class/MULTI_CLASS_PATHS.md |
| data/multi_class/training_cv.xlsx | multi_class/data/training_cv.xlsx |
| data/multi_class/duplicated.xlsx | multi_class/data/duplicated.xlsx |

## 共通のため移動しないもの

- ESM-2.pyは両方式の共通の特徴量抽出コードとして、プロジェクト直下に残します。
- data直下の元の活性別CSV・FASTA、multi-label.csv、data/multi_labelは現在位置に残します。
- data/multi_class/training_cv_esm2_650M.npzと、そのmetadata.json・skipped.csvは、multi-label側の旧特徴量再利用にも使うため現在位置に残します。再利用元のパスも維持します。
- main_multi_label.pyとmulti_label_pipelineなど、multi-label側のコードは移動しません。

## 必要になった場合の実行

プロジェクト直下から実行します。共通のESM-2.pyと旧特徴量キャッシュは、プログラムの場所から自動解決します。

```bash
python -m pip install -r multi_class/requirements_multi_class.txt
python multi_class/main_multi_class.py
python -m unittest discover -s multi_class/tests_multi_class -v
PYTHONPATH=multi_class python -m multi_class_pipeline.reporting multi_class/results_multi_class/run_20261113_093537_489106
```

## 過去の記録内のパス

Wordログ、run_config.json、特徴量のmetadata、学習済みモデル内の実験当時のパスは書き換えていません。初期のmain.pyは現在multi_class/main_multi_class.py、ml_pipelineはmulti_class/multi_class_pipelineです。初期のresults/run_20261113_093537_489106は現在multi_class/results_multi_class/run_20261113_093537_489106にあります。図はmulti_class/experiment_logs_multi_class/figures/run_20261113_093537_489106に保存しています。
