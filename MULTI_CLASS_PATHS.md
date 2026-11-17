# Multi-class 関連ファイルの名称変更

multi-label実装の追加に備え、既存のmulti-class実験を識別できるよう名称を整理しました。実験データ、特徴量、保存済みモデル、実験ログの内容は変更していません。

| 旧名称 | 新名称 |
|---|---|
| `main.py` | `main_multi_class.py` |
| `ml_pipeline/` | `multi_class_pipeline/` |
| `results/` | `results_multi_class/` |
| `tests/` | `tests_multi_class/` |
| `experiment_logs/` | `experiment_logs_multi_class/` |
| `requirements.txt` | `requirements_multi_class.txt` |
| 旧 `README.md` の詳しい実行説明 | `README_multi_class.md` |
| `data/training_cv.xlsx` | `data/multi_class/training_cv.xlsx` |
| `data/training_cv_esm2_650M.npz` | `data/multi_class/training_cv_esm2_650M.npz` |
| `data/training_cv_esm2_650M.skipped.csv` | `data/multi_class/training_cv_esm2_650M.skipped.csv` |
| `data/training_cv_esm2_650M.metadata.json` | `data/multi_class/training_cv_esm2_650M.metadata.json` |
| `data/duplicated.xlsx` | `data/multi_class/duplicated.xlsx` |

`ESM-2.py` は分類器から独立した特徴量抽出処理として共用します。現在の既定入力はmulti-labelの4列CSVです。multi-classの2列Excelも自動判定し、`main_multi_class.py` はmulti-class用の入力・出力を明示して呼び出します。

`data/` 直下の活性別CSV・元のFASTA、`git.py` は共通ファイルとして保持しています。

## 今後の実行

プロジェクト直下で実行します。

```bash
python main_multi_class.py
python -m unittest discover -s tests_multi_class -v
python -m multi_class_pipeline.reporting results_multi_class/run_20261113_093537_489106
```

`main_multi_class.py` の既定入力・特徴量・結果保存先、import、ESM-2呼び出し、テスト、README内の実行例は新名称に更新しています。

## 過去の実験記録に書かれたパス

Wordログ、`run_config.json`、特徴量のmetadataなどには、実験当時のファイル名や別マシンの絶対パスが記録されています。実験時の記録として元の内容を保持しています。

例えば、ログ内の `results/run_20261113_093537_489106` は、現在は `results_multi_class/run_20261113_093537_489106` です。ログの図は `experiment_logs_multi_class/figures/run_20261113_093537_489106/` にあります。
