# Multi-label ペプチド活性予測

2026年11月17日の研究計画に対応し、単独9モデル、511通りの確率平均、Stacking、長さによる偏り、最終テストを分けて実行します。ESM-2のencodingと既存9モデルの構造は維持しています。本実装作業で実データの学習は実行していません。

## 入力と特徴量

入力は `data/multi-label.csv`。列順は `sequence,antioxidant,antimicrobial,toxic` です。1は活性の記録あり、0は未記録・未知または陰性で、確認済みの生物学的陰性とは限りません。同一配列を1件に統合し、複数活性を保持します。欠損ラベルと重複配列は入力エラーです。

統合済み8,002配列のうち、特殊表記・小文字を含む39配列を除き、抽出対象は7,963配列です。複数活性747配列を保持します。陽性件数は抗酸化995、抗菌3,306、毒性4,411です。

```bash
python -m pip install -r requirements_multi_label.txt
python ESM-2.py --task multilabel
```

ESM-2は `facebook/esm2_t33_650M_UR50D` の最終33層、残基平均、BOS・EOS・padding除外です。出力は `data/multi_label/multi_label_esm2_650M.npz`。キーはX（N×1280 float32）、y（N×3の0/1）、sequences、source_rows、label_names、task（multilabel）。互換用excel_rowsも保持します。

既存のNPZ・metadata・除外一覧は上書き前に `data/multi_label/backups/日時/` へコピーします。入力ハッシュ・日時・ライブラリ版・新規抽出か再利用かをmetadataに保存します。encoding自体は変更していません。`--validate-only` は推論せず、別の `.validation.skipped.csv` に除外一覧を書きます。

主実験は新規抽出が前提です。再利用件数があるmetadataでは学習開始時に停止します。探索目的で再利用する場合だけ、新しい実験の開始時に `--allow-reused-features` を指定してください。mainからの自動抽出は廃止しました。共通ESM-2.pyは旧multi-classの2列入力にも引き続き対応します。

## 実行手順

Python 3.10以上で、プロジェクト直下から実行します。まず分割と設定を固定します。

```bash
python main_multi_label.py --stage prepare
```

この段階では特徴量がなくても構いません。表示されたrunフォルダを指定して続けます。

```bash
python main_multi_label.py --stage cv --run-dir results_multi_label/run_...
python main_multi_label.py --stage ensemble --run-dir results_multi_label/run_...
python main_multi_label.py --stage nested --run-dir results_multi_label/run_...
python main_multi_label.py --stage bias --run-dir results_multi_label/run_...
python main_multi_label.py --stage select --run-dir results_multi_label/run_...
```

構成を確定した後にだけ、独立テストを評価します。

```bash
python main_multi_label.py --stage final --run-dir results_multi_label/run_...
```

| stage | 処理 |
|---|---|
| prepare | 分割・設定・入力ハッシュの固定。抽出・学習なし |
| cv | 単独モデルの共通CV・OOF保存。テスト評価なし |
| ensemble | OOFから全非空部分集合を確率平均しランキング化。再学習なし |
| nested | 内側OOFで選択・メタ学習、外側の未使用foldで比較 |
| bias | 長さのみのLR、共通長範囲でのOOF評価、モデル間の誤り一致率・相関 |
| select | 開発側の結果から最終構成と比較対象を保存 |
| final | 固定した構成と事前指定した比較対象の独立テスト評価 |
| all | cv → ensemble → nested → bias → select。finalを含まない |

引数なしの実行は `--stage cv` と同じです。以前の「CV後すぐ独立テストまで評価する」動作から変更しました。最初から開発段階をまとめる場合は `python main_multi_label.py --stage all` を使います。

まず3モデルで確認する場合の例です。

```bash
python main_multi_label.py --stage all --models cnn rf lightgbm --folds 3 --inner-folds 3 --top-k 3 --split-file data/multi_label/pilot_splits.json
```

3モデルは7通り、9モデルは単独9＋複数モデル502＝511通りです。既定は外側5分割・内側5分割・TOP10。内外CVは多くの再学習を伴うため、単独CVより時間がかかります。

## 分割と再現性

約10%を独立テストとして確保し、残り約90%で開発します。現在の入力では学習7,176件、テスト787件（約9.88%）。iterative stratificationで活性ごとの比率を考慮し、同一配列の漏洩を防ぎます。希少な組み合わせを削除して分割を成立させる処理はありません。

分割は既定で `data/multi_label/splits_seed42_folds5.json` に一度だけ作成します。配列・ラベル・元位置・seed・fold数・テスト割合を照合して再利用し、各runにもコピーします。入力や分割設定を変える場合は新しい `--split-file` を指定します。

設定は最初の実行で固定します。続きの実行では **`--run-dir` と `--stage` だけ**を指定し、モデルやハイパーパラメータはconfigから復元します。変更する場合は新規runにします。コード・入力・分割・特徴量・metadataのハッシュが変わった場合も同じrunの続行を止めます。元データのハッシュ、コミットID、ライブラリ版を記録します。

prepare時点の特徴量ハッシュは未固定です。cvの前に新規抽出し、初めて特徴量を使う段階でハッシュとmetadataを固定できます。

完了したfitの予測確率を、学習・予測データ、ラベル、seed、設定の指紋とともに `prediction_cache.npz` に保存します。同じstageの再実行では整合する予測キャッシュを再利用します。失敗は `failure.json` とstage状態に記録します。

学習済みモデル自体（`model.joblib`、`model.pt`、`model.cbm`、`label_*.ubj`、`*_stacker.joblib`）は出力しません。CV・nested CV・最終testすべてに適用します。モデルはメモリ上で学習・予測し、予測確率・指標・Excel・図・学習履歴は保存します。新しい配列への推論には再学習が必要です。既存結果のモデルファイルは自動削除しません。

## モデルとOOF

| 選択名 | 初期設定 |
|---|---|
| cnn | 特徴量方向のCNN、3 logit、BCEWithLogitsLoss＋Sigmoid |
| rf | 活性別RF、balanced_subsample、300木 |
| lightgbm | 活性別binary、balanced、300木、学習率0.05 |
| lr | 学習側で標準化、活性別L2 LR、C=1、balanced |
| mlp | 全結合1280→256→64→3、GELU、Dropout 0.3、BCE＋Sigmoid |
| chain | 標準化＋LRの連鎖、抗酸化→抗菌→毒性、内部3分割予測 |
| catboost | MultiLogloss、300反復、深さ6、学習率0.05、重みなし |
| xgboost | 活性別binary:logistic、300木、hist、学習率0.05、学習側で陽性重みを計算 |
| svm | 学習側で標準化、RBF、C=1、gamma=scale、balanced、確率推定 |

CNN・MLPは学習側の約20%でエポック数を選び、外側の学習部分全体で学習し直します。既定30エポック、patience 5、batch 64。CNNは平均ESMの座標方向を扱い、残基位置方向ではありません。

3つの活性の確率は合計1にそろえません。既定閾値は各活性0.5で、000や111も予測できます。SVMも推定確率に共通閾値を適用するため、SVC自身のpredictと異なる場合があります。

OOFは、自分の配列を学習していないfoldモデルの予測を元の順番に並べたものです。各モデルのCSVと、`development_predictions.npz` の `(学習件数,モデル数,3)` テンソルに保存します。配列・ラベル・モデル・元位置・foldの対応を検査します。

## アンサンブルとStacking

確率平均は活性ごとにモデルを等しく平均します。主順位はOOF Macro-F1、同点時はMacro AP、少ないモデル数、候補名の順です。TOP10に入らなくてもベスト単独と全モデル平均を残します。511通りを選ぶために使ったOOF成績は探索値であり、そのまま汎化性能とはしません。

外側CVの各回では、外側の学習部分だけで内側OOFを作成します。そこで候補を選び、各ラベルkモデル分の確率を入力として3本のメタLRを学習します。ベースモデルを外側の学習部分全体で再学習し、外側の未使用foldで確率平均とStackingを比較します。同じ候補・同じfoldで差を保存します。

メタLRは既定C=1、class_weightなし。`--meta-c` と `--meta-class-weight` で最初に固定します。現在は閾値・正則化・確率校正の自動探索をしません。他活性の確率を入力に追加する拡張もありません。

`top_01` 等は候補選抜手順の順位です。外側foldごとに具体的なモデル構成が変わり得ます。各foldの構成と内外の学習・検証元位置を保存します。全体OOFに単純なCVを重ねてStackingを評価する実装ではありません。

selectは外側CV平均Macro-F1、平均Macro AP、平均モデル数、roleとmethodの順で最終手順を選びます。学習90%全体のランキングで具体的な構成を固定します。外側CVで手順も選ぶため、最終確認には未使用テストが必要です。

## 最終テスト

`final_recipe.json` に構成・閾値・メタLR設定と選択用ファイルのハッシュを固定します。finalは選択構成、ベスト単独、全モデル平均を一括評価します。TOP10の確率平均とStackingの公平な比較は外側CVの表を使います。

最終LRは学習90%全体のOOF、ベースモデルは90%全体で学習します。その後にテストを予測します。テスト結果で再選択しません。`final_access.json` で開始を記録し、同じrunのfinal再実行と開発段階への戻りを停止します。途中失敗もfailed_after_accessとして残し、テスト利用履歴を確認せず自動再実行しません。これは運用上の防止策で、テストデータを暗号化するものではありません。

## 指標と出力

主指標はMacro-F1。補助は活性別／Macro AP・ROC-AUC、Micro／Samples F1、Hamming loss、Subset Accuracy、Sensitivity、Specificity、Precision、MCC、Binary Log Loss、Brierです。Brierは `(確率−ラベル)^2` の平均を各活性で計算し、3活性を平均します。未定義AUC等はnull／空欄で、0埋めしません。

| 保存先 | 内容 |
|---|---|
| run_config.json・splits.json | 固定設定とハッシュ、再利用する分割 |
| data_audit.json・skipped_sequences.csv | 件数・組み合わせ・除外理由 |
| holdout_assignments.csv・fold_assignments.csv | 元配列・ラベルに対応した分割 |
| 各モデルのoof_predictions.csv | 正解3列・予測3列・確率3列・fold |
| development_predictions.npz | 整列済み全モデルOOF |
| model_comparison_cv.csv・development_summary.xlsx | 単独モデルのCV平均・SD・有効fold数 |
| oof_roc.png・oof_pr.png・各モデルのoof_confusion_matrices.png | 開発OOFの評価図 |
| CNN・MLPのloss_curves.png | 内部検証と学習の曲線 |
| ensemble/ | ranking.csv、by_model_count.csv、summary.xlsx、k別成績と上位候補の図 |
| nested/ | 各foldの内外分割・候補・比較表・平均SD・Stackingと平均のペア差の図 |
| bias/ | 長さのみのLR、長さ分布、共通長範囲、誤りの一致率・相関 |
| final_recipe.json | 学習側だけで固定した最終構成 |
| final/ | 最終学習の予測キャッシュ・学習記録、テスト予測・指標・Excel・ROC・PR・混同行列 |

共通長範囲は最初の `--length-range 最短 最長`、または学習側の陽性配列長範囲の共通部分で決めます。範囲内の既存OOFを再評価する処理であり、範囲内だけでモデルを学習し直す実験ではありません。

類似配列クラスタ分割、実測陰性を追加した未知ラベルの扱い、確率校正、溶血性モデルは今後の発展課題です。現データのソース所属は活性ラベルの定義と交絡するため、単純な出典分類器を独立の根拠としては作成していません。

## 検証と記録

```bash
python -m unittest discover -s tests_multi_label -v
```

テストは合成特徴量・モック・一時ファイルを使い、実ペプチドの性能値を作りません。[11月17日の実装対応](experiment_logs_multi_label/2026-11-17_research_plan_implementation.md) と [11月16日の集計ログ](experiment_logs_multi_label/2026-11-16_experiment_log.md) に経緯と状況を記録します。

一次資料は [ESM](https://github.com/facebookresearch/esm)、[ClassifierChain](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html)、[CatBoost MultiLogloss](https://catboost.ai/docs/en/concepts/loss-functions-multilabel-classification)、[XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html)、[iterative-stratification](https://github.com/trent-b/iterative-stratification) です。
