import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42


def find_best_threshold(y_true, probs, metric="mcc"):
    best_thresh = 0.5
    best_score = -1.0
    for t in np.arange(0.15, 0.70, 0.01):
        preds = (probs >= t).astype(int)
        score = (
            matthews_corrcoef(y_true, preds)
            if metric == "mcc"
            else f1_score(y_true, preds, zero_division=0)
        )
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score


def run_lgb(X_train, y_train, X_test, y_test):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_probs = np.zeros(len(y_train))
    test_probs_list = []
    train_aucs = []
    cv_aucs = []
    best_thresholds = []

    pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": SEED,
        "scale_pos_weight": pos_weight,
        "verbose": -1,
        "n_jobs": -1,
    }

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[t_idx], X_train[v_idx]
        y_tr, y_val = y_train[t_idx], y_train[v_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        oof_probs[v_idx] = val_probs

        tr_probs = model.predict(X_tr, num_iteration=model.best_iteration)

        opt_thresh, _ = find_best_threshold(y_val, val_probs, metric="mcc")
        best_thresholds.append(opt_thresh)

        cv_aucs.append(roc_auc_score(y_val, val_probs))
        train_aucs.append(roc_auc_score(y_tr, tr_probs))

        fold_test_probs = model.predict(
            X_test, num_iteration=model.best_iteration)
        test_probs_list.append(fold_test_probs)

    return {
        "cv_probs": oof_probs,
        "test_probs": np.mean(test_probs_list, axis=0),
        "train_auc": np.mean(train_aucs),
        "cv_auc": np.mean(cv_aucs),
        "opt_thresh": np.mean(best_thresholds),
    }
