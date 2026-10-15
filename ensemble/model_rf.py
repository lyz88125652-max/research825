import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

def find_best_threshold(y_true, probs, metric="mcc"):
    best_thresh = 0.5
    best_score = -1.0
    for t in np.arange(0.15, 0.70, 0.01):
        preds = (probs >= t).astype(int)
        if metric == "mcc":
            score = matthews_corrcoef(y_true, preds)
        elif metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        else:
            score = matthews_corrcoef(y_true, preds)
            
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score

def run_rf(X_train, y_train, X_test, y_test):
    """
    Random Forest 5-Fold Cross Validation & Test Evaluation
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros(len(y_train))
    test_probs_list = []
    train_aucs = []
    cv_aucs = []
    best_thresholds = []
    
    scaler_full = StandardScaler()
    X_train_s_full = scaler_full.fit_transform(X_train)
    X_test_s_full = scaler_full.transform(X_test)
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[t_idx], X_train[v_idx]
        y_tr, y_val = y_train[t_idx], y_train[v_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        # Random Forest モデル構築
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            oob_score=True,
            random_state=42 + fold,
            n_jobs=-1
        )
        
        model.fit(X_tr_s, y_tr)
        
        val_probs = model.predict_proba(X_val_s)[:, 1]
        tr_probs = model.predict_proba(X_tr_s)[:, 1]
        
        oof_probs[v_idx] = val_probs
        
        opt_thresh, _ = find_best_threshold(y_val, val_probs, metric="mcc")
        best_thresholds.append(opt_thresh)
        
        cv_aucs.append(roc_auc_score(y_val, val_probs))
        train_aucs.append(roc_auc_score(y_tr, tr_probs))
        
        test_p = model.predict_proba(X_test_s_full)[:, 1]
        test_probs_list.append(test_p)
        
    mean_opt_thresh = np.mean(best_thresholds)
    avg_test_probs = np.mean(test_probs_list, axis=0)
    
    return {
        "cv_probs": oof_probs,
        "test_probs": avg_test_probs,
        "train_auc": np.mean(train_aucs),
        "cv_auc": np.mean(cv_aucs),
        "opt_thresh": mean_opt_thresh
    }
