"""Binary-relevance XGBoost with training-only per-label imbalance weights."""


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    import numpy as np
    from xgboost import XGBClassifier
    from .model_utils import BinaryRelevancePredictor, positive_probabilities

    positives = y_train.sum(axis=0)
    if np.any(positives == 0) or np.any(positives == len(y_train)):
        raise ValueError("XGBoost training requires both classes for every label.")
    weights = (len(y_train) - positives) / positives
    estimators = []
    for column, weight in enumerate(weights):
        estimator = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=args.trees, learning_rate=0.05, max_depth=6,
            subsample=1.0, colsample_bytree=1.0, scale_pos_weight=float(weight),
            n_jobs=args.jobs, random_state=seed, verbosity=0)
        estimator.fit(X_train, y_train[:, column])
        estimators.append(estimator)
    model = BinaryRelevancePredictor(estimators)
    p = positive_probabilities(model, X_test)
    return p, {"method": "binary relevance", "trees_per_label": args.trees,
               "scale_pos_weight": weights.tolist(), "weight_source": "training only"}
