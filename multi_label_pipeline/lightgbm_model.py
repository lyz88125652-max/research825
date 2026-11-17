"""Binary relevance: one binary LightGBM classifier for each activity."""


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    import joblib
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.multioutput import MultiOutputClassifier

    model = MultiOutputClassifier(LGBMClassifier(
        objective="binary", n_estimators=args.trees, learning_rate=0.05,
        num_leaves=31, class_weight="balanced", random_state=seed,
        n_jobs=args.jobs, verbosity=-1, deterministic=True, force_col_wise=True), n_jobs=1)
    model.fit(X_train, y_train)
    p = np.column_stack([probs[:, list(est.classes_).index(1)]
                         for est, probs in zip(model.estimators_, model.predict_proba(X_test))])
    joblib.dump(model, output / "model.joblib")
    return p, {"method": "binary relevance", "trees_per_label": args.trees}
