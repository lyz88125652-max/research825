"""Binary relevance: a separate balanced Random Forest for each activity."""


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    model = MultiOutputClassifier(RandomForestClassifier(
        n_estimators=args.trees, class_weight="balanced_subsample",
        random_state=seed, n_jobs=args.jobs), n_jobs=1)
    model.fit(X_train, y_train)
    p = np.column_stack([probs[:, list(est.classes_).index(1)]
                         for est, probs in zip(model.estimators_, model.predict_proba(X_test))])
    joblib.dump(model, output / "model.joblib")
    return p, {"method": "binary relevance", "trees_per_label": args.trees}
