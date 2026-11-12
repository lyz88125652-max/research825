"""LightGBM multiclass classifier on ESM-2 embeddings."""


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    import joblib
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=args.trees,
        learning_rate=0.05, num_leaves=31, class_weight="balanced",
        random_state=seed, n_jobs=args.jobs, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, output / "model.joblib")
    return model.predict_proba(X_test), {"n_estimators": args.trees}
