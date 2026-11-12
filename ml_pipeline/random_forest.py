"""Random Forest classifier on 1,280-dimensional ESM-2 embeddings."""


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=args.trees, class_weight="balanced_subsample",
        random_state=seed, n_jobs=args.jobs,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, output / "model.joblib")
    return model.predict_proba(X_test), {"n_estimators": args.trees}
