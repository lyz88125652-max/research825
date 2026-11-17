"""A logistic classifier chain with training-only out-of-fold link features."""


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    import inspect
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import ClassifierChain
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from . import LABEL_NAMES
    from .data import make_splits

    # The scaler belongs inside each link so internal CV also fits it locally.
    base = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(
        C=args.linear_c, class_weight="balanced", max_iter=2000, solver="lbfgs",
        random_state=seed))])
    splits = make_splits(y_train, groups, args.chain_folds, seed)
    # sklearn renamed base_estimator to estimator in 1.7.
    estimator_key = "estimator" if "estimator" in inspect.signature(ClassifierChain).parameters else "base_estimator"
    order = list(args.chain_order)
    model = ClassifierChain(**{estimator_key: base}, order=order, cv=splits, random_state=seed)
    model.fit(X_train, y_train)
    # sklearn restores the original target column order in predict_proba.
    p = model.predict_proba(X_test)
    joblib.dump(model, output / "model.joblib")
    return p, {"method": "classifier chain", "base_model": "balanced L2 logistic regression",
               "order": [LABEL_NAMES[index] for index in order], "internal_folds": args.chain_folds,
               "link_features": "out-of-fold binary predictions; sequential predictions at inference",
               "order_selection": "fixed by CLI; independent test never used"}
