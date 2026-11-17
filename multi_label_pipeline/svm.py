"""Standardized RBF SVM with a calibrated binary probability per activity."""


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    import joblib
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from .model_utils import positive_probabilities

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifiers", MultiOutputClassifier(SVC(
            C=args.svm_c, kernel="rbf", gamma="scale", probability=True,
            class_weight="balanced", random_state=seed, cache_size=512), n_jobs=1)),
    ])
    model.fit(X_train, y_train)
    p = positive_probabilities(model.named_steps["classifiers"],
                               model.named_steps["scaler"].transform(X_test))
    joblib.dump(model, output / "model.joblib")
    return p, {"method": "binary relevance", "kernel": "rbf", "C": args.svm_c,
               "gamma": "scale", "probability": "SVC internal training-only calibration",
               "decision": "probability threshold; not SVC.predict", "scaler_fit": "training only"}
