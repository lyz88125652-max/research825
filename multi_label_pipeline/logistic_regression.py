"""Standardized L2 logistic regression, one binary classifier per activity."""


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from .model_utils import positive_probabilities

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifiers", MultiOutputClassifier(LogisticRegression(
            C=args.linear_c, class_weight="balanced", max_iter=2000,
            solver="lbfgs", random_state=seed), n_jobs=1)),
    ])
    model.fit(X_train, y_train)
    p = positive_probabilities(model.named_steps["classifiers"],
                               model.named_steps["scaler"].transform(X_test))
    return p, {"method": "binary relevance", "regularization": "L2",
               "C": args.linear_c, "scaler_fit": "training only"}
