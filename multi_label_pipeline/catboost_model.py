"""Native multi-label CatBoost with MultiLogloss and three probabilities."""


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="MultiLogloss", iterations=args.trees, depth=6,
        learning_rate=0.05, random_seed=seed, thread_count=args.jobs,
        task_type="CPU", verbose=False, allow_writing_files=False)
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)
    return p, {"method": "native multilabel", "loss": "MultiLogloss", "iterations": args.trees,
               "class_weight": "none; unweighted per-label binary cross entropy", "device": "CPU"}
