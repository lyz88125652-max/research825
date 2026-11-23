"""Three label-specific LR meta-learners trained on base-model OOF scores."""

import numpy as np
from .ensemble import validate_tensor


class LabelwiseStacker:
    def __init__(self, C=1.0, class_weight=None, seed=42):
        self.C, self.class_weight, self.seed = C, class_weight, seed

    def fit(self, probabilities, y, names):
        from sklearn.linear_model import LogisticRegression
        p = validate_tensor(y, probabilities, names)
        self.model_names_ = list(names)
        self.estimators_ = []
        for column in range(3):
            if len(np.unique(y[:, column])) != 2: raise ValueError("Meta-training needs both classes.")
            estimator = LogisticRegression(C=self.C, class_weight=self.class_weight,
                                            solver="lbfgs", max_iter=2000, random_state=self.seed)
            estimator.fit(p[:, :, column], y[:, column])
            self.estimators_.append(estimator)
        return self

    def predict_proba(self, probabilities, names):
        if list(names) != self.model_names_: raise ValueError("Stacking model order mismatch.")
        p = np.asarray(probabilities)
        validate_tensor(np.zeros((len(p), 3), dtype=int), p, names)
        return np.column_stack([est.predict_proba(p[:, :, i])[:, list(est.classes_).index(1)]
                                for i, est in enumerate(self.estimators_)])


def fit_stacker(probabilities, y, names, subset, args, seed):
    columns = [names.index(name) for name in subset]
    weight = None if args.meta_class_weight == "none" else "balanced"
    model = LabelwiseStacker(args.meta_c, weight, seed).fit(probabilities[:, columns], y, subset)
    return model, columns
