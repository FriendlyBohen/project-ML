import itertools
from sklearn.model_selection import KFold

def GridSearchCustom(df, num_splits, estimator, param_grid, scoring, verbose, X_train_split, y_train_split):
    best_score = None
    best_params = None

    param_values = param_grid.values()
    combinations_list = list(itertools.product(*param_values))

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    for combination in combinations_list:
        estimator.set_params(**dict(zip(param_grid.keys(), combination)))

        fold_scores = []
        for train_idx, val_idx in kf.split(X_train_split):
            X_train, X_val = X_train_split.iloc[train_idx], X_train_split.iloc[val_idx]
            y_train, y_val = y_train_split.iloc[train_idx], y_train_split.iloc[val_idx]

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_val)
            score = scoring._score_func(y_val, y_pred)
            fold_scores.append(score)

        avg_score = sum(fold_scores) / num_splits

        if verbose:
            print("Combination:", combination)
            print("Average Score:", avg_score)

        if best_score is None or avg_score > best_score:
            best_score = avg_score
            best_params = combination
            best_model = estimator

    return best_params, best_score, best_model