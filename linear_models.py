from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    import numpy as np
    X_train = np.asarray(data.train_features, dtype=np.float64)
    y_train = np.asarray(data.train_target, dtype=np.float64).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=np.float64)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    n_samples, n_features = X_train.shape
    lr = float(data.learning_rate)
    epochs = int(data.epochs)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    X_train_norm = (X_train - mean) / std_safe
    X_test_norm = (X_test - mean) / std_safe
    weights_norm = np.zeros(n_features, dtype=np.float64)
    bias_norm = 0.0
    loss_history: list[float] = []
    for _ in range(epochs):
        y_pred = X_train_norm @ weights_norm + bias_norm
        error = y_pred - y_train
        mse = float(np.mean(error ** 2))
        loss_history.append(mse)
        grad_w = (2.0 / n_samples) * (X_train_norm.T @ error)
        grad_b = float((2.0 / n_samples) * np.sum(error))
        weights_norm -= lr * grad_w
        bias_norm -= lr * grad_b
    weights = weights_norm / std_safe
    bias = float(bias_norm - np.sum(weights_norm * mean / std_safe))
    predictions = X_test @ weights + bias
    return RegressionResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=predictions,
    )

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    import numpy as np
    x_train = np.asarray(data.train_features, dtype=np.float64)
    y_train = np.asarray(data.train_target, dtype=np.float64).reshape(-1)
    x_test = np.asarray(data.test_features, dtype=np.float64)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    if x_test.ndim == 1:
        x_test = x_test.reshape(-1, 1)
    n_samples, n_features = x_train.shape
    lr = float(data.learning_rate)
    epochs = int(data.epochs)
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    x_train_norm = (x_train - mean) / std_safe
    x_test_norm = (x_test - mean) / std_safe
    weights_norm = np.zeros(n_features, dtype=np.float64)
    bias_norm = 0.0
    loss_history: list[float] = []
    eps = 1e-15
    for _ in range(epochs):
        logits = np.clip(x_train_norm @ weights_norm + bias_norm, -500.0, 500.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        bce = float(
            -np.mean(
                y_train * np.log(probs_clipped)
                + (1.0 - y_train) * np.log(1.0 - probs_clipped)
            )
        )
        loss_history.append(bce)
        error = probs - y_train
        grad_w = (1.0 / n_samples) * (x_train_norm.T @ error)
        grad_b = float((1.0 / n_samples) * np.sum(error))
        weights_norm -= lr * grad_w
        bias_norm -= lr * grad_b

    weights = weights_norm / std_safe
    bias = float(bias_norm - np.sum(weights_norm * mean / std_safe))
    test_logits = np.clip(x_test @ weights + bias, -500.0, 500.0)
    probabilities = 1.0 / (1.0 + np.exp(-test_logits))
    predictions = (probabilities >= 0.5).astype(int)
    return ClassificationResult(
            weights=weights,
            bias=bias,
            loss_history=loss_history,
            probabilities=probabilities,
            predictions=predictions,
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    X_train = np.asarray(data.train_features, dtype=np.float64)
    y_train = np.asarray(data.train_target, dtype=int).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=np.float64)
    y_test = np.asarray(data.test_target, dtype=int).reshape(-1)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    def evaluate_model(model, X_tr, X_te) -> ClassificationMetrics:
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        return ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_test, y_prob)),
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
        )

    lr_metrics = evaluate_model(
        LogisticRegression(max_iter=1000, random_state=42),
        X_train_scaled,
        X_test_scaled,
    )
    dt_metrics = evaluate_model(
        DecisionTreeClassifier(max_depth=5, random_state=42),
        X_train,
        X_test,
    )
    knn_metrics = evaluate_model(
        KNeighborsClassifier(n_neighbors=5),
        X_train_scaled,
        X_test_scaled,
    )

    return BaselineComparisonResult(
        logistic_regression=lr_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics,
    )

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler


    df_reg = pd.read_csv("Student_Performance.csv").dropna()
    df_reg["Extracurricular Activities"] = (
        df_reg["Extracurricular Activities"]
        .astype(str)
        .str.strip()
        .map({"Yes": 1.0, "No": 0.0, "1": 1.0, "0": 0.0})
        .fillna(0.0)
    )
    X_reg = df_reg.drop(columns=["Performance Index"]).to_numpy(dtype=np.float64)
    y_reg = df_reg["Performance Index"].to_numpy(dtype=np.float64)

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    scaler_r = StandardScaler()
    X_tr_r_s = scaler_r.fit_transform(X_tr_r)
    X_te_r_s = scaler_r.transform(X_te_r)

    reg_res = train_linear_regression(
        RegressionInput(
            train_features=X_tr_r_s,
            train_target=y_tr_r,
            test_features=X_te_r_s,
            test_target=y_te_r,
            learning_rate=0.05,
            epochs=800,
        )
    )
    test_mse = float(np.mean((reg_res.predictions - y_te_r) ** 2))
    print("=== Задача 1: Линейная регрессия ===")
    print(f"Итоговая MSE на test: {test_mse:.4f}")


    df_clf = pd.read_csv("german.csv", sep=None, engine="python").dropna()
    target_col = "Creditability" if "Creditability" in df_clf.columns else df_clf.columns[0]

    X_clf = df_clf.drop(columns=[target_col]).to_numpy(dtype=np.float64)
    y_clf = df_clf[target_col].to_numpy(dtype=int)

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    scaler_c = StandardScaler()
    X_tr_c_s = scaler_c.fit_transform(X_tr_c)
    X_te_c_s = scaler_c.transform(X_te_c)

    clf_input = ClassificationInput(
        train_features=X_tr_c_s,
        train_target=y_tr_c,
        test_features=X_te_c_s,
        test_target=y_te_c,
        learning_rate=0.1,
        epochs=1000,
    )
    clf_res = train_logistic_classifier(clf_input)

    print("\n=== Задача 2: Логистическая регрессия (german.csv) ===")
    print(f"Начальная BCE (1 эпоха):    {clf_res.loss_history[0]:.4f}")
    print(f"Конечная BCE (1000 эпоха):  {clf_res.loss_history[-1]:.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_te_c, clf_res.probabilities):.4f}")
    print(f"Accuracy:  {accuracy_score(y_te_c, clf_res.predictions):.4f}")
    print(f"Precision: {precision_score(y_te_c, clf_res.predictions):.4f}")
    print(f"Recall:    {recall_score(y_te_c, clf_res.predictions):.4f}")
    print(f"F1-score:  {f1_score(y_te_c, clf_res.predictions):.4f}")
    baseline_res = compare_classical_models(clf_input)
    print("\n=== Задача 3: Сравнение классификаторов ===")
    print(f"sklearn LogisticRegression ROC-AUC: {baseline_res.logistic_regression.roc_auc:.4f}")
    print(f"DecisionTree ROC-AUC:               {baseline_res.decision_tree.roc_auc:.4f}")
    print(f"KNN ROC-AUC:                        {baseline_res.knn.roc_auc:.4f}")


    import json
    from dataclasses import asdict

    results_data = {
        "linear_regression": {"test_mse": round(test_mse, 6)},
        "classification": {
            "own_logistic": {
                "roc_auc": round(float(roc_auc_score(y_te_c, clf_res.probabilities)), 6),
                "accuracy": round(float(accuracy_score(y_te_c, clf_res.predictions)), 6),
                "precision": round(float(precision_score(y_te_c, clf_res.predictions, zero_division=0)), 6),
                "recall": round(float(recall_score(y_te_c, clf_res.predictions, zero_division=0)), 6),
                "f1": round(float(f1_score(y_te_c, clf_res.predictions, zero_division=0)), 6),
            },
            "logistic_regression": {k: round(v, 6) for k, v in asdict(baseline_res.logistic_regression).items()},
            "decision_tree": {k: round(v, 6) for k, v in asdict(baseline_res.decision_tree).items()},
            "knn": {k: round(v, 6) for k, v in asdict(baseline_res.knn).items()},
        },
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print("\nФайл experiment_results.json успешно создан в папке проекта!")