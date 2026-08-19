from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/customers.csv")
MODEL_DIR = Path("models")

RANDOM_STATE = 42
TARGET = "churn"

FEATURES = [
    "age",
    "tenure_months",
    "monthly_spend",
    "orders",
    "avg_order_value",
    "days_since_last_order",
    "sessions",
    "support_tickets",
    "discount_usage",
    "satisfaction",
    "mobile_app",
]


def build_preprocessor():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, FEATURES)
        ],
        remainder="drop",
    )


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def find_best_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_f1 = 0

    for threshold in [i / 100 for i in range(20, 81)]:
        predictions = (probabilities >= threshold).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run "
            "`python -m src.generate_data` first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    print(f"Dataset size: {len(df):,}")
    print(f"Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    results = []
    fitted_models = {}

    for name, estimator in get_models().items():

        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        threshold, _ = find_best_threshold(
            y_test,
            probabilities,
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        metrics = {
            "model": name,
            "roc_auc": roc_auc_score(
                y_test,
                probabilities,
            ),
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "threshold": threshold,
        }

        results.append(metrics)
        fitted_models[name] = pipeline

    results_df = (
        pd.DataFrame(results)
        .sort_values("f1", ascending=False)
    )

    best_name = results_df.iloc[0]["model"]
    best_threshold = float(
        results_df.iloc[0]["threshold"]
    )

    best_model = fitted_models[best_name]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        MODEL_DIR / "churn_model.joblib",
    )

    results_df.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False,
    )

    best_probabilities = (
        best_model.predict_proba(X_test)[:, 1]
    )

    best_predictions = (
        best_probabilities >= best_threshold
    ).astype(int)

    evaluation = {
        "best_model": best_name,
        "threshold": best_threshold,
        "metrics": {
            "roc_auc": float(
                roc_auc_score(
                    y_test,
                    best_probabilities,
                )
            ),
            "accuracy": float(
                accuracy_score(
                    y_test,
                    best_predictions,
                )
            ),
            "precision": float(
                precision_score(
                    y_test,
                    best_predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    best_predictions,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    best_predictions,
                    zero_division=0,
                )
            ),
        },
    }

    with open(
        MODEL_DIR / "evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=2,
        )

    print("\nModel comparison:")
    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nBest model:", best_name)
    print(
        "Best threshold:",
        best_threshold,
    )

    print(
        "\nSaved model:",
        MODEL_DIR / "churn_model.joblib",
    )


if __name__ == "__main__":
    main()