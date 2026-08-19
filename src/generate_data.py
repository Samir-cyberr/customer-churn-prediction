from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_SAMPLES = 3000


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(18, 65, n_samples)
    tenure_months = rng.integers(1, 61, n_samples)
    monthly_spend = np.clip(rng.normal(75, 30, n_samples), 15, 250)
    orders = np.maximum(rng.poisson(7, n_samples), 0)
    avg_order_value = np.clip(monthly_spend / np.maximum(orders / 2, 1), 10, 250)
    days_since_last_order = np.clip(
        rng.gamma(shape=2.2, scale=14, size=n_samples), 1, 180
    )
    sessions = np.clip(rng.poisson(12, n_samples), 1, 60)
    support_tickets = np.clip(rng.poisson(1.5, n_samples), 0, 10)
    discount_usage = np.clip(rng.beta(2, 5, n_samples), 0, 1)
    satisfaction = np.clip(rng.normal(3.5, 0.8, n_samples), 1, 5)
    mobile_app = rng.binomial(1, 0.62, n_samples)

    # Synthetic churn mechanism with noise.
    risk = (
        -1.8
        + 0.018 * days_since_last_order
        - 0.22 * satisfaction
        - 0.018 * orders
        - 0.003 * monthly_spend
        + 0.10 * support_tickets
        - 0.25 * mobile_app
        + 0.35 * discount_usage
        + rng.normal(0, 0.55, n_samples)
    )

    probability = 1 / (1 + np.exp(-risk))
    churn = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "monthly_spend": np.round(monthly_spend, 2),
            "orders": orders,
            "avg_order_value": np.round(avg_order_value, 2),
            "days_since_last_order": np.round(days_since_last_order, 1),
            "sessions": sessions,
            "support_tickets": support_tickets,
            "discount_usage": np.round(discount_usage, 3),
            "satisfaction": np.round(satisfaction, 2),
            "mobile_app": mobile_app,
            "churn": churn,
        }
    )


def main() -> None:
    output = Path("data/customers.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    df.to_csv(output, index=False)

    print(f"Saved {len(df):,} rows to {output}")
    print(f"Churn rate: {df['churn'].mean():.2%}")


if __name__ == "__main__":
    main()
