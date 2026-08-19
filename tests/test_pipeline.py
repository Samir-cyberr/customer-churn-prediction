from src.generate_data import generate_dataset


def test_generated_dataset_shape():
    df = generate_dataset(100)
    assert len(df) == 100
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()


def test_required_features_exist():
    df = generate_dataset(20)
    required = {
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
        "churn",
    }
    assert required.issubset(df.columns)
