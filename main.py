from src.data_cleaning import load_data, clean_data
from src.feature_engineering import add_features
from src.model import train_model
from src.utils import (
    top_n,
    avg_rating_by_group,
    add_price_bucket,
    business_insights,
    correlation_with_rating
)

import matplotlib.pyplot as plt
import seaborn as sns


def run_pipeline():
    # -----------------------------
    # 1. LOAD DATA
    # -----------------------------
    df = load_data("data/zomato.csv")
    print("Initial Shape:", df.shape)

    # -----------------------------
    # 2. CLEAN DATA
    # -----------------------------
    df = clean_data(df)
    print("After Cleaning:", df.shape)

    # -----------------------------
    # 3. FEATURE ENGINEERING
    # -----------------------------
    df = add_features(df)
    df = add_price_bucket(df)

    # -----------------------------
    # 4. BASIC EDA
    # -----------------------------
    print("\n===== TOP LOCATIONS =====")
    print(top_n(df, 'location'))

    print("\n===== TOP CUISINES =====")
    print(top_n(df, 'cuisines'))

    print("\n===== AVG RATING BY LOCATION =====")
    print(avg_rating_by_group(df, 'location').head(10))

    print("\n===== CORRELATION WITH RATING =====")
    print(correlation_with_rating(df))

    # -----------------------------
    # 5. VISUALIZATIONS
    # -----------------------------
    plt.figure(figsize=(10, 5))
    sns.histplot(df['rate'], bins=20)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.boxplot(x='price_bucket', y='rate', data=df)
    plt.title("Price Range vs Rating")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    top_locations = df['location'].value_counts().head(10).index
    sns.barplot(
        x=df[df['location'].isin(top_locations)]['location'],
        y=df[df['location'].isin(top_locations)]['rate']
    )
    plt.title("Top Locations vs Ratings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 6. BUSINESS INSIGHTS
    # -----------------------------
    print("\n===== BUSINESS INSIGHTS =====")
    insights = business_insights(df)

    for key, value in insights.items():
        print(f"\n{key.upper()}:")
        print(value)

    # -----------------------------
    # 7. MACHINE LEARNING MODEL
    # -----------------------------
    print("\n===== MODEL TRAINING =====")
    model, error = train_model(df)

    print(f"Model Mean Absolute Error: {error:.3f}")

    # -----------------------------
    # 8. FINAL TAKEAWAYS
    # -----------------------------
    print("\n===== FINAL TAKEAWAYS =====")
    print("- Online ordering tends to increase ratings.")
    print("- Mid-range pricing often gives better ratings.")
    print("- Some locations are highly competitive but not high quality.")


if __name__ == "__main__":
    run_pipeline()

