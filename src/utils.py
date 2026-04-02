import pandas as pd

def top_n(df, column, n=10):
    return df[column].value_counts().head(n)


def avg_rating_by_group(df, group_col):
    return df.groupby(group_col)['rate'].mean().sort_values(ascending=False)


def correlation_with_rating(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    return numeric_df.corr()['rate'].sort_values(ascending=False)


def price_bucket(cost):
    if cost < 300:
        return 'Low'
    elif cost < 700:
        return 'Medium'
    else:
        return 'High'


def add_price_bucket(df):
    df = df.copy()
    df['price_bucket'] = df['approx_cost(for two people)'].apply(price_bucket)
    return df


def business_insights(df):
    insights = {}

    insights['online_order_impact'] = df.groupby('online_order')['rate'].mean()
    insights['table_booking_impact'] = df.groupby('book_table')['rate'].mean()
    insights['price_vs_rating'] = df.groupby('price_bucket')['rate'].mean()

    return insights