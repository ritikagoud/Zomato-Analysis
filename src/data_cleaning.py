import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    df['rate'] = df['rate'].replace('NEW', np.nan)
    df['rate'] = df['rate'].replace('-', np.nan)
    df['rate'] = df['rate'].str.split('/').str[0]
    df['rate'] = df['rate'].astype(float)

    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

    # Cost cleaning
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str)
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.replace(',', '')
    df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

    # Drop unnecessary columns
    drop_cols = ['url', 'phone', 'dish_liked', 'reviews_list']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df.dropna(inplace=True)

    return df