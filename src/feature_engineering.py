def add_features(df):
    df = df.copy()

    # Binary encoding
    df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})
    df['book_table'] = df['book_table'].map({'Yes': 1, 'No': 0})

    # Restaurant type encoding (simplified)
    df['rest_type'] = df['rest_type'].astype(str).apply(lambda x: x.split(',')[0])

    # Location popularity
    location_counts = df['location'].value_counts().to_dict()
    df['location_popularity'] = df['location'].map(location_counts)

    return df