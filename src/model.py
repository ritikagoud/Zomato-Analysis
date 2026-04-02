from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_model(df):
    df = df.copy()

    features = [
        'votes',
        'approx_cost(for two people)',
        'online_order',
        'book_table',
        'location_popularity'
    ]

    X = df[features]
    y = df['rate']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    error = mean_absolute_error(y_test, preds)

    return model, error