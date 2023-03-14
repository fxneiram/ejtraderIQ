from ia_core.rf.process import DataProcessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def model_params(df):
    last = df.tail(1)[
        [
            "rsi",
            "k_percent",
            "r_percent",
            "macd",
            "macd_ema",
            "price_rate_of_change",
            "on_balance_volume",
        ]
    ]
    df.dropna(inplace=True)

    x_cols = df[
        [
            "rsi",
            "k_percent",
            "r_percent",
            "macd",
            "macd_ema",
            "price_rate_of_change",
            "on_balance_volume",
        ]
    ]
    y_cols = df["predictions"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_cols, y_cols, random_state=0, shuffle=False
    )

    return last, x_cols, y_cols, x_train, x_test, y_train, y_test


def predict_rf(df):
    df = DataProcessor(df).df
    rand_frst_clf = RandomForestClassifier(
        n_estimators=100, oob_score=True, criterion="gini", random_state=0
    )

    today, x_cols, y_cols, x_train, x_test, y_train, y_test = model_params(df)

    rand_frst_clf.fit(x_train, y_train)

    y_pred = rand_frst_clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100.0
    accuracy = round(accuracy, 2)

    next = rand_frst_clf.predict_proba(today)[0]

    baja = round(next[0] * 100, 2)
    alta = round(next[1] * 100, 2)

    return {
        "operation": "put" if next[0] > next[1] else "call",
        "accuracy": accuracy,
        "put": baja,
        "call": alta
    }
