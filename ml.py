# ml.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_and_evaluate_model(df: pd.DataFrame):
    features = ["hours_studied", "attendance_rate", "midterm_score"]
    target = "final_score"

    X = df[features]
    y = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    return model, score

def predict_score(model, input_data):
    return model.predict(input_data)
