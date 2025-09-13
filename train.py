# train.py
import os
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

def main():
    print("📥 Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    print("✅ Dataset loaded successfully!")
    print(df.head())

    # Features & target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"📊 Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Preprocessing + model
    numeric_features = X.columns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    print("⚡ Training the model...")
    model.fit(X_train, y_train)
    print("✅ Model training complete!")

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📈 Model Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/house_price_pipeline.joblib")
    print("\n💾 Model saved as models/house_price_pipeline.joblib")

if __name__ == "__main__":
    main()
