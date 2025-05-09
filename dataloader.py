import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from missforest import MissForest


def load_customer_data():
    """
    Cleaning
    """
    # Load and clean raw data
    data = pd.read_csv("CustomerChurn.csv")
    data = data.drop(columns=['Customer ID', 'LoyaltyID'], errors='ignore')
    data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

    # Separate Total Charges for imputation
    data_encoded = pd.get_dummies(data.drop(columns=["Total Charges"]))
    data_encoded["Total Charges"] = data["Total Charges"]
    data_encoded = data_encoded.astype(np.float64)

    # Impute missing values using MissForest
    imputer = MissForest()
    data_imputed_array = imputer.fit_transform(data_encoded)
    data_imputed = pd.DataFrame(data_imputed_array, columns=data_encoded.columns)

    # Re-attach target
    data_imputed["Churn"] = data["Churn"].values

    # Final feature matrix and label vector
    feature_names = list(data_imputed.columns)
    feature_names.remove("Churn")
    X = data_imputed[feature_names].to_numpy(dtype=np.float64)
    Y = data_imputed["Churn"].to_numpy(dtype=np.float64)

    # Split: Train/Val/Test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)

    # Standardize using training set statistics
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_train = (X_train - means) / stds
    X_val = (X_val - means) / stds
    X_test = (X_test - means) / stds

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, feature_names