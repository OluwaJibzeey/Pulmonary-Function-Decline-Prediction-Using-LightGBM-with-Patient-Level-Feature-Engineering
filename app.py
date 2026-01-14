import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import KFold

# Load data
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# Preprocess Patient column to ensure consistent matching
train_df["Patient"] = train_df["Patient"].astype(str).str.strip()
test_df["Patient"] = test_df["Patient"].astype(str).str.strip()

# Encode categorical variables using combined training and test data
le_sex = LabelEncoder()
le_smoking = LabelEncoder()

combined_sex = pd.concat([train_df["Sex"], test_df["Sex"]])
le_sex.fit(combined_sex)

combined_smoking = pd.concat([train_df["SmokingStatus"], test_df["SmokingStatus"]])
le_smoking.fit(combined_smoking)

train_df["Sex"] = le_sex.transform(train_df["Sex"])
test_df["Sex"] = le_sex.transform(test_df["Sex"])

train_df["SmokingStatus"] = le_smoking.transform(train_df["SmokingStatus"])
test_df["SmokingStatus"] = le_smoking.transform(test_df["SmokingStatus"])

# Prepare training data with enhanced features including baseline Percent and logarithmic transformations
train_data = []
for patient, group in train_df.groupby("Patient"):
    baseline = group[group["Weeks"] == 0]
    if baseline.empty:
        continue
    age = baseline["Age"].values[0]
    sex = baseline["Sex"].values[0]
    smoking = baseline["SmokingStatus"].values[0]
    baseline_fvc = baseline["FVC"].values[0]
    baseline_percent = baseline["Percent"].values[0]
    log_baseline_fvc = np.log1p(baseline_fvc)  # Logarithmic transformation
    log_baseline_percent = np.log1p(baseline_percent)  # Logarithmic transformation

    for _, row in group.iterrows():
        week = row["Weeks"]
        fvc = row["FVC"]
        train_data.append(
            {
                "Age": age,
                "Sex": sex,
                "SmokingStatus": smoking,
                "baseline_fvc": baseline_fvc,
                "baseline_percent": baseline_percent,
                "log_baseline_fvc": log_baseline_fvc,
                "log_baseline_percent": log_baseline_percent,
                "week": week,
                "FVC": fvc,
            }
        )

train_df_processed = pd.DataFrame(train_data)

# Add interaction, polynomial, and logarithmic features
X_train = train_df_processed[
    [
        "Age",
        "Sex",
        "SmokingStatus",
        "baseline_fvc",
        "baseline_percent",
        "log_baseline_fvc",
        "log_baseline_percent",
        "week",
    ]
]
X_train["week_sq"] = X_train["week"] ** 2
X_train["week_cubed"] = X_train["week"] ** 3  # Higher-order polynomial feature
X_train["age_week"] = X_train["Age"] * X_train["week"]
X_train["baseline_fvc_week"] = X_train["baseline_fvc"] * X_train["week"]
X_train["baseline_percent_week"] = X_train["baseline_percent"] * X_train["week"]
X_train["log_baseline_fvc_week"] = X_train["log_baseline_fvc"] * X_train["week"]
X_train["log_baseline_percent_week"] = X_train["log_baseline_percent"] * X_train["week"]
y_train = train_df_processed["FVC"]

# Train LightGBM models with 5-fold CV and collect predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    y_pred = model.predict(X_val)
    fold_score = np.sqrt(np.mean((y_val - y_pred) ** 2))
    cv_scores.append(fold_score)

# Average predictions from all folds (simplified ensembling)
score = np.mean(cv_scores)
print(f"5-fold CV RMSE: {score}")

# Prepare test data with enhanced features using test set's baseline information
test_data = []
for patient, group in test_df.groupby("Patient"):
    baseline = group  # Use test set's baseline information
    age = baseline["Age"].values[0]
    sex = baseline["Sex"].values[0]
    smoking = baseline["SmokingStatus"].values[0]
    baseline_fvc = baseline["FVC"].values[0]
    baseline_percent = baseline["Percent"].values[0]
    log_baseline_fvc = np.log1p(baseline_fvc)
    log_baseline_percent = np.log1p(baseline_percent)

    # Predict for weeks 1 to 100
    for week in range(1, 101):
        test_data.append(
            {
                "Patient": patient,
                "week": week,
                "Age": age,
                "Sex": sex,
                "SmokingStatus": smoking,
                "baseline_fvc": baseline_fvc,
                "baseline_percent": baseline_percent,
                "log_baseline_fvc": log_baseline_fvc,
                "log_baseline_percent": log_baseline_percent,
            }
        )

test_df_processed = pd.DataFrame(test_data)

# Add interaction, polynomial, and logarithmic features for test data
X_test = test_df_processed[
    [
        "Age",
        "Sex",
        "SmokingStatus",
        "baseline_fvc",
        "baseline_percent",
        "log_baseline_fvc",
        "log_baseline_percent",
        "week",
    ]
]
X_test["week_sq"] = X_test["week"] ** 2
X_test["week_cubed"] = X_test["week"] ** 3
X_test["age_week"] = X_test["Age"] * X_test["week"]
X_test["baseline_fvc_week"] = X_test["baseline_fvc"] * X_test["week"]
X_test["baseline_percent_week"] = X_test["baseline_percent"] * X_test["week"]
X_test["log_baseline_fvc_week"] = X_test["log_baseline_fvc"] * X_test["week"]
X_test["log_baseline_percent_week"] = X_test["log_baseline_percent"] * X_test["week"]

# Predict using all models and average predictions
all_test_preds = []
for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    y_pred = model.predict(X_test)
    all_test_preds.append(y_pred)

avg_test_pred = np.mean(all_test_preds, axis=0)

# Prepare submission
submission = []
for i, row in test_df_processed.iterrows():
    patient_week = f"{row['Patient']}_{row['week']}"
    fvc = avg_test_pred[i]
    confidence = (
        70  # Fixed confidence value, could be adjusted based on prediction uncertainty
    )
    submission.append(
        {"Patient_Week": patient_week, "FVC": fvc, "Confidence": confidence}
    )

submission_df = pd.DataFrame(submission)
submission_df.to_csv("submission.csv", index=False)
