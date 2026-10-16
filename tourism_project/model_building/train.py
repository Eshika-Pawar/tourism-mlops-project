# train.py
# Trains several tree based models, tunes them, tracks every run with MLflow,
# and registers the best model on the Hugging Face Model Hub.
# Run from the repo root:  python tourism_project/model_building/train.py

import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
username = api.whoami()["name"]
dataset_repo = f"{username}/tourism-wellness-dataset"
model_repo = f"{username}/tourism-wellness-model"

# if no MLflow server is given we just log to a local folder
#mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("tourism-wellness-package")

THRESHOLD = 0.45  # a bit below 0.5 because buyers are the minority class


# ---------------------------------------------------------------- load data
def load(name):
    path = hf_hub_download(
        repo_id=dataset_repo, filename=name, repo_type="dataset", token=HF_TOKEN
    )
    return pd.read_csv(path)


Xtrain, Xtest = load("Xtrain.csv"), load("Xtest.csv")
ytrain, ytest = load("ytrain.csv").squeeze(), load("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# class weight to handle the imbalance (about 4 non-buyers for every buyer)
neg, pos = (ytrain == 0).sum(), (ytrain == 1).sum()
scale_pos_weight = neg / pos

# ------------------------------------------------- models + parameter grids
models = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"decisiontreeclassifier__max_depth": [4, 6, 8],
         "decisiontreeclassifier__min_samples_leaf": [5, 10]},
    ),
    "Bagging": (
        BaggingClassifier(random_state=42),
        {"baggingclassifier__n_estimators": [50, 100],
         "baggingclassifier__max_samples": [0.7, 1.0]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"randomforestclassifier__n_estimators": [100, 200],
         "randomforestclassifier__max_depth": [10, 20, None],
         "randomforestclassifier__min_samples_leaf": [1, 3]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {"adaboostclassifier__n_estimators": [100, 200],
         "adaboostclassifier__learning_rate": [0.1, 0.5, 1.0]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"gradientboostingclassifier__n_estimators": [100, 200],
         "gradientboostingclassifier__learning_rate": [0.05, 0.1],
         "gradientboostingclassifier__max_depth": [3, 5]},
    ),
    "XGBoost": (
        xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight,
                          eval_metric="logloss"),
        {"xgbclassifier__n_estimators": [100, 200],
         "xgbclassifier__max_depth": [3, 5],
         "xgbclassifier__learning_rate": [0.05, 0.1],
         "xgbclassifier__colsample_bytree": [0.6, 0.8]},
    ),
}


def scores(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


# ------------------------------------------------------ train and track runs
results, fitted = [], {}

with mlflow.start_run(run_name="tourism_model_comparison"):
    for name, (estimator, grid) in models.items():
        with mlflow.start_run(run_name=name, nested=True):
            pipe = make_pipeline(preprocessor, estimator)
            search = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
            search.fit(Xtrain, ytrain)

            best = search.best_estimator_
            tr, te = scores(best, Xtrain, ytrain), scores(best, Xtest, ytest)

            mlflow.log_params({k.split("__")[1]: v for k, v in search.best_params_.items()})
            mlflow.log_metric("cv_best_f1", search.best_score_)
            mlflow.log_metrics({f"train_{k}": v for k, v in tr.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in te.items()})
            mlflow.log_param("threshold", THRESHOLD)

            fitted[name] = best
            results.append({"model": name, **{f"train_{k}": v for k, v in tr.items()},
                            **{f"test_{k}": v for k, v in te.items()}})
            print(f"{name:17s} test F1 = {te['f1']:.3f}  recall = {te['recall']:.3f}  "
                  f"ROC-AUC = {te['roc_auc']:.3f}")

    results_df = pd.DataFrame(results).sort_values("test_f1", ascending=False)
    print("\n", results_df.round(3).to_string(index=False))

    # best model = highest F1 on the test data
    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    mlflow.log_param("best_model", best_name)
    mlflow.log_metrics({f"best_test_{k}": results_df.iloc[0][f"test_{k}"]
                        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]})
    #mlflow.sklearn.log_model(best_model, artifact_path="best_tourism_model"
    try:
           try:
               mlflow.sklearn.log_model(best_model, name="best_tourism_model",
                                        serialization_format="cloudpickle")
           except TypeError:
               mlflow.sklearn.log_model(best_model, artifact_path="best_tourism_model",
                                        serialization_format="cloudpickle")
    except Exception as e:
           print("Warning: could not store the model file inside MLflow:", e)  

    print(f"\nBest model: {best_name}")
    print(classification_report(
        ytest, (best_model.predict_proba(Xtest)[:, 1] >= THRESHOLD).astype(int)))

# ---------------------------------------------- save + register on the Hub
os.makedirs("tourism_project/model_building/artifacts", exist_ok=True)
model_path = "tourism_project/model_building/artifacts/best_tourism_model_v1.joblib"
joblib.dump(best_model, model_path)

meta_path = "tourism_project/model_building/artifacts/model_meta.json"
with open(meta_path, "w") as f:
    json.dump({"best_model": best_name, "threshold": THRESHOLD}, f)

try:
    api.repo_info(repo_id=model_repo, repo_type="model")
    print(f"Model repo '{model_repo}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=model_repo, repo_type="model", private=False, token=HF_TOKEN)
    print(f"Model repo '{model_repo}' created.")

for p in [model_path, meta_path]:
    api.upload_file(
        path_or_fileobj=p,
        path_in_repo=os.path.basename(p),
        repo_id=model_repo,
        repo_type="model",
        commit_message=f"Register best model ({best_name})",
    )
print("Best model registered on the Hugging Face Model Hub.")
