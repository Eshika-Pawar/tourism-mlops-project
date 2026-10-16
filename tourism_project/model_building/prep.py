# prep.py
# Loads the raw data from the Hugging Face dataset repo, cleans it,
# splits it into train / test and pushes the splits back to the Hub.
# Run from the repo root:  python tourism_project/model_building/prep.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
username = api.whoami()["name"]
repo_id = f"{username}/tourism-wellness-dataset"

# 1. Load the raw data from the Hub
raw_path = hf_hub_download(
    repo_id=repo_id, filename="tourism.csv", repo_type="dataset", token=HF_TOKEN
)
df = pd.read_csv(raw_path)
print("Raw data shape:", df.shape)

# 2. Cleaning
# the first column is just the old index and CustomerID is an identifier, neither helps prediction
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# fix wrong labels
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})

# remove duplicate rows if any
df = df.drop_duplicates().reset_index(drop=True)

# fill missing values (the file has none right now, but new data coming into the
# pipeline later might) -> median for numbers, mode for text
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == "object" or str(df[col].dtype).startswith("str"):
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

print("Clean data shape:", df.shape)

# 3. Train / test split (stratified because only ~19% customers bought the package)
target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

os.makedirs("tourism_project/data/processed", exist_ok=True)
Xtrain.to_csv("tourism_project/data/processed/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/data/processed/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/data/processed/ytrain.csv", index=False)
ytest.to_csv("tourism_project/data/processed/ytest.csv", index=False)
print("Train:", Xtrain.shape, " Test:", Xtest.shape)

# 4. Push the processed files back to the Hub
for fname in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project/data/processed/{fname}",
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {fname}",
    )
print("Train and test splits uploaded to the Hugging Face Hub.")
