# data_register.py
# Registers the raw tourism.csv on the Hugging Face Hub as a dataset repo.
# Run from the repo root:  python tourism_project/model_building/data_register.py

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# Take the username from the token so nothing is hard coded
username = api.whoami()["name"]
repo_id = f"{username}/tourism-wellness-dataset"
repo_type = "dataset"

# Create the dataset repo if it does not exist yet
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating a new one...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=HF_TOKEN)
    print(f"Dataset repo '{repo_id}' created.")

# Upload the raw data folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Register raw tourism dataset",
)
print("Raw data uploaded to the Hugging Face Hub.")
