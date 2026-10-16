# hosting.py
# Prepares the deployment folder for the Streamlit app and, if the Hugging Face account
# is allowed to run Docker Spaces (needs a PRO plan), pushes it to a Hugging Face Space.
# Without PRO the Space cannot be created, so the app is deployed from the GitHub repo
# with Streamlit Community Cloud instead (the same app.py / requirements.txt are used).
# Run from the repo root:  python tourism_project/deployment/hosting.py

import os
import ast
import sys
import json
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:  # older huggingface_hub versions
    from huggingface_hub.utils import HfHubHTTPError

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
username = api.whoami()["name"]

space_id = f"{username}/tourism-wellness-app"
model_repo = f"{username}/tourism-wellness-model"

# 1. quick checks before deploying anything
with open("tourism_project/deployment/app.py", encoding="utf-8") as f:
    ast.parse(f.read())
print("app.py has no syntax errors")

api.repo_info(repo_id=model_repo, repo_type="model")
print(f"Model repo '{model_repo}' is on the Hub and reachable")

# 2. config.json tells the app which model repo to load
with open("tourism_project/deployment/config.json", "w") as f:
    json.dump({"model_repo": model_repo}, f)
print("config.json written to tourism_project/deployment")

# 3. Hugging Face Space (Docker SDK)
try:
    api.repo_info(repo_id=space_id, repo_type="space")
    print(f"Space '{space_id}' already exists.")
except RepositoryNotFoundError:
    try:
        create_repo(repo_id=space_id, repo_type="space", space_sdk="docker",
                    private=False, token=HF_TOKEN)
        print(f"Space '{space_id}' created.")
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 402:
            print("\nHugging Face needs a PRO plan to host Docker Spaces, so no Space was created.")
            print("The app files (app.py, requirements.txt, config.json) are ready in "
                  "tourism_project/deployment.")
            print("Deploy them from the GitHub repo with Streamlit Community Cloud "
                  "(main file: tourism_project/deployment/app.py).")
            sys.exit(0)
        raise

# README.md with the Space settings (Docker SDK + port used by Streamlit)
readme = """---
title: Wellness Tourism Package Predictor
emoji: 🧳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

Streamlit app that predicts whether a customer will buy the Wellness Tourism Package.
"""
with open("tourism_project/deployment/README.md", "w", encoding="utf-8") as f:
    f.write(readme)

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=space_id,
    repo_type="space",
    commit_message="Deploy Streamlit app",
    ignore_patterns=["hosting.py", "__pycache__/*"],
)
print(f"App pushed. Live at: https://huggingface.co/spaces/{space_id}")
