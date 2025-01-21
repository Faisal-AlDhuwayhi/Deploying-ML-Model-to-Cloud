#!/bin/bash

# Initialize Git in /app
git init
git remote add origin <your-github-repo-url>

# Pull the latest branch
git fetch origin main
git reset --hard origin/main

# Pull DVC data
dvc pull
