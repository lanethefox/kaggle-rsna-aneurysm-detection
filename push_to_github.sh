#!/bin/bash

# Replace YOUR_USERNAME with your GitHub username
# Replace REPO_NAME with your repository name

echo "Setting up GitHub remote..."

# Option 1: Using HTTPS (will prompt for username/password or token)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Option 2: Using SSH (requires SSH key setup)
# git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "Done! Your code is now on GitHub."