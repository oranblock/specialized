#!/bin/bash
# Script to create a new GitHub repository and push the project

# Configuration - modify these variables
REPO_NAME="specialized"
REPO_DESCRIPTION="A modular, component-based system for forex pattern detection and prediction"
GITHUB_USERNAME="oranblock"  # Fill in your GitHub username
VISIBILITY="public"  # or "private" for a private repository

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Creating GitHub repository for Forex Spike Predictor...${NC}"

# Check if GitHub username is set
if [ -z "$GITHUB_USERNAME" ]; then
    echo -e "${RED}ERROR: Please edit this script and set your GitHub username.${NC}"
    exit 1
fi

# Create the repository using GitHub CLI if available
if command -v gh &> /dev/null; then
    echo -e "${GREEN}GitHub CLI found, using it to create repository...${NC}"
    gh repo create "$REPO_NAME" --description "$REPO_DESCRIPTION" --"$VISIBILITY"
else
    echo -e "${YELLOW}GitHub CLI not found. Creating repository via API...${NC}"
    echo -e "${RED}Manual creation required. Please create a repository named '$REPO_NAME' on GitHub.${NC}"
    echo -e "${YELLOW}Then press any key to continue with pushing the code...${NC}"
    read -n 1 -s
fi

# Initialize Git repository if not already
if [ ! -d ".git" ]; then
    echo -e "${GREEN}Initializing Git repository...${NC}"
    git init
fi

# Add all files
echo -e "${GREEN}Adding files to Git...${NC}"
git add .

# Commit changes
echo -e "${GREEN}Committing files...${NC}"
git commit -m "Initial commit of Forex Spike Predictor"

# Add remote
echo -e "${GREEN}Adding GitHub remote...${NC}"
git remote add origin "git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"

# Push to GitHub
echo -e "${GREEN}Pushing to GitHub...${NC}"
git push -u origin master || git push -u origin main

echo -e "${GREEN}Done! Your project has been pushed to GitHub.${NC}"
echo -e "${GREEN}Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME${NC}"