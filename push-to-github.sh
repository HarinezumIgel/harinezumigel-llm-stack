#!/bin/bash
# Script to push harinezumigel-llm-stack to GitHub
# Run this AFTER deleting the old repo and creating the new one on GitHub

cd "$(dirname "$0")"

echo "Setting up git repository..."

# Commit all staged files
git commit -m "Initial release of harinezumigel-llm-stack v1.0.0"

# Add the new remote (if it doesn't exist)
if git remote | grep -q origin; then
    echo "Removing old origin..."
    git remote remove origin
fi

echo "Adding new remote..."
git remote add origin https://github.com/HarinezumIgel/harinezumigel-llm-stack.git

# Ensure we're on main branch
git branch -M main

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "Done! Your repository is now at:"
echo "https://github.com/HarinezumIgel/harinezumigel-llm-stack"
