#!/bin/bash
# Helper script to push E-QFT v5.0 to GitHub

echo "========================================="
echo "E-QFT v5.0 GitHub Push Helper"
echo "========================================="
echo ""
echo "This script will you push the repository to GitHub."
echo ""
echo "STEP 1: Create a new repository on GitHub"
echo "----------------------------------------"
echo "1. Go to: https://github.com/new"
echo "2. Repository name: egravity-v5 (or your preferred name)"
echo "3. Description: Full General Relativity from Emergent Quantum Field Theory"
echo "4. Make it PUBLIC (for open science!)"
echo "5. DO NOT initialize with README, .gitignore, or license"
echo "6. Click 'Create repository'"
echo ""
read -p "Press ENTER when you've created the repository..."

echo ""
echo "STEP 2: Add remote repository"
echo "-----------------------------"
echo "Enter your GitHub username:"
read GITHUB_USERNAME
echo "Enter your repository name (e.g., egravity-v5):"
read REPO_NAME

# Add remote
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

echo ""
echo "Remote added: https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo ""

echo "STEP 3: Push to GitHub"
echo "---------------------"
echo "Pushing main branch..."
git push -u origin main

echo ""
echo "STEP 4: Create release"
echo "---------------------"
echo "Creating v5.0 tag..."
git tag -a v5.0 -m "E-QFT v5.0: Full General Relativity

Major release demonstrating emergence of GR from quantum projector fields.

Key results:
- Lieb-Robinson velocity: v_LR = (0.96 ± 0.01)c
- Lorentz invariance confirmed
- BSSN evolution stable
- Waveforms match NR to 10^-14
- Matter coupling successful

See RELEASE_NOTES.md for details."

echo "Pushing tag..."
git push origin v5.0

echo ""
echo "========================================="
echo "SUCCESS! Repository pushed to GitHub"
echo "========================================="
echo ""
echo "Your repository is now live at:"
echo "https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "Next steps:"
echo "1. Go to your repository on GitHub"
echo "2. Click on 'Releases' → 'Create a new release'"
echo "3. Select tag: v5.0"
echo "4. Release title: 'E-QFT v5.0: Full General Relativity'"
echo "5. Copy content from RELEASE_NOTES.md"
echo "6. Attach publication PDF if available"
echo "7. Click 'Publish release'"
echo ""
echo "For Zenodo archiving:"
echo "1. Go to https://zenodo.org/account/settings/github/"
echo "2. Enable repository: $REPO_NAME"
echo "3. Create new release on GitHub"
echo "4. Zenodo will automatically archive and assign DOI"
echo ""
echo "Remember to update DOIs in:"
echo "- README.md"
echo "- tex/full_gr.tex"
echo "- CITATION.cff"
echo "========================================="