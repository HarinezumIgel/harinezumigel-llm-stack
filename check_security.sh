#!/bin/bash
# Pre-deployment security check
# Scans files for potential secrets before publishing to GitHub

set -e

echo "=================================="
echo "Pre-Deployment Security Scanner"
echo "=================================="
echo

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ISSUES_FOUND=0

echo "Scanning for potential secrets and sensitive data..."
echo

# Check for common secret patterns
echo "1. Checking for API keys..."
if grep -rn "api.*key.*=\s*['\"][a-zA-Z0-9]\{20,\}" --include="*.md" --include="*.py" --include="*.yaml" --include="*.sh" . 2>/dev/null | grep -v "os.environ" | grep -v "example" | grep -v "your-secret" | grep -v "placeholder"; then
    echo -e "${RED}⚠ Potential API keys found!${NC}"
    ((ISSUES_FOUND++))
else
    echo -e "${GREEN}✓ No hardcoded API keys detected${NC}"
fi

echo
echo "2. Checking for passwords..."
if grep -rn "password.*=.*['\"]" --include="*.md" --include="*.py" --include="*.yaml" --include="*.sh" . 2>/dev/null | grep -v "example" | grep -v "placeholder" | grep -v "\-\-password"; then
    echo -e "${RED}⚠ Potential passwords found!${NC}"
    ((ISSUES_FOUND++))
else
    echo -e "${GREEN}✓ No hardcoded passwords detected${NC}"
fi

echo
echo "3. Checking for tokens..."
if grep -rn "token.*=\s*['\"][a-zA-Z0-9]\{20,\}" --include="*.md" --include="*.py" --include="*.yaml" --include="*.sh" . 2>/dev/null | grep -v "example" | grep -v "placeholder" | grep -v "\-\-token" | grep -v "batched_tokens"; then
    echo -e "${RED}⚠ Potential tokens found!${NC}"
    ((ISSUES_FOUND++))
else
    echo -e "${GREEN}✓ No hardcoded tokens detected${NC}"
fi

echo
echo "4. Checking for absolute paths with usernames..."
if grep -rn "/home/[a-z]" --include="*.md" --include="*.py" --include="*.yaml" --include="*.sh" . 2>/dev/null | grep -v "example" | grep -v "/home/igel" | grep -v "/home/user"; then
    echo -e "${YELLOW}⚠ Absolute paths with usernames found${NC}"
    echo -e "  (These may be in examples and could be OK)"
    ((ISSUES_FOUND++))
else
    echo -e "${GREEN}✓ No problematic absolute paths${NC}"
fi

echo
echo "5. Checking for email addresses..."
EMAIL_COUNT=$(grep -ro "[a-zA-Z0-9._%+-]\+@[a-zA-Z0-9.-]\+\.[a-zA-Z]\{2,\}" --include="*.md" --include="*.py" --include="*.yaml" . 2>/dev/null | wc -l)
if [ "$EMAIL_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $EMAIL_COUNT email address(es)${NC}"
    echo -e "  (Review to ensure these are intentional)"
else
    echo -e "${GREEN}✓ No email addresses found${NC}"
fi

echo
echo "6. Checking for IP addresses..."
if grep -rno "[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}" --include="*.md" --include="*.py" --include="*.yaml" --include="*.sh" . 2>/dev/null | grep -v "0.0.0.0" | grep -v "127.0.0.1" | grep -v "localhost"; then
    echo -e "${YELLOW}⚠ IP addresses found (excluding localhost)${NC}"
    echo -e "  (These may be examples and could be OK)"
else
    echo -e "${GREEN}✓ Only localhost IPs detected${NC}"
fi

echo
echo "7. Checking file permissions..."
if [ -x "harinezumigel-llm-stack.py" ] && [ -x "install.sh" ]; then
    echo -e "${GREEN}✓${NC} Script files are executable"
else
    echo -e "${YELLOW}⚠${NC} Some scripts may not be executable"
    echo "  Run: chmod +x harinezumigel-llm-stack.py install.sh"
fi

echo
echo "8. Checking for .env files..."
if [ -f ".env" ]; then
    echo -e "${RED}⚠ .env file present - this should NOT be committed!${NC}"
    ((ISSUES_FOUND++))
else
    echo -e "${GREEN}✓ No .env file (good - only .env.example should exist)${NC}"
fi

echo
echo "9. Checking .gitignore..."
if [ -f ".gitignore" ]; then
    if grep -q "^\.env$" .gitignore; then
        echo -e "${GREEN}✓ .gitignore properly excludes .env files${NC}"
    else
        echo -e "${YELLOW}⚠ .gitignore might not exclude .env files${NC}"
    fi
else
    echo -e "${RED}⚠ No .gitignore file found${NC}"
    ((ISSUES_FOUND++))
fi

echo
echo "10. Checking for required files..."
REQUIRED_FILES=("harinezumigel-llm-stack.py" "README.md" "LICENSE" ".gitignore" ".env.example" "config.yaml.example")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file missing"
        ((ISSUES_FOUND++))
    fi
done

# Summary
echo
echo "=================================="
echo "Scan Complete"
echo "=================================="
echo

if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}✓ No critical issues found!${NC}"
    echo "Your repository appears ready for GitHub publication."
    echo
    echo "Next steps:"
    echo "  1. Review DEPLOY.md for deployment instructions"
    echo "  2. Update URLs with your GitHub username"
    echo "  3. Initialize git and push to GitHub"
    exit 0
else
    echo -e "${YELLOW}⚠ Found $ISSUES_FOUND potential issue(s)${NC}"
    echo
    echo "Please review the warnings above before publishing."
    echo "Some warnings may be acceptable (e.g., examples in documentation)."
    echo
    echo "Critical items to fix:"
    echo "  - Remove any .env files"
    echo "  - Remove hardcoded API keys, passwords, tokens"
    echo "  - Review and sanitize file paths"
    echo
    exit 1
fi
