#!/bin/bash
# Installation script for harinezumigel-llm-stack
# This script sets up the required directory structure and configuration files

set -e

echo "=================================="
echo "harinezumigel-llm-stack Installation Script"
echo "=================================="
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/litellm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}Error: Do not run this script as root${NC}"
   echo "Run as a regular user. The script will use sudo when needed."
   exit 1
fi

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker found"

# Check NVIDIA Docker runtime
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA Docker runtime available"
else
    echo -e "${YELLOW}⚠${NC} NVIDIA Docker runtime not available or no GPUs detected"
    echo "  This is required for GPU acceleration"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Python dependencies
echo
echo "Installing Python dependencies..."

# Check if pyyaml is already available
if python3 -c "import yaml" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} PyYAML already installed"
else
    echo "PyYAML not found. Attempting installation..."

    # Try system package manager first (works on Debian/Ubuntu)
    if command -v apt &> /dev/null; then
        echo "Installing via apt..."
        sudo apt update -qq
        if sudo apt install -y python3-yaml; then
            echo -e "${GREEN}✓${NC} PyYAML installed via apt"
        else
            echo -e "${YELLOW}⚠${NC} Could not install via apt"
        fi
    elif command -v dnf &> /dev/null; then
        echo "Installing via dnf..."
        if sudo dnf install -y python3-pyyaml; then
            echo -e "${GREEN}✓${NC} PyYAML installed via dnf"
        else
            echo -e "${YELLOW}⚠${NC} Could not install via dnf"
        fi
    else
        # Fall back to pip with --break-system-packages (not recommended but functional)
        echo -e "${YELLOW}⚠${NC} No system package manager found"
        echo "Attempting pip installation with --break-system-packages..."
        if pip3 install --user --break-system-packages pyyaml 2>/dev/null; then
            echo -e "${GREEN}✓${NC} PyYAML installed via pip"
        else
            echo -e "${RED}✗${NC} Could not install PyYAML"
            echo "Please install manually:"
            echo "  Ubuntu/Debian: sudo apt install python3-yaml"
            echo "  Fedora/RHEL:   sudo dnf install python3-pyyaml"
            echo "  Or use a venv: python3 -m venv ~/harinezumigel-llm-stack-venv && ~/harinezumigel-llm-stack-venv/bin/pip install pyyaml"
            exit 1
        fi
    fi

    # Verify installation
    if python3 -c "import yaml" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} PyYAML is now available"
    else
        echo -e "${RED}✗${NC} PyYAML installation failed"
        echo "Please install manually before continuing."
        exit 1
    fi
fi

# Create installation directory
echo
echo "Creating installation directory..."
if [[ -d "$INSTALL_DIR" ]]; then
    echo -e "${YELLOW}⚠${NC} Directory $INSTALL_DIR already exists"
    read -p "Continue and potentially overwrite files? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown $USER:$USER "$INSTALL_DIR"
    echo -e "${GREEN}✓${NC} Created $INSTALL_DIR"
fi

# Copy example configuration files
echo
echo "Setting up configuration files..."

if [[ ! -f "$INSTALL_DIR/.env" ]]; then
    cp "$SCRIPT_DIR/.env.example" "$INSTALL_DIR/.env"
    echo -e "${GREEN}✓${NC} Created $INSTALL_DIR/.env"
    echo -e "${YELLOW}  → You must edit this file with your paths${NC}"
else
    echo -e "${YELLOW}⚠${NC} $INSTALL_DIR/.env already exists (not overwritten)"
fi

if [[ ! -f "$INSTALL_DIR/config.yaml" ]]; then
    cp "$SCRIPT_DIR/config.yaml.example" "$INSTALL_DIR/config.yaml"
    echo -e "${GREEN}✓${NC} Created $INSTALL_DIR/config.yaml"
    echo -e "${YELLOW}  → You must edit this file with your models${NC}"
else
    echo -e "${YELLOW}⚠${NC} $INSTALL_DIR/config.yaml already exists (not overwritten)"
fi

# Create symlink for easy access
echo
echo "Creating command symlink..."
SYMLINK_PATH="$HOME/.local/bin/harinezumigel-llm-stack"
mkdir -p "$HOME/.local/bin"

if [[ -L "$SYMLINK_PATH" || -f "$SYMLINK_PATH" ]]; then
    rm -f "$SYMLINK_PATH"
fi

ln -s "$SCRIPT_DIR/harinezumigel-llm-stack.py" "$SYMLINK_PATH"
chmod +x "$SCRIPT_DIR/harinezumigel-llm-stack.py"
echo -e "${GREEN}✓${NC} Created symlink: $SYMLINK_PATH"

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo -e "${YELLOW}⚠${NC} $HOME/.local/bin is not in your PATH"
    echo "  Add this line to your ~/.bashrc or ~/.zshrc:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# Summary
echo
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo
echo "Next steps:"
echo
echo "1. Edit configuration files:"
echo "   - $INSTALL_DIR/.env"
echo "   - $INSTALL_DIR/config.yaml"
echo
echo "2. Set up model directory:"
echo "   - Download your models"
echo "   - Update MODEL_ROOT in .env"
echo
echo "3. Set up LiteLLM virtual environment:"
echo "   - Create venv: python3 -m venv /opt/litellm/venv"
echo "   - Activate: source /opt/litellm/venv/bin/activate"
echo "   - Install: pip install litellm"
echo "   - Update LITELLM_VENV_ACTIVATE in .env"
echo
echo "4. Pull vLLM Docker image:"
echo "   docker pull vllm/vllm-openai:latest"
echo
echo "5. Test installation:"
echo "   harinezumigel-llm-stack --list"
echo
echo "For detailed instructions, see README.md"
echo

# Offer to open editor
read -p "Open .env file for editing now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ${EDITOR:-nano} "$INSTALL_DIR/.env"
fi

echo
echo -e "${GREEN}Installation complete!${NC}"
