#!/bin/bash
# Installation script for harinezumigel-llm-stack
# This script sets up the required directory structure and configuration files

set -e

echo "=================================="
echo "harinezumigel-llm-stack Installation Script"
echo "=================================="
echo
echo "This script will:"
echo "  1. Check that Python 3, Docker, and NVIDIA GPU are installed"
echo "  2. Install PyYAML (required for YAML config parsing)"
echo "  3. Create /opt/litellm directory structure"
echo "  4. Copy example configuration files (.env and config.yaml)"
echo "  5. Create a symlink for easy command-line access"
echo "  6. Provide next steps for completing setup"
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
echo "=================================="
echo "Step 1: Checking Prerequisites"
echo "=================================="
echo

# Check Python
echo "Checking Python 3 installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Error: Python 3 is not installed${NC}"
    echo "  Please install Python 3.10 or higher first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
echo

# Check Docker
echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Error: Docker is not installed${NC}"
    echo "  Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker found"

# Verify Docker daemon is running
echo "Checking Docker daemon status..."
if docker info &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker daemon is running"
else
    echo -e "${YELLOW}⚠${NC} Docker is installed but daemon may not be running"
    echo "  Please start Docker and try again."
    exit 1
fi
echo

# Check NVIDIA GPU via nvidia-smi (no Docker pull required)
echo "Checking NVIDIA GPU availability..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} NVIDIA GPU(s) detected ($GPU_COUNT found)"
    echo "  GPU information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | while read line; do
        echo "    $line"
    done
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found or no GPUs detected"
    echo "  This is required for GPU acceleration with vLLM models"
    echo "  Without GPU, you can only run CPU-based models (very slow)"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please install NVIDIA drivers and nvidia-smi first."
        exit 1
    fi
fi
echo

# Install Python dependencies
echo "=================================="
echo "Step 2: Installing Python Dependencies"
echo "=================================="
echo
echo "Checking for PyYAML (required for YAML configuration parsing)..."

# Check if pyyaml is already available
if python3 -c "import yaml" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} PyYAML is already installed"
else
    echo -e "${YELLOW}⚠${NC} PyYAML not found. Installing now..."
    echo

    # Try system package manager first (works on Debian/Ubuntu)
    if command -v apt &> /dev/null; then
        echo "  Attempting to install via apt (Debian/Ubuntu)..."
        echo "  Running: sudo apt update -qq && sudo apt install -y python3-yaml"
        sudo apt update -qq
        if sudo apt install -y python3-yaml; then
            echo -e "${GREEN}✓${NC} PyYAML installed successfully via apt"
        else
            echo -e "${YELLOW}⚠${NC} Could not install via apt"
        fi
    elif command -v dnf &> /dev/null; then
        echo "  Attempting to install via dnf (Fedora/RHEL)..."
        echo "  Running: sudo dnf install -y python3-pyyaml"
        if sudo dnf install -y python3-pyyaml; then
            echo -e "${GREEN}✓${NC} PyYAML installed successfully via dnf"
        else
            echo -e "${YELLOW}⚠${NC} Could not install via dnf"
        fi
    else
        # Fall back to pip with --break-system-packages (not recommended but functional)
        echo -e "${YELLOW}⚠${NC} No system package manager (apt/dnf) found"
        echo "  Attempting pip installation with --break-system-packages..."
        if pip3 install --user --break-system-packages pyyaml 2>/dev/null; then
            echo -e "${GREEN}✓${NC} PyYAML installed via pip"
        else
            echo -e "${RED}✗${NC} Could not install PyYAML"
            echo
            echo "  Please install manually:"
            echo "    Ubuntu/Debian: sudo apt install python3-yaml"
            echo "    Fedora/RHEL:   sudo dnf install python3-pyyaml"
            echo "    Or use a venv: python3 -m venv ~/harinezumigel-llm-stack-venv && ~/harinezumigel-llm-stack-venv/bin/pip install pyyaml"
            exit 1
        fi
    fi
    echo

    # Verify installation
    echo "Verifying PyYAML installation..."
    if python3 -c "import yaml" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} PyYAML is now available and working"
    else
        echo -e "${RED}✗${NC} PyYAML installation failed"
        echo "  Please install manually before continuing."
        exit 1
    fi
fi
echo

# Create installation directory
echo "=================================="
echo "Step 3: Creating Installation Directory"
echo "=================================="
echo
echo "Setting up /opt/litellm directory structure..."
if [[ -d "$INSTALL_DIR" ]]; then
    echo -e "${YELLOW}⚠${NC} Directory $INSTALL_DIR already exists"
    echo "  Existing files will be preserved, but configuration files will not be overwritten."
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "  Creating $INSTALL_DIR..."
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown $USER:$USER "$INSTALL_DIR"
    echo -e "${GREEN}✓${NC} Created $INSTALL_DIR"
fi
echo

# Copy example configuration files
echo "=================================="
echo "Step 4: Setting Up Configuration Files"
echo "=================================="
echo
echo "Copying example configuration files to $INSTALL_DIR..."
echo

if [[ ! -f "$INSTALL_DIR/.env" ]]; then
    echo "  Creating .env file from example..."
    cp "$SCRIPT_DIR/.env.example" "$INSTALL_DIR/.env"
    echo -e "${GREEN}✓${NC} Created $INSTALL_DIR/.env"
    echo "  → You must edit this file with your paths and API keys"
    echo "  → Run: ${YELLOW}nano $INSTALL_DIR/.env${NC} (or your preferred editor)"
else
    echo -e "${GREEN}✓${NC} $INSTALL_DIR/.env already exists (not overwritten)"
    echo "  → Your existing configuration is preserved"
fi
echo

if [[ ! -f "$INSTALL_DIR/config.yaml" ]]; then
    echo "  Creating config.yaml file from example..."
    cp "$SCRIPT_DIR/config.yaml.example" "$INSTALL_DIR/config.yaml"
    echo -e "${GREEN}✓${NC} Created $INSTALL_DIR/config.yaml"
    echo "  → You must edit this file to configure your models"
    echo "  → Run: ${YELLOW}nano $INSTALL_DIR/config.yaml${NC} (or your preferred editor)"
else
    echo -e "${GREEN}✓${NC} $INSTALL_DIR/config.yaml already exists (not overwritten)"
    echo "  → Your existing configuration is preserved"
fi
echo

# Create symlink for easy access
echo "=================================="
echo "Step 5: Creating Command Symlink"
echo "=================================="
echo
echo "Creating symlink for easy command-line access..."
SYMLINK_PATH="$HOME/.local/bin/harinezumigel-llm-stack"
mkdir -p "$HOME/.local/bin"

if [[ -L "$SYMLINK_PATH" ]]; then
    echo "  Removing existing symlink at $SYMLINK_PATH..."
    rm -f "$SYMLINK_PATH"
elif [[ -f "$SYMLINK_PATH" ]]; then
    echo -e "${YELLOW}⚠${NC} A regular file (not a symlink) already exists at $SYMLINK_PATH"
    echo "  This may be an existing installation or an unrelated file."
    read -p "  Delete it and replace with a symlink? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please remove $SYMLINK_PATH manually and re-run."
        exit 1
    fi
    rm -f "$SYMLINK_PATH"
fi

echo "  Creating symlink: $SCRIPT_DIR/harinezumigel-llm-stack.py → $SYMLINK_PATH"
ln -s "$SCRIPT_DIR/harinezumigel-llm-stack.py" "$SYMLINK_PATH"
chmod +x "$SCRIPT_DIR/harinezumigel-llm-stack.py"
echo -e "${GREEN}✓${NC} Created symlink: $SYMLINK_PATH"
echo

# Check if ~/.local/bin is in PATH
echo "Checking if ~/.local/bin is in your PATH..."
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo -e "${YELLOW}⚠${NC} $HOME/.local/bin is not in your PATH"
    echo "  To use the 'harinezumigel-llm-stack' command, add this line to your shell config:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "  Then run: source ~/.bashrc  (or source ~/.zshrc)"
    echo
else
    echo -e "${GREEN}✓${NC} $HOME/.local/bin is already in your PATH"
fi
echo

# Create LiteLLM virtual environment
echo "=================================="
echo "Step 6: LiteLLM Virtual Environment Setup (Optional)"
echo "=================================="
echo
echo "LiteLLM requires a Python virtual environment with the 'litellm[proxy]' package."

VENV_PATH="$INSTALL_DIR/venv"

if [[ -d "$VENV_PATH" ]]; then
    echo
    echo -e "${GREEN}✓${NC} Found existing virtual environment at $VENV_PATH"
    echo
    echo "This step will:"
    echo "  • Use the existing virtual environment"
    echo "  • Install/upgrade litellm[proxy] package (this may take a few minutes)"
    echo "  • Configure the .env file with the venv activation path"
    echo
    read -p "Use existing virtual environment and install/upgrade litellm? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo
        echo "Using existing virtual environment at $VENV_PATH..."
        echo
        echo "Installing/upgrading litellm[proxy] package..."
        echo "This may take a few minutes. Please wait..."
        echo
        if "$VENV_PATH/bin/pip" install --upgrade 'litellm[proxy]'; then
            echo -e "${GREEN}✓${NC} litellm[proxy] installed/upgraded successfully"
            echo
            echo "Updating .env with venv activation path..."
            if grep -q "^LITELLM_VENV_ACTIVATE=" "$INSTALL_DIR/.env" 2>/dev/null; then
                sed -i.bak "s|^LITELLM_VENV_ACTIVATE=.*|LITELLM_VENV_ACTIVATE=$VENV_PATH/bin/activate|" "$INSTALL_DIR/.env" \
                    && rm -f "$INSTALL_DIR/.env.bak"
            else
                echo "LITELLM_VENV_ACTIVATE=$VENV_PATH/bin/activate" >> "$INSTALL_DIR/.env"
            fi
            echo -e "${GREEN}✓${NC} .env file updated with venv path"
        else
            echo -e "${RED}✗${NC} Failed to install/upgrade litellm[proxy]"
            echo "  You can install it manually later with:"
            echo "  source $VENV_PATH/bin/activate && pip install --upgrade 'litellm[proxy]'"
        fi
    else
        echo "Skipping virtual environment setup."
        echo "The existing virtual environment at $VENV_PATH will remain unchanged."
    fi
else
    echo
    echo "No existing virtual environment found."
    echo
    echo "This step will:"
    echo "  • Create a new virtual environment at $VENV_PATH"
    echo "  • Install litellm[proxy] package (this may take a few minutes)"
    echo "  • Configure the .env file with the venv activation path"
    echo
    echo "You can skip this step and set it up manually later if you prefer."
    echo
    read -p "Create LiteLLM virtual environment now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo
        echo "Creating virtual environment at $VENV_PATH..."
        if python3 -m venv "$VENV_PATH"; then
            echo -e "${GREEN}✓${NC} Virtual environment created successfully"
            echo
            echo "Installing litellm[proxy] package..."
            echo "This may take a few minutes. Please wait..."
            echo
            if "$VENV_PATH/bin/pip" install 'litellm[proxy]'; then
                echo -e "${GREEN}✓${NC} litellm[proxy] installed successfully"
                echo
                echo "Updating .env with venv activation path..."
                if grep -q "^LITELLM_VENV_ACTIVATE=" "$INSTALL_DIR/.env" 2>/dev/null; then
                    sed -i.bak "s|^LITELLM_VENV_ACTIVATE=.*|LITELLM_VENV_ACTIVATE=$VENV_PATH/bin/activate|" "$INSTALL_DIR/.env" \
                        && rm -f "$INSTALL_DIR/.env.bak"
                else
                    echo "LITELLM_VENV_ACTIVATE=$VENV_PATH/bin/activate" >> "$INSTALL_DIR/.env"
                fi
                echo -e "${GREEN}✓${NC} .env file updated with venv path"
            else
                echo -e "${RED}✗${NC} Failed to install litellm[proxy]"
                echo "  You can install it manually later with:"
                echo "  source $VENV_PATH/bin/activate && pip install 'litellm[proxy]'"
            fi
        else
            echo -e "${RED}✗${NC} Failed to create virtual environment"
            echo "  You can create it manually later with:"
            echo "  python3 -m venv $VENV_PATH"
        fi
    else
        echo "Skipping virtual environment setup."
        echo "You can create it manually later (see next steps below)."
    fi
fi
echo

# Summary
echo "=================================="
echo "Setup Summary"
echo "=================================="
echo
echo "The harinezumigel-llm-stack has been installed successfully."
echo "You still need to complete the following setup steps:"
echo
echo "1. Edit configuration files:"
echo "   • $INSTALL_DIR/.env      (paths, API keys, Docker image)"
echo "   • $INSTALL_DIR/config.yaml (model definitions)"
echo
echo "2. Set up model directory:"
echo "   • Download your models to a directory"
echo "   • Update MODEL_ROOT in .env to point to that directory"
echo
if [[ ! -d "$INSTALL_DIR/venv" ]]; then
echo "3. Set up LiteLLM virtual environment (skipped above):"
echo "   • Create venv: python3 -m venv /opt/litellm/venv"
echo "   • Activate:      source /opt/litellm/venv/bin/activate"
echo "   • Install:       pip install 'litellm[proxy]'"
echo "   • Update LITELLM_VENV_ACTIVATE in .env"
echo
fi
VLLM_IMAGE=$(grep '^VLLM_DOCKER_IMAGE=' "$INSTALL_DIR/.env" 2>/dev/null | cut -d= -f2)
VLLM_IMAGE=${VLLM_IMAGE:-nvcr.io/nvidia/vllm:latest}
echo "4. Pull vLLM Docker image:"
echo "   docker pull $VLLM_IMAGE"
echo "   (This may take a few minutes depending on your connection)"
echo
echo "5. Test installation:"
echo "   harinezumigel-llm-stack --list"
echo
echo "For detailed instructions, see README.md"
echo
echo "=================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=================================="
echo

# Offer to open editor
echo "Would you like to edit the configuration files now?"
echo
read -p "Open .env file for editing? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ${EDITOR:-nano} "$INSTALL_DIR/.env"
fi

echo
read -p "Open config.yaml file for editing? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ${EDITOR:-nano} "$INSTALL_DIR/config.yaml"
fi

echo
echo -e "${GREEN}Setup complete!${NC}"
echo "Next: Edit configs, download models, pull Docker image, then run 'harinezumigel-llm-stack --list'"
echo
