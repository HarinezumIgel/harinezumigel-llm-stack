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
echo "  4. Copy example configuration files (env and config.yaml)"
echo "  5. Create a symlink for easy command-line access"
echo "  6. Provide next steps for completing setup"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
TURQUOISE=$'\033[0;36m'
PROMPT_RESET=$'\033[0m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/litellm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNTIME_MODULES=(
    hls_preflight.py
    hls_selection.py
    hls_startup.py
    hls_terminal.py
    harinezumigel-llm-stack.py
)
for module in "${RUNTIME_MODULES[@]}"; do
    if [[ ! -f "$SCRIPT_DIR/src/$module" ]]; then
        echo "Error: Missing $SCRIPT_DIR/src/$module. Install from the complete repository."
        exit 1
    fi
done

for config_file in env.example config.yaml.example; do
    if [[ ! -f "$SCRIPT_DIR/config/$config_file" ]]; then
        echo "Error: Missing $SCRIPT_DIR/config/$config_file. Install from the complete repository."
        exit 1
    fi
done

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
    read -p "${TURQUOISE}Continue anyway? [y/N] ${PROMPT_RESET}" -n 1 -r
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
    echo "  Existing config files will be backed up (with date) before replacement."
    read -p "${TURQUOISE}Continue? [y/N] ${PROMPT_RESET}" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
else
    read -p "${TURQUOISE}Create $INSTALL_DIR? [y/N] ${PROMPT_RESET}" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
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

CONFIG_DIR="$INSTALL_DIR/config"
BACKUP_DIR="$INSTALL_DIR/old_configs"
mkdir -p "$CONFIG_DIR"
CONFIG_BACKUPS=()
BACKED_UP_SOURCES=()

backup_config() {
    local source="$1"
    local backup

    local backed_up_source
    for backed_up_source in "${BACKED_UP_SOURCES[@]}"; do
        if [[ "$backed_up_source" == "$source" ]]; then
            return
        fi
    done

    mkdir -p -m 700 "$BACKUP_DIR"
    backup=$(mktemp "$BACKUP_DIR/$(basename "$source")_$(date +%Y%m%d_%H%M%S).XXXXXX.bak")
    cp "$source" "$backup"
    CONFIG_BACKUPS+=("$backup")
    BACKED_UP_SOURCES+=("$source")
}

report_config_backups() {
    echo "Previous configuration backups:"
    if [[ ${#CONFIG_BACKUPS[@]} -eq 0 ]]; then
        echo "  None; no existing configuration files needed backing up."
        return
    fi
    local backup
    for backup in "${CONFIG_BACKUPS[@]}"; do
        echo "  $backup"
    done
}

# Preserve legacy configurations as backups; initialize the new layout from
# current examples instead of treating an older schema as a ready-to-use config.
for legacy_env in "$CONFIG_DIR/.env" "$INSTALL_DIR/env" "$INSTALL_DIR/.env"; do
    if [[ -f "$legacy_env" ]]; then
        backup_config "$legacy_env"
    fi
done
if [[ -f "$INSTALL_DIR/config.yaml" ]]; then
    backup_config "$INSTALL_DIR/config.yaml"
fi

if [[ -s "$SCRIPT_DIR/config/env.example" ]]; then
    if [[ -s "$CONFIG_DIR/env" ]]; then
        read -p "${TURQUOISE}Copy config/env.example to $CONFIG_DIR/env? [y/N] ${PROMPT_RESET}" -n 1 -r
        echo
    else
        echo "  Missing or empty env; initializing it from config/env.example."
        REPLY=y
    fi
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ -f "$CONFIG_DIR/env" ]]; then
            backup_config "$CONFIG_DIR/env"
        fi
        cp "$SCRIPT_DIR/config/env.example" "$CONFIG_DIR/env"
        echo -e "${GREEN}✓${NC} Created $CONFIG_DIR/env"
        echo "  → You must edit this file with your paths and API keys"
        echo "  → IMPORTANT: Set VLLM_DOCKER_IMAGE to a tag that matches your NVIDIA driver"
        echo "  → Check with: nvidia-smi (Driver Version), then see README.md"
        echo "  → Run: ${YELLOW}nano $CONFIG_DIR/env${NC} (or your preferred editor)"
    else
        echo "  Skipping env."
    fi
else
    echo -e "${RED}Error: config/env.example is missing or empty in $SCRIPT_DIR.${NC}"
    exit 1
fi
echo

if [[ -s "$SCRIPT_DIR/config/config.yaml.example" ]]; then
    if [[ -s "$CONFIG_DIR/config.yaml" ]]; then
        read -p "${TURQUOISE}Copy config/config.yaml.example to $CONFIG_DIR/config.yaml? [y/N] ${PROMPT_RESET}" -n 1 -r
        echo
    else
        echo "  Missing or empty config.yaml; initializing it from config/config.yaml.example."
        REPLY=y
    fi
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ -f "$CONFIG_DIR/config.yaml" ]]; then
            backup_config "$CONFIG_DIR/config.yaml"
        fi
        cp "$SCRIPT_DIR/config/config.yaml.example" "$CONFIG_DIR/config.yaml"
        echo -e "${GREEN}✓${NC} Created $CONFIG_DIR/config.yaml"
        echo "  → You must edit this file to configure your models"
        echo "  → Run: ${YELLOW}nano $CONFIG_DIR/config.yaml${NC} (or your preferred editor)"
    else
        echo "  Skipping config.yaml."
    fi
else
    echo -e "${RED}Error: config/config.yaml.example is missing or empty in $SCRIPT_DIR.${NC}"
    exit 1
fi
echo

# Create symlink for easy access
echo "=================================="
echo "Step 5: Creating Command Symlink"
echo "=================================="
echo
echo "Creating symlink for easy command-line access..."
mkdir -p "$INSTALL_DIR/src"
for module in "${RUNTIME_MODULES[@]}"; do
    if [[ "$SCRIPT_DIR/src/$module" -ef "$INSTALL_DIR/src/$module" ]]; then
        continue
    fi
    install -m 644 "$SCRIPT_DIR/src/$module" "$INSTALL_DIR/src/$module"
done
chmod +x "$INSTALL_DIR/src/harinezumigel-llm-stack.py"
SYMLINK_PATH="$HOME/.local/bin/harinezumigel-llm-stack"
mkdir -p "$HOME/.local/bin"

if [[ -L "$SYMLINK_PATH" ]]; then
    echo "  Removing existing symlink at $SYMLINK_PATH..."
    rm -f "$SYMLINK_PATH"
elif [[ -f "$SYMLINK_PATH" ]]; then
    echo -e "${YELLOW}⚠${NC} A regular file (not a symlink) already exists at $SYMLINK_PATH"
    echo "  This may be an existing installation or an unrelated file."
    read -p "${TURQUOISE}  Delete it and replace with a symlink? [y/N] ${PROMPT_RESET}" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please remove $SYMLINK_PATH manually and re-run."
        exit 1
    fi
    rm -f "$SYMLINK_PATH"
fi

echo "  Creating symlink: $INSTALL_DIR/src/harinezumigel-llm-stack.py → $SYMLINK_PATH"
ln -s "$INSTALL_DIR/src/harinezumigel-llm-stack.py" "$SYMLINK_PATH"
echo -e "${GREEN}✓${NC} Created symlink: $SYMLINK_PATH"
echo "  Runtime modules are installed in $INSTALL_DIR/src; the checkout is not needed to run the command."
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
echo "Step 6: LiteLLM Virtual Environment Setup"
echo "=================================="
echo
echo "LiteLLM requires a Python virtual environment with 'litellm[proxy]' and prisma."

VENV_PATH="$INSTALL_DIR/venv"

install_litellm_proxy_deps() {
    local venv_path="$1"
    local mode="$2"
    local pip_bin="$venv_path/bin/pip"
    local check_output

    if [[ "$mode" == "upgrade" ]]; then
        if ! "$pip_bin" install --upgrade "litellm[proxy]"; then
            return 1
        fi
        echo -e "${GREEN}✓${NC} litellm[proxy] installed/upgraded successfully"
    else
        if ! "$pip_bin" install "litellm[proxy]"; then
            return 1
        fi
        echo -e "${GREEN}✓${NC} litellm[proxy] installed successfully"
    fi

    # prisma is imported by LiteLLM auth error handlers; ensure it exists
    # to avoid ModuleNotFoundError during malformed key/auth paths.
    if "$pip_bin" show prisma >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} prisma is already installed"
    else
        if ! "$pip_bin" install prisma; then
            echo -e "${RED}✗${NC} Failed to install prisma"
            return 1
        fi
        echo -e "${GREEN}✓${NC} prisma installed successfully"
    fi

    # Keep dependency resolution owned by litellm[proxy]; explicit pinning here
    # can create artificial conflicts as upstream dependencies evolve.
    if check_output=$("$pip_bin" check 2>&1); then
        echo -e "${GREEN}✓${NC} LiteLLM proxy dependency check passed"
    else
        echo -e "${YELLOW}⚠ Dependency check reported issues in $venv_path${NC}"
        echo "$check_output"
        echo -e "${YELLOW}  If this venv was reused, consider recreating $venv_path and running install.sh again.${NC}"
    fi

    return 0
}

CREATE_VENV=false
if [[ -d "$VENV_PATH" ]]; then
    echo
    echo -e "${GREEN}✓${NC} Found existing virtual environment at $VENV_PATH"
    echo
    echo "This step will:"
    echo "  • Use the existing virtual environment"
    echo "  • Install/upgrade litellm[proxy] package (this may take a few minutes)"
    echo "  • Configure the env file with the venv activation path"
    echo
    read -p "${TURQUOISE}Use existing virtual environment and install/upgrade litellm? [y/N] ${PROMPT_RESET}" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo
        echo "Using existing virtual environment at $VENV_PATH..."
        echo
        echo "Installing/upgrading litellm[proxy] package..."
        echo "This may take a few minutes. Please wait..."
        echo
        if ! install_litellm_proxy_deps "$VENV_PATH" "upgrade"; then
            echo -e "${RED}Error: Failed to upgrade LiteLLM. Setup aborted.${NC}"
            exit 1
        fi
    else
        echo "Reinstalling will replace $VENV_PATH; the old environment will be retained."
        read -p "${TURQUOISE}Reinstall the LiteLLM virtual environment? [y/N] (No aborts setup) ${PROMPT_RESET}" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted. The existing virtual environment remains unchanged."
            exit 1
        fi
        VENV_BACKUP=$(mktemp -d "$INSTALL_DIR/venv_backup_$(date +%Y%m%d_%H%M%S).XXXXXX")
        mv "$VENV_PATH" "$VENV_BACKUP/venv"
        echo "Previous environment retained at $VENV_BACKUP/venv"
        CREATE_VENV=true
    fi
else
    echo
    echo "No existing virtual environment found."
    echo
    echo "This step will:"
    echo "  • Create a new virtual environment at $VENV_PATH"
    echo "  • Install litellm[proxy] package (this may take a few minutes)"
    echo "  • Configure the env file with the venv activation path"
    echo
    echo "Declining creation will abort setup."
    echo
    read -p "${TURQUOISE}Create LiteLLM virtual environment now? [y/N] ${PROMPT_RESET}" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        CREATE_VENV=true
    else
        echo "Aborted. A LiteLLM virtual environment is required."
        exit 1
    fi
fi

if [[ "$CREATE_VENV" == true ]]; then
    echo "Creating virtual environment at $VENV_PATH..."
    if ! python3 -m venv "$VENV_PATH"; then
        echo -e "${RED}Error: Failed to create virtual environment. Setup aborted.${NC}"
        exit 1
    fi
    if ! install_litellm_proxy_deps "$VENV_PATH" "install"; then
        echo -e "${RED}Error: Failed to install LiteLLM. Setup aborted.${NC}"
        exit 1
    fi
fi

echo "Updating env with virtual environment paths..."
ENV_PATHS_CHANGE=false
for setting in LITELLM_VENV_ACTIVATE LITELLM_BIN; do
    if [[ "$setting" == LITELLM_VENV_ACTIVATE ]]; then
        value="$VENV_PATH/bin/activate"
    else
        value="$VENV_PATH/bin/litellm"
    fi
    if ! grep -Fqx "$setting=$value" "$CONFIG_DIR/env"; then
        ENV_PATHS_CHANGE=true
    fi
done
if [[ "$ENV_PATHS_CHANGE" == true ]]; then
    backup_config "$CONFIG_DIR/env"
fi
for setting in LITELLM_VENV_ACTIVATE LITELLM_BIN; do
    if [[ "$setting" == LITELLM_VENV_ACTIVATE ]]; then
        value="$VENV_PATH/bin/activate"
    else
        value="$VENV_PATH/bin/litellm"
    fi
    if grep -q "^$setting=" "$CONFIG_DIR/env"; then
        sed -i "s|^$setting=.*|$setting=$value|" "$CONFIG_DIR/env"
    else
        printf '%s=%s\n' "$setting" "$value" >> "$CONFIG_DIR/env"
    fi
done
echo -e "${GREEN}✓${NC} config/env updated with activation and LiteLLM executable paths"
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
echo "   • $CONFIG_DIR/env      (paths, API keys, Docker image)"
echo "   • $CONFIG_DIR/config.yaml (model definitions)"
echo "   • Ensure VLLM_DOCKER_IMAGE is compatible with your NVIDIA driver"
echo
echo "2. Set up model directory:"
echo "   • Download your models to a directory"
echo "   • Update MODEL_ROOT in env to point to that directory"
echo
if [[ ! -d "$INSTALL_DIR/venv" ]]; then
echo "3. Set up LiteLLM virtual environment (skipped above):"
echo "   • Create venv: python3 -m venv /opt/litellm/venv"
echo "   • Activate:      source /opt/litellm/venv/bin/activate"
echo "   • Install:       pip install 'litellm[proxy]' prisma"
echo "   • Update LITELLM_VENV_ACTIVATE in env"
echo
fi
VLLM_IMAGE=$(grep '^VLLM_DOCKER_IMAGE=' "$CONFIG_DIR/env" 2>/dev/null | cut -d= -f2)
VLLM_IMAGE=${VLLM_IMAGE:-nvcr.io/nvidia/vllm:latest}
echo "4. Verify NVIDIA driver and image compatibility:"
echo "   nvidia-smi | grep 'Driver Version'"
echo "   grep '^VLLM_DOCKER_IMAGE=' $CONFIG_DIR/env"
echo
echo "5. Pull vLLM Docker image:"
echo "   docker pull $VLLM_IMAGE"
echo "   (This may take a few minutes depending on your connection)"
echo
echo "6. Test installation:"
echo "   harinezumigel-llm-stack --list"
echo
echo "For detailed instructions, see README.md"
echo
echo "=================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=================================="
echo

open_config_in_editor() {
    local target_file="$1"

    if [[ ! -s "$target_file" ]]; then
        echo -e "${RED}Error: Configuration file is missing or empty: $target_file${NC}"
        echo "Re-run install.sh to initialize it from the example configuration."
        return 1
    fi

    if [[ "${EDITOR:-nano}" == "nano" ]]; then
        nano "$target_file"
    else
        ${EDITOR:-nano} "$target_file"
    fi
}

# Offer to open editor
echo "Would you like to edit the configuration files now?"
echo -e "${YELLOW}⚠ Note: the next file is config/env (no leading dot), not venv or .venv.${NC}"
echo
read -p "${TURQUOISE}Open env file for editing? [y/N] ${PROMPT_RESET}" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open_config_in_editor "$CONFIG_DIR/env"
fi

echo
read -p "${TURQUOISE}Open config.yaml file for editing? [y/N] ${PROMPT_RESET}" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open_config_in_editor "$CONFIG_DIR/config.yaml"
fi

echo
if [[ ! -s "$CONFIG_DIR/env" || ! -s "$CONFIG_DIR/config.yaml" ]]; then
    echo -e "${RED}Error: Configuration files are missing or empty. Setup is incomplete.${NC}"
    exit 1
fi
echo -e "${GREEN}Setup complete!${NC}"
echo "Next: Edit configs, download models, pull Docker image, then run 'harinezumigel-llm-stack --list'"
echo
echo -e "${RED}Did you update $CONFIG_DIR/env to match your LLM setup?${NC}"
echo -e "${RED}Did you replace the example LITELLM_BIND_HOST=192.168.100.50 in $CONFIG_DIR/env with an IP assigned to your target host (or confirm it is already correct)?${NC}"
echo -e "${RED}Did you configure $CONFIG_DIR/config.yaml for your models?${NC}"
echo
report_config_backups
