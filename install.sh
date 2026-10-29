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
echo "  6. Download and install hash-pinned LiteLLM in /opt/litellm/venv"
echo "  7. Provide next steps for completing setup"
echo
echo "Validated target: Linux/glibc arm64 and amd64. Other hosts may need compatible"
echo "PyYAML/LiteLLM artifacts and an NVIDIA vLLM image; see README.md."
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
TURQUOISE=$'\033[0;36m'
PROMPT_RESET=$'\033[0m'
NC='\033[0m' # No Color

pyyaml_requirements() {
    # PyPI non-yanked wheel SHA-256s retrieved 2026-10-28; review updates as a set.
    cat <<'PYYAML_REQUIREMENTS'
pyyaml==6.0.3 \
    --hash=sha256:00c4bdeba853cc34e7dd471f16b4114f4162dc03e6b7afcc2128711f0eca823c \
    --hash=sha256:0150219816b6a1fa26fb4699fb7daa9caf09eb1999f3b70fb6e786805e80375a \
    --hash=sha256:02893d100e99e03eda1c8fd5c441d8c60103fd175728e23e431db1b589cf5ab3 \
    --hash=sha256:02ea2dfa234451bbb8772601d7b8e426c2bfa197136796224e50e35a78777956 \
    --hash=sha256:0f29edc409a6392443abf94b9cf89ce99889a1dd5376d94316ae5145dfedd5d6 \
    --hash=sha256:10892704fc220243f5305762e276552a0395f7beb4dbf9b14ec8fd43b57f126c \
    --hash=sha256:16249ee61e95f858e83976573de0f5b2893b3677ba71c9dd36b9cf8be9ac6d65 \
    --hash=sha256:1d37d57ad971609cf3c53ba6a7e365e40660e3be0e5175fa9f2365a379d6095a \
    --hash=sha256:1ebe39cb5fc479422b83de611d14e2c0d3bb2a18bbcb01f229ab3cfbd8fee7a0 \
    --hash=sha256:214ed4befebe12df36bcc8bc2b64b396ca31be9304b8f59e25c11cf94a4c033b \
    --hash=sha256:2283a07e2c21a2aa78d9c4442724ec1eb15f5e42a723b99cb3d822d48f5f7ad1 \
    --hash=sha256:22ba7cfcad58ef3ecddc7ed1db3409af68d023b7f940da23c6c2a1890976eda6 \
    --hash=sha256:27c0abcb4a5dac13684a37f76e701e054692a9b2d3064b70f5e4eb54810553d7 \
    --hash=sha256:28c8d926f98f432f88adc23edf2e6d4921ac26fb084b028c733d01868d19007e \
    --hash=sha256:2e71d11abed7344e42a8849600193d15b6def118602c4c176f748e4583246007 \
    --hash=sha256:34d5fcd24b8445fadc33f9cf348c1047101756fd760b4dacb5c3e99755703310 \
    --hash=sha256:37503bfbfc9d2c40b344d06b2199cf0e96e97957ab1c1b546fd4f87e53e5d3e4 \
    --hash=sha256:3c5677e12444c15717b902a5798264fa7909e41153cdf9ef7ad571b704a63dd9 \
    --hash=sha256:3ff07ec89bae51176c0549bc4c63aa6202991da2d9a6129d7aef7f1407d3f295 \
    --hash=sha256:41715c910c881bc081f1e8872880d3c650acf13dfa8214bad49ed4cede7c34ea \
    --hash=sha256:418cf3f2111bc80e0933b2cd8cd04f286338bb88bdc7bc8e6dd775ebde60b5e0 \
    --hash=sha256:44edc647873928551a01e7a563d7452ccdebee747728c1080d881d68af7b997e \
    --hash=sha256:4a2e8cebe2ff6ab7d1050ecd59c25d4c8bd7e6f400f5f82b96557ac0abafd0ac \
    --hash=sha256:4ad1906908f2f5ae4e5a8ddfce73c320c2a1429ec52eafd27138b7f1cbe341c9 \
    --hash=sha256:501a031947e3a9025ed4405a168e6ef5ae3126c59f90ce0cd6f2bfc477be31b7 \
    --hash=sha256:5190d403f121660ce8d1d2c1bb2ef1bd05b5f68533fc5c2ea899bd15f4399b35 \
    --hash=sha256:5498cd1645aa724a7c71c8f378eb29ebe23da2fc0d7a08071d89469bf1d2defb \
    --hash=sha256:5cf4e27da7e3fbed4d6c3d8e797387aaad68102272f8f9752883bc32d61cb87b \
    --hash=sha256:5e0b74767e5f8c593e8c9b5912019159ed0533c70051e9cce3e8b6aa699fcd69 \
    --hash=sha256:5ed875a24292240029e4483f9d4a4b8a1ae08843b9c54f43fcc11e404532a8a5 \
    --hash=sha256:5fcd34e47f6e0b794d17de1b4ff496c00986e1c83f7ab2fb8fcfe9616ff7477b \
    --hash=sha256:5fdec68f91a0c6739b380c83b951e2c72ac0197ace422360e6d5a959d8d97b2c \
    --hash=sha256:6344df0d5755a2c9a276d4473ae6b90647e216ab4757f8426893b5dd2ac3f369 \
    --hash=sha256:64386e5e707d03a7e172c0701abfb7e10f0fb753ee1d773128192742712a98fd \
    --hash=sha256:652cb6edd41e718550aad172851962662ff2681490a8a711af6a4d288dd96824 \
    --hash=sha256:66291b10affd76d76f54fad28e22e51719ef9ba22b29e1d7d03d6777a9174198 \
    --hash=sha256:66e1674c3ef6f541c35191caae2d429b967b99e02040f5ba928632d9a7f0f065 \
    --hash=sha256:6adc77889b628398debc7b65c073bcb99c4a0237b248cacaf3fe8a557563ef6c \
    --hash=sha256:79005a0d97d5ddabfeeea4cf676af11e647e41d81c9a7722a193022accdb6b7c \
    --hash=sha256:7c6610def4f163542a622a73fb39f534f8c101d690126992300bf3207eab9764 \
    --hash=sha256:7f047e29dcae44602496db43be01ad42fc6f1cc0d8cd6c83d342306c32270196 \
    --hash=sha256:8098f252adfa6c80ab48096053f512f2321f0b998f98150cea9bd23d83e1467b \
    --hash=sha256:850774a7879607d3a6f50d36d04f00ee69e7fc816450e5f7e58d7f17f1ae5c00 \
    --hash=sha256:8d1fab6bb153a416f9aeb4b8763bc0f22a5586065f86f7664fc23339fc1c1fac \
    --hash=sha256:8da9669d359f02c0b91ccc01cac4a67f16afec0dac22c2ad09f46bee0697eba8 \
    --hash=sha256:8dc52c23056b9ddd46818a57b78404882310fb473d63f17b07d5c40421e47f8e \
    --hash=sha256:9149cad251584d5fb4981be1ecde53a1ca46c891a79788c0df828d2f166bda28 \
    --hash=sha256:93dda82c9c22deb0a405ea4dc5f2d0cda384168e466364dec6255b293923b2f3 \
    --hash=sha256:96b533f0e99f6579b3d4d4995707cf36df9100d67e0c8303a0c55b27b5f99bc5 \
    --hash=sha256:9c57bb8c96f6d1808c030b1687b9b5fb476abaa47f0db9c0101f5e9f394e97f4 \
    --hash=sha256:9c7708761fccb9397fe64bbc0395abcae8c4bf7b0eac081e12b809bf47700d0b \
    --hash=sha256:9f3bfb4965eb874431221a3ff3fdcddc7e74e3b07799e0e84ca4a0f867d449bf \
    --hash=sha256:a33284e20b78bd4a18c8c2282d549d10bc8408a2a7ff57653c0cf0b9be0afce5 \
    --hash=sha256:a80cb027f6b349846a3bf6d73b5e95e782175e52f22108cfa17876aaeff93702 \
    --hash=sha256:b30236e45cf30d2b8e7b3e85881719e98507abed1011bf463a8fa23e9c3e98a8 \
    --hash=sha256:b3bc83488de33889877a0f2543ade9f70c67d66d9ebb4ac959502e12de895788 \
    --hash=sha256:b865addae83924361678b652338317d1bd7e79b1f4596f96b96c77a5a34b34da \
    --hash=sha256:b8bb0864c5a28024fac8a632c443c87c5aa6f215c0b126c449ae1a150412f31d \
    --hash=sha256:ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc \
    --hash=sha256:bdb2c67c6c1390b63c6ff89f210c8fd09d9a1217a465701eac7316313c915e4c \
    --hash=sha256:c1ff362665ae507275af2853520967820d9124984e0f7466736aea23d8611fba \
    --hash=sha256:c2514fceb77bc5e7a2f7adfaa1feb2fb311607c9cb518dbc378688ec73d8292f \
    --hash=sha256:c3355370a2c156cffb25e876646f149d5d68f5e0a3ce86a5084dd0b64a994917 \
    --hash=sha256:c458b6d084f9b935061bc36216e8a69a7e293a2f1e68bf956dcd9e6cbcd143f5 \
    --hash=sha256:d0eae10f8159e8fdad514efdc92d74fd8d682c933a6dd088030f3834bc8e6b26 \
    --hash=sha256:ebc55a14a21cb14062aa4162f906cd962b28e2e9ea38f9b4391244cd8de4ae0b \
    --hash=sha256:eda16858a3cab07b80edaf74336ece1f986ba330fdb8ee0d6c0d68fe82bc96be \
    --hash=sha256:ee2922902c45ae8ccada2c5b501ab86c36525b883eff4255313a253a3160861c \
    --hash=sha256:efd7b85f94a6f21e4932043973a7ba2613b059c4a000551892ac9f1d11f5baf3 \
    --hash=sha256:f7057c9a337546edc7973c0d3ba84ddcdf0daa14533c2065749c9075001090e6 \
    --hash=sha256:fa160448684b4e94d80416c0fa4aac48967a969efe22931448d853ada8baf926 \
    --hash=sha256:fc09d0aa354569bc501d4e787133afc08552722d3ab34836a80547331bb5d4a0
PYYAML_REQUIREMENTS
}

show_platform_hint() {
    echo "Platform compatibility: this setup was validated on Linux/glibc arm64 and amd64."
    echo "A successful download does not guarantee installation or runtime compatibility."
    echo "For another distro/Unix host, check Python version, CPU architecture, libc,"
    echo "NVIDIA driver, and Docker GPU support against the selected artifacts."
    echo "Adjust PyYAML wheel hashes in install.sh, the reviewed LiteLLM"
    echo "wheel/version/hashes in src/hls_litellm_security.py, and VLLM_DOCKER_IMAGE"
    echo "in config/env as needed for your platform. Not every Unix host is supported."
    echo "Use trusted compatible releases and verified hashes; do not bypass verification."
    echo "For a hash mismatch, stop and investigate the artifact; do not simply replace its hash."
    echo "See README.md: Platform compatibility and installation failures."
}

# Configuration
INSTALL_DIR="/opt/litellm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNTIME_MODULES=(
    hls_litellm_security.py
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

# Never start an existing environment's Python/pip before checking for malicious .pth.
existing_version=$(python3 -I -S "$SCRIPT_DIR/src/hls_litellm_security.py" check \
    "$INSTALL_DIR/venv" --allow-missing)
if [[ -n "$existing_version" ]]; then
    echo "Existing LiteLLM version: $existing_version"
fi

if ! python3 -I -S -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Error: Python 3.10 or higher is required."
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
        if python3 -m pip --isolated install --user --break-system-packages \
            --no-cache-dir --require-hashes --only-binary=:all: \
            --index-url https://pypi.org/simple \
            -r <(pyyaml_requirements); then
            echo -e "${GREEN}✓${NC} PyYAML installed via pip"
        else
            echo -e "${RED}✗${NC} Could not install PyYAML"
            show_platform_hint
            echo
            echo "  Please install manually:"
            echo "    Ubuntu/Debian: sudo apt install python3-yaml"
            echo "    Fedora/RHEL:   sudo dnf install python3-pyyaml"
            echo "    Or create a launcher venv and install PyYAML with the approved"
            echo "    version and hashes embedded in install.sh."
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
        show_platform_hint
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

install_litellm_proxy_deps() (
    local venv_path="$1"
    local mode="$2"
    local pip_bin="$venv_path/bin/pip"
    local check_output
    local requirement
    local installed_version
    local wheel_dir
    local -a yaml_wheels
    local security_check="$SCRIPT_DIR/src/hls_litellm_security.py"

    if ! installed_version=$(python3 -I -S "$security_check" check "$venv_path" --allow-missing); then
        return 1
    fi
    if [[ -n "$installed_version" ]]; then
        echo "Existing LiteLLM version: $installed_version"
    fi
    if ! requirement=$(python3 -I -S "$security_check" requirement); then
        show_platform_hint
        return 1
    fi
    if ! wheel_dir=$(mktemp -d); then
        echo -e "${RED}✗${NC} Could not create a private wheel download directory"
        return 1
    fi
    trap 'rm -f -- "$wheel_dir"/*.whl; rmdir -- "$wheel_dir"' EXIT
    echo "Downloading PyYAML with predefined SHA-256 verification before installation."
    if ! "$pip_bin" --isolated download --no-cache-dir --no-deps \
        --require-hashes --only-binary=:all: --index-url https://pypi.org/simple \
        --dest "$wheel_dir" -r <(pyyaml_requirements); then
        show_platform_hint
        return 1
    fi
    yaml_wheels=("$wheel_dir"/*.whl)
    if [[ ${#yaml_wheels[@]} -ne 1 || ! -f "${yaml_wheels[0]}" ]]; then
        echo -e "${RED}✗${NC} Expected exactly one hash-verified PyYAML wheel"
        return 1
    fi
    echo "Installing hash-verified LiteLLM and PyYAML with platform-resolved dependencies."
    if ! "$pip_bin" --isolated install --no-cache-dir --upgrade --force-reinstall \
        --index-url https://pypi.org/simple "$requirement" prisma "${yaml_wheels[0]}"; then
        show_platform_hint
        return 1
    fi
    if ! installed_version=$(python3 -I -S "$security_check" check "$venv_path" --require-pinned); then
        return 1
    fi
    echo -e "${GREEN}✓${NC} LiteLLM $installed_version installed successfully ($mode; hash-verified wheel)"

    if check_output=$("$pip_bin" check 2>&1); then
        echo -e "${GREEN}✓${NC} LiteLLM proxy dependency check passed"
    else
        echo -e "${YELLOW}⚠ Dependency check reported issues in $venv_path${NC}"
        echo "$check_output"
        show_platform_hint
        echo -e "${RED}  Recreate $venv_path and run install.sh again.${NC}"
        return 1
    fi

    return 0
)

CREATE_VENV=false
if [[ -d "$VENV_PATH" ]]; then
    echo
    echo -e "${GREEN}✓${NC} Found existing virtual environment at $VENV_PATH"
    echo
    echo "This step will:"
    echo "  • Use the existing virtual environment"
    echo "  • Install/reinstall the pinned, hash-verified litellm[proxy] version (this may take a few minutes)"
    echo "  • Configure the env file with the venv activation path"
    echo
    read -p "${TURQUOISE}Use existing virtual environment and install the pinned LiteLLM version? [y/N] ${PROMPT_RESET}" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo
        echo "Using existing virtual environment at $VENV_PATH..."
        echo
        echo "Installing/reinstalling the pinned, hash-verified litellm[proxy] version..."
        echo "This may take a few minutes. Please wait..."
        echo
        if ! install_litellm_proxy_deps "$VENV_PATH" "upgrade"; then
            echo -e "${RED}Error: Failed to install the pinned LiteLLM version. Setup aborted.${NC}"
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
    echo "  • Install the pinned, hash-verified litellm[proxy] version (this may take a few minutes)"
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
echo "   • Install:       rerun install.sh for the hash-pinned clean LiteLLM wheel"
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
echo "   If pulling or starting fails, verify image architecture and NVIDIA driver compatibility."
echo "   Choose a compatible VLLM_DOCKER_IMAGE with a verified digest; see README.md."
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
